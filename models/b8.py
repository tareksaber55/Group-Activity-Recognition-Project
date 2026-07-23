import torch
import torch.nn as nn

class Baseline8(nn.Module):
    def __init__(self, player_backbone, image_backbone, num_classes=8):
        super().__init__()

        # Frozen backbones
        self.image_cnn = nn.Sequential(
            *list(image_backbone.model.children())[:-1]
        )
        self.player_cnn = player_backbone.cnn
        self.player_lstm = player_backbone.lstm

        for module in [self.image_cnn, self.player_cnn, self.player_lstm]:
            for param in module.parameters():
                param.requires_grad = False

        # Static image projection (2048 -> 512)
        self.image_proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Static player projection (2048 -> 512)
        self.player_proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Group temporal modeling
        self.lstm2 = nn.LSTM(
            input_size=3584,      # 1536 + 1536 + 512
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def train(self, mode=True):
        super().train(mode)
        self.image_cnn.eval()
        self.player_cnn.eval()
        self.player_lstm.eval()
        return self

    def forward(self, x_player, x_image):

        # x_player : (B,F,P,C,H,W)
        # x_image  : (B,F,C,H,W)

        B, F, P, C, H, W = x_player.shape

        with torch.no_grad():


            image = x_image.view(B * F, C, H, W)
            image = self.image_cnn(image)
            image = torch.flatten(image, 1)


            player = x_player.view(B * F * P, C, H, W)
            player = self.player_cnn(player)
            player = torch.flatten(player, 1)

            player = player.view(B, F, P, 2048)

            lstm_input = (
                player.permute(0, 2, 1, 3)
                      .contiguous()
                      .view(B * P, F, 2048)
            )


        player_temp, _ = self.player_lstm(lstm_input)

        player_temp = (
            player_temp.view(B, P, F, 1024)
                       .contiguous()
        )


        image = self.image_proj(image)
        image = image.view(B, F, 512)

        player_static = self.player_proj(player.view(B * F * P, 2048))
        player_static = player_static.view(B, F, P, 512)
        player_static = player_static.permute(0, 2, 1, 3)


        player_features = torch.cat(
            [player_static, player_temp],
            dim=-1
        )                               # (B,P,F,1536)


        left_features, _ = player_features[:, :6].max(dim=1)
        right_features, _ = player_features[:, 6:].max(dim=1)

        # (B,F,3072)
        player_features = torch.cat(
            [left_features, right_features],
            dim=-1
        )


        fusion = torch.cat(
            [player_features, image],
            dim=-1
        )                               # (B,F,3584)


        out, _ = self.lstm2(fusion)

        out = out[:, -1]

        return self.classifier(out)