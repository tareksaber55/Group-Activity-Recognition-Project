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

        self.lstm2 = nn.LSTM(
            input_size=4096,
            hidden_size=1024,
            num_layers=1,
            batch_first=True
        )

        for module in [self.image_cnn, self.player_cnn, self.player_lstm]:
            for param in module.parameters():
                param.requires_grad = False

        # Player feature projection
        self.player_proj = nn.Sequential(
            nn.Linear(6144, 3072),
            nn.BatchNorm1d(3072),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(3072, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2)
        )



        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
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

            # Image branch
            image = x_image.view(B * F, C, H, W)
            image = self.image_cnn(image)
            image = torch.flatten(image, 1)
            image = image.view(B, F, 2048)

            # Player branch
            player = x_player.view(B * F * P, C, H, W)
            player = self.player_cnn(player)
            player = torch.flatten(player, 1)
            player = player.view(B, F, P, 2048)

            lstm_input = (
                player.permute(0, 2, 1, 3)
                      .contiguous()
                      .view(B * P, F, 2048)
            )

        # Person temporal modeling 
        player_temp, _ = self.player_lstm(lstm_input)

        player_temp = (
            player_temp.view(B, P, F, 1024)
                       .permute(0, 2, 1, 3)
                       .contiguous()
        )

        # Static + Temporal player features
        player_features = torch.cat([player, player_temp],dim=-1) # (B,F,P,3072)                                  

        # Aggregate players

        left_player_features,_ = player_features[:,:,:6,:].max(dim=2) # B , F , 3072

        right_player_features,_ = player_features[:,:,6:,:].max(dim=2) # B , F , 3072

        player_features = torch.cat([left_player_features,right_player_features] , dim=2) # B , F , 6144

        player_features = player_features.view(B * F , 6144)
        player_features = self.player_proj(player_features)
        player_features = player_features.view(B, F, 2048)

        

        # image + players
        fusion = torch.cat( [image, player_features] ,dim=-1) # (B,F,4096)


        # Group temporal modeling
        out, _ = self.lstm2(fusion)

        out = out[:, -1, :]

        return self.classifier(out)




        








    