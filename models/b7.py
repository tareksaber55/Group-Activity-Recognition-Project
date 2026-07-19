import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

class Baseline7(nn.Module):
    def __init__(self,backbone,num_classes = 8):
        super(Baseline7,self).__init__()
        self.cnn = backbone.cnn
        self.lstm1 = backbone.lstm
        self.lstm2 = nn.LSTM(
            input_size=2048,
            hidden_size=1024,
            num_layers=1,
            batch_first=True
        )
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.lstm1.parameters():
            param.requires_grad = False
        self.player_proj = nn.Sequential(
            nn.Linear(3072,2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024,512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )

    def train(self,mode=True):
        super().train(mode)
        self.cnn.eval()
        return self
    

    def forward(self, x):
        B,F,P,C,H,W = x.shape

        x = x.view(B*F*P,C,H,W)

        cnn_out = self.cnn(x)
        cnn_out = torch.flatten(cnn_out,1)
        cnn_out = cnn_out.view(B,F,P,2048)

        lstm_input = cnn_out.permute(0,2,1,3)
        lstm_input = lstm_input.reshape(B*P,F,2048)

        lstm1_out,_ = self.lstm1(lstm_input)

        lstm1_out = lstm1_out.reshape(B,P,F,1024)
        lstm1_out = lstm1_out.permute(0,2,1,3)

        person_features = torch.cat([cnn_out,lstm1_out],dim=-1)

        person_features,_ = torch.max(person_features,dim=2)

        person_features = self.player_proj(person_features)

        lstm2_out,_ = self.lstm2(person_features)

        lstm2_out = lstm2_out[:,-1,:]

        return self.classifier(lstm2_out)


