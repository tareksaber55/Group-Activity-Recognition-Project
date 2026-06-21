
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models



class PlayerClassifier(nn.Module):
    def __init__(self,fine_tune_all = True,num_player_actions = 8):
        super(PlayerClassifier,self).__init__()
        self.model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        if not fine_tune_all:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features,1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,num_player_actions)
        )
    def forward(self,x):
        # batch , frames , persons , channels , width , height
        # but resnet expects 3d (channels,width,height)
        B,F,P,C,W,H = x.shape
        x = x.view(B*F*P,C,W,H)
        out = self.model(x)
        out = out.view(B, F, P, -1)
        return out
    
