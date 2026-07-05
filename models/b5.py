import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

class Baseline5(nn.Module):
    def __init__(self,backbone,num_classes = 8):
        super(Baseline5,self).__init__()
        self.backbone = backbone
        self.backbone.model.fc = nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.lstm = nn.LSTM(input_size=2048,
                            hidden_size=1024,
                            num_layers=1,
                            batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )
    def forward(self,x):
        B,F,P,C,H,W = x.shape
        x = x.view(B*F,P,C,H,W)
        x = self.backbone(x)
        x,_ = torch.max(x,dim=1)
        x = x.view(B,F,-1)
        x , _ = self.lstm(x)
        x = x[:,-1,:]
        x = self.classifier(x)
        return x
    
