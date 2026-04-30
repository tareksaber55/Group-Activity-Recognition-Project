
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

class Baseline1(nn.Module):
    def __init__(self):
        super(Baseline1,self).__init__()
        self.model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512,8)
        )
    def forward(self,x):
        return self.model(x)
    
