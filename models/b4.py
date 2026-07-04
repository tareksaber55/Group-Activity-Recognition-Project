
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

class Baseline4(nn.Module):
    def __init__(self,fine_tune_all = True,num_classes = 8):
        super(Baseline4,self).__init__()
        self.cnn = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        in_features = self.cnn.fc.in_features
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        if not fine_tune_all:
            for param in self.cnn.parameters():
                param.requires_grad = False
        self.lstm = nn.LSTM(input_size=in_features,
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
        B,F,C,H,W = x.shape
        x = x.view(B*F,C,H,W)
        x = self.cnn(x)
        x = x.view(B,F,-1)
        x , _ = self.lstm(x)
        x = x[:,-1,:]
        x = self.classifier(x)
        return x
    
# Input → (B, F, C, H, W)
# Flatten frames → (B*F, C, H, W)
# CNN features → (B*F, 2048, 1, 1)
# Reshape sequence → (B, F, 2048)
# LSTM output → (B, F, hidden_size)
# Take last timestep → (B, hidden_size)
# Classifier → (B, 8)
