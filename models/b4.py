
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

class Baseline4(nn.Module):
    def __init__(self,fine_tune_all = True):
        super(Baseline4,self).__init__()
        self.cnn = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        in_features = self.cnn.fc.in_features
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        if not fine_tune_all:
            for param in self.cnn.parameters():
                param.requires_grad = False
        self.lstm = nn.LSTM(input_size=in_features,hidden_size=1024,num_layers=1,batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,8)
        )
    def forward(self,x):
        pass
    
