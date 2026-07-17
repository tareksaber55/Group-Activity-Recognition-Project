
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


class B5PlayerClassifier(nn.Module):
    def __init__(self,backbone = None , num_classes = 9):
        super(B5PlayerClassifier,self).__init__()
        if backbone:
            self.cnn = nn.Sequential(*list(backbone.model.children())[:-1])
        else:
            self.cnn = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        for param in self.cnn.parameters():
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
    
    def train(self,mode=True):
        super().train(mode)
        self.cnn.eval()
        return self

    def forward(self,x):
        B,F,P,C,H,W = x.shape
        x = x.view(B*F*P,C,H,W)
        x = self.cnn(x)  # (B*F*P , 2048 , 1 , 1)
        x = torch.flatten(x,1) # (B*F*P , 2048)
        x = x.view(B,F,P,2048)
        x = x.permute(0,2,1,3)
        x = x.reshape(B*P,F,2048)
        x,_ = self.lstm(x)
        x = x[:,-1,:]
        x = self.classifier(x)
        x = x.view(B,P,-1)
        return x


 
class B5GroupClassifier(nn.Module):
    def __init__(self,backbone ,num_classes = 8):
        super(B5GroupClassifier,self).__init__()
        self.cnn = backbone.cnn
        self.lstm = backbone.lstm
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )
    def train(self,mode=True):
        super().train(mode)
        self.cnn.eval()
        return self
    

    def forward(self,x):
        B,F,P,C,H,W = x.shape
        x = x.view(B*F*P,C,H,W)
        x = self.cnn(x)  # (B*F*P , 2048 , 1 , 1)
        x = torch.flatten(x,1) # (B*F*P , 2048)
        x = x.view(B,F,P,2048)
        x = x.permute(0,2,1,3)
        x = x.reshape(B*P,F,2048)
        x,_ = self.lstm(x)
        x = x[:,-1,:]
        x = x.view(B,P,-1)
        x,_ = torch.max(x,dim=1) 
        x = self.classifier(x)
        return x

