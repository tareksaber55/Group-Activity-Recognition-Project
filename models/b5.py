
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


class B5PlayerClassifier(nn.Module):
    def __init__(self,num_classes = 9):
        super(B5PlayerClassifier,self).__init__()
        self.cnn = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        lstm_in_features = self.cnn.fc.in_features
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.lstm = nn.LSTM(input_size=lstm_in_features,
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


