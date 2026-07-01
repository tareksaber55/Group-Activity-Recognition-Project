
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models



class B3PlayerClassifier(nn.Module):
    def __init__(self,fine_tune_all = True,num_classes = 9):
        super(B3PlayerClassifier,self).__init__()
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
            nn.Linear(512,num_classes)
        )
    def forward(self,x):
        # batch , frames , persons , channels , width , height
        # but resnet expects 4d (batch,channels,width,height)
        B,P,C,H,W = x.shape
        x = x.view(B*P,C,H,W)
        out = self.model(x)
        out = out.view(B, P, -1)
        return out
    
class B3GroupClassifier(nn.Module):
    def __init__(self,backbone ,num_classes = 8):
        super(B3GroupClassifier,self).__init__()
        self.backbone = backbone
        self.backbone = nn.Sequential(*list(backbone.model.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )
    def forward(self,x):
        B,P,C,H,W = x.shape
        x = x.view(B*P,C,H,W)
        out = self.backbone(x)
        out = out.view(B, P, -1)
        out,_ = torch.max(out,dim=1)
        out = self.classifier(out)
        return out


