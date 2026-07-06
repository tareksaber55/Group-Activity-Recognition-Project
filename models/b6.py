import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

class Baseline6(nn.Module):
    def __init__(self,backbone,num_classes = 8):
        super(Baseline6,self).__init__()
        self.backbone = nn.Sequential(*list(backbone.model.children())[:-1])
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
    def forward(self, x):
        B,F,P,C,H,W = x.shape

        # Merge batch, frame and player dimensions
        x = x.view(B*F*P, C, H, W)

        # Feature extraction
        x = self.backbone(x)          # (BFP,2048,1,1)
        x = torch.flatten(x,1)        # (BFP,2048)

        # Restore dimensions
        x = x.view(B,F,P,2048)

        # Aggregate players
        x,_ = torch.max(x, dim=2)     # (B,F,2048)

        # Temporal modeling
        x,_ = self.lstm(x)            # (B,F,1024)

        # Last frame
        x = x[:,-1,:]

        return self.classifier(x)
    
