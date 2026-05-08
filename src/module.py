import torch
import torch.nn as nn
import torch.nn.init as init

class ChurnModule(nn.Module):
    def __init__(self):
        super(ChurnModule, self).__init__()
        self.layer1 = nn.Linear(13, 30)
        self.layer2 = nn.Linear(30, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.actv1 = nn.LeakyReLU(-0.1)
        self.layer3 = nn.Linear(1024, 50)
        self.bn2 = nn.BatchNorm1d(50)
        self.layer4 = nn.Linear(50, 1024)

        self.actv2 = nn.ReLU()
        self.layer5 = nn.Linear(1024, 1)
        self.final = nn.Sigmoid()
        
        init.kaiming_normal_(self.layer1.weight)
        init.kaiming_normal_(self.layer2.weight)
        init.kaiming_normal_(self.layer3.weight)
        init.kaiming_normal_(self.layer4.weight)
        init.kaiming_normal_(self.layer5.weight)
        
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.actv1(x)
        x = self.layer2(x)
        x = self.actv1(x)
        x = self.bn1(x)

        x = self.layer3(x)
        x = self.actv1(x)
        x = self.bn2(x)
        x = self.layer4(x)
        
        x = self.actv1(x)
        x = self.layer5(x)
        score = self.final(x)
        return score
    