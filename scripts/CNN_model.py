# cnn for feature extraction

import torch
import torch.optim as optim
import torch.nn as nn

import torchvision.models as models


#define the simple model 
class Resize_Alexnet(nn.Module):
    def __init__(self, depth=4): # (self,input_size = (80,80), pretrained=True)
        super(Resize_Alexnet,self).__init__()
        alexnet = models.alexnet(pretrained=True)
        if depth==4:
            self.features = alexnet.features[0:9]
        elif depth==3:
            self.features = alexnet.features[0:7]
        elif depth==2:
            self.features = alexnet.features[0:4]
        elif depth==1:
            self.features = alexnet.features[0]
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
    def forward(self,x):
        x = self.features(x)
#         x = self.avgpool(x)
        return x
        

