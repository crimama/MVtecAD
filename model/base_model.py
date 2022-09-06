import timm 
import torch 
import torchvision 
from torch import nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model1 = timm.create_model('resnet50', pretrained=True, num_classes=88)
        
    def forward(self, x):
        x = self.model1(x)
        return x