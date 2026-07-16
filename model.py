import torch 
import torch.nn as nn
import torchvision

class CNN_Model: 
    def __init__(self):
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,stride=1)