import torch 
import torch.nn as nn
import torchvision

class CNN_Model(nn.Module): 
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            #First Layer
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), 
            nn.ReLU(), 
            nn.MaxPool2d(2) ,
            #Second Layer
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.MaxPool2d(2), 
            #Final Layer 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.MaxPool2d(2) 
        )

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1), 
            nn.Linear(64*4*4, 128),
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(128, 10)  
        )

    def forward(self, x): 
        x = self.features(x)
        x = self.classifier(x) 
        return x 