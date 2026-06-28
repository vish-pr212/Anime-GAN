import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, feature_d=64):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3,feature_d,4,2,1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_d,feature_d*2,4,2,1),
            nn.BatchNorm2d(feature_d*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_d*2,feature_d*4,4,2,1),
            nn.BatchNorm2d(feature_d*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_d*4,feature_d*8,4,2,1),
            nn.BatchNorm2d(feature_d*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_d*8,1,4,1,0),
            
        )

    def forward(self,x):
        return self.net(x).view(-1)
    