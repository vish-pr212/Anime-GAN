import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100, feature_g=64):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_dim,feature_g*16,4,1,0),
            nn.BatchNorm2d(feature_g*16),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g*16,feature_g*8,4,2,1),
            nn.BatchNorm2d(feature_g*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g*8,feature_g*4,4,2,1),
            nn.BatchNorm2d(feature_g*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g*4,feature_g*2,4,2,1),
            nn.BatchNorm2d(feature_g*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g*2,3,4,2,1),
            nn.Tanh()
        )

    def forward (self,x):
        return self.net(x)