import torch
from torch import nn

class conv3_dis(torch.nn.Module):
    def __init__(self, channels=3, dataset=None):
        super().__init__()
        
        if 'mnist' in dataset:
            stride = 1
        else:
            stride = 2
        #filters = [128, 256, 512]
        self.main_module = nn.Sequential(

            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=stride, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=4, stride=1, padding=0)
        )


    def forward(self, x):
        x = self.main_module(x)
        return x
