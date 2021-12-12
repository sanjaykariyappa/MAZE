import torch.nn as nn
import torch.nn.functional as F
import torch


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class conv3_gen(nn.Module):
    def __init__(self, z_dim, start_dim=8, out_channels=3):
        super(conv3_gen, self).__init__()

        self.linear = nn.Linear(z_dim, 128 * start_dim ** 2)
        self.flatten = View((-1, 128, start_dim, start_dim))
        self.bn0 = nn.BatchNorm2d(128)

        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.up2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(64, out_channels, 3, stride=1, padding=1)

        self.bn3 = nn.BatchNorm2d(out_channels, affine=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.flatten(x)
        x = self.bn0(x)

        x = self.up1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.up2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x_pre = self.bn3(x)
        x = self.tanh(x_pre)
        return x, x_pre
