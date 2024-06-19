import torch
import torch.nn as nn
from res2resblocks import Bottle2neckX

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out

class RainEstimationNetwork(nn.Module):
    def __init__(self):
        super(RainEstimationNetwork, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=9, stride=1, padding=4)
        self.relu = nn.ReLU(inplace=True)
        self.residual_blocks = nn.Sequential(
            Bottle2neckX(64,64,64,4)
            # Add more blocks for a deeper model
        )
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        residual = x
        out = self.conv0(x)
        out = self.relu(self.conv1(out))
        out = self.residual_blocks(out)
        out = self.conv2(out)
        out += residual
        # out = self.conv3(out)
        return out
