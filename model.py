import torch
import torch.nn as nn
from res2resblocks import Bottle2neckX
import matplotlib.pyplot as plt

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


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
        plt.figure(figsize=(10, 5))
        imshow(x.cpu().data[0], title="Output")
        plt.show()
        print(x.shape)        
        out = self.conv0(x)
        out = self.relu(self.conv1(out))
        out = self.residual_blocks(out)
        out = self.conv2(out)
        out = self.conv3(out)
        print(out.shape)
        out += residual
        plt.figure(figsize=(10, 5))
        imshow(out.cpu().data[0], title="Output")
        plt.show()
        return out
