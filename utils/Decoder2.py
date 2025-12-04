import torch
import torch.nn as nn


class TokenToImageDecoder(nn.Module):
    def __init__(self, in_channels=768, mid_channels=[256, 128], out_channels=32): 
        super(TokenToImageDecoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=mid_channels[0], out_channels=mid_channels[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=mid_channels[1], out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()  
    def forward(self, x, height, width):
        b = x.shape[0]
        x = x.view(b, -1, height, width)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x) 
        return x