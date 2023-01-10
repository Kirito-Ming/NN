""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_layer(nn.Module):
    def __init__(self, inChannel, outChannel) -> None:
        super().__init__()
        self.convNet = nn.Sequential(
            nn.Conv2d(inChannel,outChannel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace=False)
        )
    def forward(self, x):
        return self.convNet(x)

class down_block(nn.Module):
    def __init__(self, inChannel, outChannel, midChannel) -> None:
        super().__init__()
        self.downNet = nn.Sequential(
            nn.MaxPool2d(2),
            conv_layer(inChannel,midChannel),
            conv_layer(midChannel,outChannel)
        )
    def forward(self, x):
        return self.downNet(x)

class up_block(nn.Module):
    def __init__(self, inChannel, outChannel, midChannel) -> None:
        super().__init__()
        self.upNet = nn.Sequential(
            conv_layer(inChannel,midChannel),
            conv_layer(midChannel,outChannel)
        )
    def forward(self, x, feature_map, mode):
        up = F.interpolate(x, scale_factor=2, mode=mode)
        x = self.upNet(up)
        return torch.cat((x,feature_map),dim=1)

class input_block(nn.Module):
    def __init__(self, inChannel, outChannel, midChannel) -> None:
        super().__init__()
        self.inputNet = nn.Sequential(
            conv_layer(inChannel,midChannel),
            conv_layer(midChannel,outChannel)
        )
    def forward(self, x):
        return self.inputNet(x)

class output_block(nn.Module):
    def __init__(self, inChannel, midChannel, shuffle) -> None:
        super().__init__()
        self.outputNet = nn.Sequential(
            conv_layer(inChannel,midChannel),
            nn.PixelShuffle(shuffle)
        )
    def forward(self, x):
        return self.outputNet(x)