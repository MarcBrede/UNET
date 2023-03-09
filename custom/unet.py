import torch
from torch import nn
import torch.nn.functional as F

FIRST_LVL_CHANNEL = 32
SECOND_LVL_CHANNEL = 64
THIRD_LVL_CHANNEL = 128
FOURTH_LVL_CHANNEL = 256
FIFTH_LVL_CHANNEL = 512

class UNET(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()
        self.conv1 = Conv_Block(in_channels, FIRST_LVL_CHANNEL)
        self.down1 = DownBlock(FIRST_LVL_CHANNEL, SECOND_LVL_CHANNEL)
        self.down2 = DownBlock(SECOND_LVL_CHANNEL, THIRD_LVL_CHANNEL)
        self.down3 = DownBlock(THIRD_LVL_CHANNEL, FOURTH_LVL_CHANNEL)
        self.down4 = DownBlock(FOURTH_LVL_CHANNEL, FIFTH_LVL_CHANNEL)

        self.up4 = UpBlock(FIFTH_LVL_CHANNEL+FOURTH_LVL_CHANNEL, FOURTH_LVL_CHANNEL)
        self.up3 = UpBlock(FOURTH_LVL_CHANNEL+THIRD_LVL_CHANNEL, THIRD_LVL_CHANNEL)
        self.up2 = UpBlock(THIRD_LVL_CHANNEL+SECOND_LVL_CHANNEL, SECOND_LVL_CHANNEL)
        self.up1 = UpBlock(SECOND_LVL_CHANNEL+FIRST_LVL_CHANNEL, FIRST_LVL_CHANNEL)
        self.conv2 = Conv_Block(FIRST_LVL_CHANNEL, FIRST_LVL_CHANNEL)
        self.final_conv = nn.Conv2d(FIRST_LVL_CHANNEL, num_classes, kernel_size=1)
        self.final_activation = nn.Sigmoid()
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.conv2(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv_block(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Conv_Block(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = Conv_Block(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)