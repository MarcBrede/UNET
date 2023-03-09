import torch
import torch.nn as nn

KERNEL_SIZE = 3
PADDING = 1

class SimpleNet(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_size) -> None:
        super().__init__()
        self.conv1 = Conv_Block(start=True, end=False, in_channels=in_channels, num_classes=num_classes, hidden_size=hidden_size)
        self.conv2 = Conv_Block(start=False, end=False, in_channels=in_channels, num_classes=num_classes, hidden_size=hidden_size)
        self.conv3 = Conv_Block(start=False, end=False, in_channels=in_channels, num_classes=num_classes, hidden_size=hidden_size)
        self.conv4 = Conv_Block(start=False, end=False, in_channels=in_channels, num_classes=num_classes, hidden_size=hidden_size)
        self.conv5 = Conv_Block(start=False, end=True, in_channels=in_channels, num_classes=num_classes, hidden_size=hidden_size)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class UNET_SMALL(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_size) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, hidden_size=hidden_size)
        self.decoder = Decoder(hidden_size=hidden_size, num_classes=num_classes)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_size) -> None:
        super().__init__()
        self.conv1 = Conv_Block(start=True, end=False, in_channels=in_channels, num_classes=None, hidden_size=hidden_size)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv_Block(start=False, end=False, in_channels=hidden_size, num_classes=None, hidden_size=hidden_size)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = Conv_Block(start=False, end=False, in_channels=hidden_size, num_classes=None, hidden_size=hidden_size)
        
    def forward(self, x):
        x = self.conv1(x)
        # x = self.pool1(x)
        x = self.conv2(x)
        # x = self.pool2(x)
        x = self.conv3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_size, num_classes) -> None:
        super().__init__()
        # self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv1 = Conv_Block(start=False, end=False, in_channels=None, num_classes=num_classes, hidden_size=hidden_size)
        # self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.conv3 = Conv_Block(start=False, end=True, in_channels=None, num_classes=num_classes, hidden_size=hidden_size)
    
    def forward(self, x):
        # x = self.upsample1(x)
        x = self.conv1(x)
        # x = self.upsample2(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class Conv_Block(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_size, start=False, end=False) -> None:
        super().__init__()
        if start:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=int(hidden_size/2), kernel_size=KERNEL_SIZE, padding=PADDING)
            self.activation1 = nn.ReLU()
            self.conv2 = nn.Conv2d(in_channels=int(hidden_size/2), out_channels=hidden_size, kernel_size=KERNEL_SIZE, padding=PADDING)
            self.activation2 = nn.ReLU()
        elif end:
            self.conv1 = nn.Conv2d(in_channels=hidden_size, out_channels=int(hidden_size/2), kernel_size=KERNEL_SIZE, padding=PADDING)
            self.activation1 = nn.ReLU()
            self.conv2 = nn.Conv2d(in_channels=int(hidden_size/2), out_channels=num_classes, kernel_size=KERNEL_SIZE, padding=PADDING)
            self.activation2 = nn.Sigmoid()
        else:
            self.conv1 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=KERNEL_SIZE, padding=PADDING)
            self.activation1 = nn.ReLU()
            self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=KERNEL_SIZE, padding=PADDING)
            self.activation2 = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        return x