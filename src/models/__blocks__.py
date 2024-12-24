import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            SiLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale


class SPP(nn.Module):
    """Spatial Pyramid Pooling as used in YOLOv3-SPP."""

    def __init__(self):
        super(SPP, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        return torch.cat([x, x1, x2, x3], dim=1)

class conv_block(nn.Module):
    def __init__(self,in_channels, out_channels, stride=2):
        super(conv_block, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            SiLU()
        )
        
    def forward(self, x):
        return self.block1(x)

# Just a 3x3 conv block without downsampling
class extra_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(extra_conv_block, self).__init__()
        self.block1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        SiLU()
        )
    
    def forward(self, x):
        return self.block1(x)

    def forward(self, x):
        x = self.layer1(x)  # 32-ch
        x = self.layer2(x)  # 64-ch
        x = self.layer3(x)  # 128-ch
        x = self.se1(x)
        x = self.layer4(x)  # 128-ch
        x = self.layer5(x)  # 256-ch
        x = self.se2(x)
        x = self.layer6(x)  # 256-ch
        x = self.layer7(x)  # 512-ch
        x = self.se3(x)
        x = self.layer8(x)  # 512-ch
        x = self.layer9(x)  # 512-ch (downsample to final size)

        # Apply SPP
        x = self.spp(x)  # 512*4 = 2048-ch
        x = self.reduce_conv(x)  # Back to 512-ch

        x = self.pred(x)
        return x

    def count_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total_params}")
