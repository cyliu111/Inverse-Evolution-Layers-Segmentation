""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class Gradient_Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_x = torch.tensor([[[[0., 0., 0.],
                                        [-0.5, 0., 0.5],
                                        [0., 0., 0.]]]]).to(
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.kernel_y = torch.tensor([[[[0., 0.5, 0.],
                                        [0., 0., 0.],
                                        [0., -0.5, 0.]]]]).to(
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.padding = (1, 1, 1, 1)  # 左右上下各填充1行
        self.eps = 1e-6

    def forward(self, x):
        # 使用 F.pad 进行填充
        x_padded = F.pad(x, self.padding, 'replicate')

        # 使用 F.conv2d 进行中心差分计算
        gradient_x = F.conv2d(x_padded, self.kernel_x)
        gradient_y = F.conv2d(x_padded, self.kernel_y)

        return torch.sqrt(torch.square(gradient_x) + torch.square(gradient_y) + self.eps)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
  
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels, eps= 1e-5, momentum= 0.1),
            nn.LeakyReLU(negative_slope = 1e-2,inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels, eps= 1e-5, momentum= 0.1),
            nn.LeakyReLU(negative_slope = 1e-2,inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

