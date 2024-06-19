""" Full assembly of the parts to form the complete network """
import torch
from torch import nn
from .parts import *
import torch.nn.functional as F
import cv2
import copy


def get_laplacian_kernel(n_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kernel = torch.zeros(3, 3, device=device)
    kernel[0, 1], kernel[1, 0], kernel[1, 2], kernel[2, 1] = 0.25, 0.25, 0.25, 0.25
    kernel[1, 1] = -1
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return kernel.repeat(n_classes, 1, 1, 1)


class UNet_with_IELs(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, iels_dt=0.1, iels_num=30):
        super(UNet_with_IELs, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.Laplacian_kernel = get_laplacian_kernel(n_classes)
        self.iels_dt = iels_dt
        self.iels_num = iels_num

    def iels(self, x):
        for i in range(self.iels_num):
            x = x - self.iels_dt * (F.conv2d(x, self.Laplacian_kernel, padding=1, groups=self.n_classes))
        return x

    def forward(self, x, required_iels):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        if required_iels:
            x = self.iels(x)
        return x
