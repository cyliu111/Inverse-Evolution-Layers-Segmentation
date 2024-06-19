""" Full assembly of the parts to form the complete network """
import torch
from torch import nn
from .parts import *
import torch.nn.functional as F
import cv2
import copy


def get_ones_kernel(r_out):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    r_out = 10
    return torch.ones(1, 1, 2 * r_out + 1, 2 * r_out + 1).to(device=device) / ((2 * r_out + 1) ** 2)


def get_laplacian_kernel(n_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kernel = torch.zeros(3, 3, device=device)
    kernel[0, 1], kernel[1, 0], kernel[1, 2], kernel[2, 1] = 0.25, 0.25, 0.25, 0.25
    kernel[1, 1] = -1
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return kernel.repeat(n_classes, 1, 1, 1)


# def get_gradient_kernel(n_classes, dim='x'):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     kernel = torch.zeros(3,3, device = device)
#     if dim == 'x':
#       kernel[1,0], kernel[1,1] = -1, 1
#     else:
#       kernel[0,1], kernel[1,1] = -1, 1
#     kernel = kernel.unsqueeze(0).unsqueeze(0)
#     return  kernel.repeat(n_classes, 1, 1, 1)

def get_gradient_kernel(n_classes, dim='x'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kernel = torch.zeros(3, 3, device=device)
    if dim == 'x':
        kernel[1, 0], kernel[1, 2] = -1, 1
    else:
        kernel[0, 1], kernel[2, 1] = -1, 1
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
        self.kernel_x = get_gradient_kernel(self.n_classes, dim='x')
        self.kernel_y = get_gradient_kernel(self.n_classes, dim='y')
        self.compute_gnorm = Gradient_Norm()
        self.kernel_ones = get_ones_kernel(r_out=5)
        self.kernel_ones2 = get_ones_kernel(r_out=10)
        self.kernel_ones3 = get_ones_kernel(r_out=15)

    # def iels(self, x):
    #     #diffusion
    #     for i in range(self.iels_num):
    #         x = x - self.iels_dt * (F.conv2d(x, self.Laplacian_kernel, padding=1, groups=self.n_classes))
    #     return x

    # def iels(self, x):
    #     gradient_penalty_x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), self.kernel_x, groups=self.n_classes)
    #     gradient_penalty_y = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), self.kernel_y, groups=self.n_classes)
    #     a = torch.sqrt(torch.square(gradient_penalty_x)+torch.square(gradient_penalty_y)) + 1e-8  
    #     for i in range(self.iels_num):
    #         x = x + self.iels_dt * a *(F.conv2d(F.pad(gradient_penalty_x/a, (1, 1, 1, 1), mode='replicate'),\
    #          self.kernel_x, groups=self.n_classes)+\
    #         F.conv2d(F.pad(gradient_penalty_y/a, (1, 1, 1, 1), mode='replicate'), self.kernel_y, groups=self.n_classes))
    #     return x

    # def iels(self, x): 
    #     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     # mask = torch.zeros(x.size(), device = device)
    #     # mask[:,:,1:-1,1:-1] = 1
    #     for i in range(self.iels_num):
    #         # padded_x = F.pad(x, (1, 1, 1, 1), mode='constant')
    #         gradient_x = F.conv2d(x, self.kernel_x, padding=1, groups=self.n_classes)
    #         gradient_y = F.conv2d(x, self.kernel_y, padding=1, groups=self.n_classes)
    #         a = torch.sqrt(torch.square(gradient_x)+torch.square(gradient_y)+1e-12) 
    #         # x = x + 0*a
    #         x = x + self.iels_dt*a*F.conv2d(x, self.Laplacian_kernel, padding=1, groups=self.n_classes)
    #         # x = x + self.iels_dt *F.normalize(mask*a, p=2, dim=[2,3], eps=1e-4)* F.conv2d(x, self.Laplacian_kernel, padding=1, groups=self.n_classes)
    #     return x

    def average(self, x, nonezero_map):
        b = nonezero_map.sum(dim=(2, 3), keepdim=True)
        a = (x * nonezero_map).sum(dim=(2, 3), keepdim=True)
        c = torch.div(a, b)
        return x - c

    # def iels(self, x):
    #     for i in range(self.iels_num):
    #         x[:,1:,:,:] = x[:,1:,:,:] + self.iels_dt*self.average(x[:,1:,:,:], nonezero_map=1-(x.argmax(dim=1,keepdim=True)==1).float())
    #     return x

    # def compute_dx(self, x):
    #       y = F.conv2d(x, self.Laplacian_kernel, padding=1, groups=self.n_classes)
    #       dx = F.relu(y, inplace=False)
    #       return dx

    # def iels(self, x):
    #     #convex
    #     for i in range(self.iels_num):
    #         target_class = 1
    #         nonezero_map = (x.argmax(dim=1, keepdim=True)!=0).float()
    #         laplacian = F.conv2d(nonezero_map, self.Laplacian_kernel[0:1, :, :, :], padding=1)>0.5
    #         non_convex_map = 1 - nonezero_map + ((laplacian>=-0.5) & (laplacian<0)).float()
    #         y = F.conv2d(x[:, target_class:target_class+1, :, :], self.Laplacian_kernel[1:2, :, :, :], padding=1)
    #         dx = F.relu(y, inplace=False)
    #         # dx = self.compute_dx(x)
    #         x[:, target_class:target_class+1, :, :] = x[:, target_class:target_class+1, :, :] - self.iels_dt*torch.mul(non_convex_map, dx)
    #     return x

    def dilate_ones(self, input_tensor, radius):
        # 创建卷积核，所有元素都为1
        kernel_size = 2 * radius + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(input_tensor.device)

        # 使用卷积操作对输入张量进行膨胀操作
        dilated_tensor = F.conv2d(input_tensor, kernel, padding=radius)

        # 将取值大于0的位置设为1
        dilated_tensor = (dilated_tensor > 0).float()

        return dilated_tensor

    def sharp(self, x, nonezero_map):
        for i in range(5):
            x = x - F.relu(0.1 * nonezero_map * (x - 5)) \
                + F.relu(0.1 * (1 - nonezero_map) * (-5 - x))
        return x

    def curvature(self, x):
        result = torch.zeros_like(x)
        non_zero_indices = torch.nonzero(x)

        # 对非零元素求1/X，并将结果存储在result中
        result[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2], non_zero_indices[:, 3]] \
            = 1.0 / x[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2], non_zero_indices[:, 3]]

        return result

    # def iels(self, x):
    #     # convex
    #     target_class = 1
    #     target_region_summed = torch.zeros_like(x[:, target_class:target_class + 1, :, :])
    #     r = 1
    #     kernel = torch.ones(1, 1, 2 * r + 1, 2 * r + 1).to(device=x.device) / ((2 * r + 1) ** 2 - 1)
    #     kernel[:, :, r, r] = -1
    #     for i in range(self.iels_num):
    #         x_idx = x[:, target_class:target_class + 1, :, :]
    #         nonezero_map = (x.argmax(dim=1, keepdim=True) != 0).float()
    #         laplacian = F.conv2d(nonezero_map, kernel, padding=r)
    #         target_region = ((laplacian > -0.5 + 2/((2 * r + 1) ** 2 - 1)) & (laplacian < 0)).float()
    #         target_region_dilated = self.dilate_ones(target_region, radius=1)
    #         # non_convex_map = ((1 - nonezero_map + laplacian_dilated)>0).float()
    #         # non_convex_map = ((1 - nonezero_map + target_region_dilated)>0).float()
    #         # non_convex_map = 1 - nonezero_map + laplacian_dilated
    #         # y = F.conv2d(x_idx, self.Laplacian_kernel[0:1, :, :, :], padding=1)
    #         # dx = F.relu(y, inplace=False)
    #         # for j in range(2):
    #         # dx = F.conv2d(x_idx - x[:, 0:1, :, :], self.Laplacian_kernel[0:1, :, :, :], padding=1)
    #         target_region_summed = (target_region_summed + target_region_dilated * (1 - nonezero_map) > 0).float()
    #         # dx = self.average(x_idx, nonezero_map=nonezero_map)
    #         # x_idx = x_idx - self.iels_dt * torch.mul(target_region_summed, dx)
    #         # x_idx = x_idx - self.iels_dt*dx
    #         dx = self.average(x[:, target_class:target_class + 1, :, :] - x[:, 0:1, :, :],
    #                           nonezero_map=target_region_summed)
    #         dx = F.relu(dx)
    #         # x = x.clone()
    #         # x[:, target_class:target_class + 1, :, :] = x_idx
    #         # x[:, target_class:target_class+1, :, :] = self.sharp(x[:, target_class:target_class+1, :, :],
    #         #                            nonezero_map=(x.argmax(dim=1, keepdim=True)==target_class).float())
    #         # nonezero_map2 = nonezero_map - target_region
    #         # laplacian2 = F.conv2d(nonezero_map2, kernel, padding=r)
    #         x[:, target_class:target_class + 1, :, :] = x[:, target_class:target_class + 1, :, :] - self.iels_dt * torch.mul(
    #             target_region_summed + target_region_dilated * nonezero_map, dx)
    #     return x

    def iels(self, x):
        # convex
        target_class = 1
        target_region_summed = torch.zeros_like(x[:, target_class:target_class + 1, :, :])
        # r = 1
        # kernel = torch.ones(1, 1, 2 * r + 1, 2 * r + 1).to(device=x.device) / ((2 * r + 1) ** 2 - 1)
        # kernel[:, :, r, r] = -1

        # r_out = 10
        # kernel_ones = torch.ones(1, 1, 2 * r_out + 1, 2 * r_out + 1).to(device=x.device) / ((2 * r_out + 1) ** 2)
        #
        # r_out2 = 15
        # kernel_ones2 = torch.ones(1, 1, 2 * r_out2 + 1, 2 * r_out2 + 1).to(device=x.device) / ((2 * r_out2 + 1) ** 2)
        #
        # r_out3 = 5
        # kernel_ones3 = torch.ones(1, 1, 2 * r_out3 + 1, 2 * r_out3 + 1).to(device=x.device) / ((2 * r_out3 + 1) ** 2)
        r_out, r_out2, r_out3 = self.kernel_ones.size(-1) // 2, self.kernel_ones2.size(-1) // 2, self.kernel_ones3.size(-1) // 2
        for i in range(self.iels_num):
            x_idx = x[:, target_class:target_class + 1, :, :]
            nonezero_map = (x.argmax(dim=1, keepdim=True) != 0).float()
            # laplacian = F.conv2d(nonezero_map, kernel, padding=r)
            target_region_out = (1 - nonezero_map) * (
                (((F.conv2d(nonezero_map, self.kernel_ones, padding=r_out) >= 0.5)
                  + (F.conv2d(nonezero_map, self.kernel_ones2, padding=r_out2) >= 0.5)
                  + (F.conv2d(nonezero_map, self.kernel_ones3, padding=r_out3) >= 0.5)) > 0).float())
            # target_region_in = ((laplacian > -0.5 + 1/((2 * r + 1) ** 2 - 1)) & (laplacian<0)).float()\
            #           *self.dilate_ones(target_region_out, radius=2)
            target_region_in = self.dilate_ones(target_region_out, radius=3) * nonezero_map
            target_region = target_region_in + target_region_out
            # target_region_dilated = self.dilate_ones(target_region, radius=1)
            dx = self.compute_gnorm(x_idx - x[:, 0:1, :, :])
            target_region_summed = (target_region_summed + target_region > 0).float()
            x[:, target_class:target_class + 1, :, :] = x[:, target_class:target_class + 1, :,
                                                        :] - self.iels_dt * torch.mul(target_region_summed, dx)
            # x[:, target_class:target_class+1, :, :] = self.sharp(x[:, target_class:target_class+1, :, :],
            #                            nonezero_map=(x.argmax(dim=1, keepdim=True)==target_class).float())
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
        # x = -self.logsoftmax(x)
        if required_iels:
            x = self.iels(x)
        return x
