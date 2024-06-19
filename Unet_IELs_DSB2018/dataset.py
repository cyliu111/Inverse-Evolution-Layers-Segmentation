import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pylab as plt
import torch.nn.functional as F


def get_laplacian_kernel(n_classes=1):
    device = torch.device('cpu')
    kernel = torch.zeros(3, 3, device=device)
    kernel[0, 1], kernel[1, 0], kernel[1, 2], kernel[2, 1] = 0.25, 0.25, 0.25, 0.25
    kernel[1, 1] = -1
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return kernel.repeat(n_classes, 1, 1, 1)


class DSB2018Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, train=True, transform=None):
        self.image_dir = image_dir
        self.img_id = os.listdir(image_dir)
        self.mask_dir = mask_dir
        self.train = train
        self.transform = transform
        self.Laplacian_kernel = get_laplacian_kernel()
        self.preprocess = False

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        if self.train:
            img_dir = os.path.join(self.image_dir, self.img_id[idx], 'image.png')
            mask_dir = os.path.join(self.mask_dir, self.img_id[idx], 'mask.png')
            image = Image.open(img_dir).convert('RGB')
            mask = Image.open(mask_dir).convert('L')
        else:
            img_dir = os.path.join(self.root_dir, self.img_id[idx], 'image.png')
            image = Image.open(img_dir).convert('RGB')
            return {'image': image}
            # size = (img.shape[0],img.shape[1])  # (Height, Weidth)

        if self.transform:
            image, mask = self.transform(image, mask)

        mask = (mask / 255).unsqueeze(0).unsqueeze(0)
        if self.preprocess:
            for i in range(100):
                mask = mask + 0.1 * (F.conv2d(mask, self.Laplacian_kernel, padding=1, groups=1))
        mask = (mask > 0.5).float()
        mask = mask.squeeze(0).squeeze(0)
        return {'image': image, 'mask': mask}
