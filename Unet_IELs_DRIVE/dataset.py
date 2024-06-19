import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pylab as plt


class DSB2018Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, train=True, transform=None):
        self.image_dir = image_dir
        self.img_id = os.listdir(image_dir)
        self.mask_dir = mask_dir
        self.train = train
        self.transform = transform

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

        return {'image': image, 'mask': mask / 255}


class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.transform = transform

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image.astype(np.float32)
        image = image / 255.0  ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.int64)

        if self.transform:
            image, mask = self.transform(image, mask)

        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        mask = mask.astype(np.int64)
        mask = torch.from_numpy(mask)

        return {'image': image, 'mask': mask}

    def __len__(self):
        return self.n_samples
