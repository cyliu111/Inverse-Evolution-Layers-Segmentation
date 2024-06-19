import logging
from os import listdir
from os.path import splitext, join
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .utils import *


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        self.id = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        self.ids = list(set(self.id))
        self.ids.sort()
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    # resize
    def preprocess(pil_img, img_size, is_mask):
        newW, newH = int(img_size[0]), int(img_size[1])
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))
            img_ndarray = np.asarray(img_ndarray, np.float32)
        else:
            img_ndarray = np.asarray(img_ndarray / 127, np.int64)
        return img_ndarray

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask = Image.open(self.masks_dir / (name + '.png')).convert('L')
        img = Image.open(self.masks_dir / (name + '.bmp')).convert('RGB')

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        mask = np.array(mask) // 127
        mask = Image.fromarray(mask)
        if self.transform:
            img, mask = self.transform(img, mask)

        return {'image': img, 'mask': mask}
