import collections
import torchvision
import torch
import torchvision.transforms.functional as F
import random
import numbers
import numpy as np
from PIL import Image
# import skimage
import cv2


#  Extended Transforms for Semantic Segmentation
class ExtCompose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl):
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ExtToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type

    def __call__(self, pic, lbl):
        """
        Note that labels will not be normalized to [0, 1].
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
            lbl (PIL Image or numpy.ndarray): Label to be converted to tensor.
        Returns:
            Tensor: Converted image and label
        """
        if self.normalize:
            return F.to_tensor(pic), torch.from_numpy(np.array(lbl, dtype=self.target_type))
        else:
            return torch.from_numpy(np.array(pic, dtype=np.float32).transpose(2, 0, 1)), torch.from_numpy(
                np.array(lbl, dtype=self.target_type))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ExtResize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.abc.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        return F.resize(img, self.size, self.interpolation), F.resize(lbl, self.size, Image.NEAREST)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class add_noise_to_lbl(object):
    """add noise to label.
    """

    def __init__(self, num_classes, scale=2, keep_prop=0.8):
        self.num_classes = num_classes
        self.scale = scale
        self.keep_prop = keep_prop

    def __call__(self, img, lbl):
        label = np.array(lbl)
        shape = label.shape
        low_shape = (shape[0] // self.scale, shape[1] // self.scale)

        np.random.seed(0)
        noise = np.random.randint(0, self.num_classes, low_shape)
        noise_up = cv2.resize(noise, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_NEAREST)

        # the random mask is fixed
        # np.random.seed(0)
        mask = np.floor(self.keep_prop + np.random.rand(*low_shape))
        mask_up = cv2.resize(mask, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_NEAREST)

        noised_label = mask_up * label + (1 - mask_up) * noise_up

        return img, Image.fromarray(noised_label.astype(np.uint8))
