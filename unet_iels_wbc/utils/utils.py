import numpy as np
import cv2

def add_noise_to_label(label, num_classes, scale=2, keep_prop=0.9):
    """
    Add noise to labels.
    """
    shape = label.shape
    low_shape = (shape[0] // scale, shape[1] // scale)

    np.random.seed()
    noise = np.random.randint(0, num_classes, low_shape)
    noise_up = cv2.resize(noise, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_NEAREST)
    
    #the random mask is fixed
    np.random.seed(0)
    mask = np.floor(keep_prop + np.random.rand(*low_shape))
    mask_up = cv2.resize(mask, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_NEAREST)

    noised_label = mask_up * label + (1 - mask_up) * noise_up
    
    return noised_label

