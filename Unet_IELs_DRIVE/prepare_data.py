import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image

# read all training data and save them to other folder
def prepare_training_data(stage1_train_src, stage1_train_dest):  
    # get imageId
    train_ids = next(os.walk(stage1_train_src))[1]

    # read training data
    X_train = []
    Y_train = []
    print('reading training data starts...')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids)):
        path = os.path.join(stage1_train_src, id_)
        dest = os.path.join(stage1_train_dest, id_)
        img = Image.open(os.path.join(path, 'images', id_ + '.png')).convert("RGB")
        mask = assemble_masks(path)
        if not os.path.exists(dest):
            os.makedirs(dest)        
        img.save(os.path.join(dest, 'image.png'))
        Image.fromarray(mask).save(os.path.join(dest, 'mask.png'))

    print('reading training data done...')

def assemble_masks(path):
    mask = None
    for i, mask_file in enumerate(next(os.walk(os.path.join(path, 'masks')))[2]):
        mask_ = Image.open(os.path.join(path, 'masks', mask_file)).convert("RGB")
        mask_ = np.asarray(mask_)
        if i == 0:
            mask = mask_
            continue
        mask = mask | mask_
    # mask = np.expand_dims(mask, axis=-1)
    return mask
      
if __name__ == '__main__':
    """ Prepare training data
    read data and overlay masks and save to destination path
    """
    stage1_train_src = './data/stage1_train'
    stage1_train_dest = './data/combined'
    
    prepare_training_data(stage1_train_src, stage1_train_dest)
