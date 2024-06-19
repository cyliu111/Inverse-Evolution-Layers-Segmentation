import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from utils.dice_score import multiclass_dice_coeff, dice_coeff

def evaluate(net, val_loader, device, required_iels=False):
    net.eval()
    num_val_batches = len(val_loader)
    dice_score = 0
    for batch in val_loader:
        image, mask_true = batch['image'], batch['mask']
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            # predict the mask
            mask_pred = net(image, required_iels)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

                f1 = f1_score(mask_true.cpu().numpy().reshape(-1), mask_pred.cpu().numpy().reshape(-1))    

           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score

    return dice_score / num_val_batches, f1
