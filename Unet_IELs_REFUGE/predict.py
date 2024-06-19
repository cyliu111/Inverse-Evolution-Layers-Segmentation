import torch
import torch.nn.functional as F
from utils.dice_score import multiclass_dice_coeff, dice_coeff


def predict(net, batch, device, required_iels):
    net.eval()
    dice_score = 0
    image, mask_true = batch['image'], batch['mask']
    # image = image + 50*torch.randn(image.s  hape)
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
            mask_pred2 = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_pred2[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

    net.train()

    return mask_pred, dice_score
