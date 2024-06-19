import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


def get_laplacian_kernel(n_classes=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kernel = torch.zeros(3, 3, device=device)
    kernel[0, 1], kernel[1, 0], kernel[1, 2], kernel[2, 1] = 0.25, 0.25, 0.25, 0.25
    kernel[1, 1] = -1
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return kernel.repeat(n_classes, 1, 1, 1)


Laplacian_kernel = get_laplacian_kernel()


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

            # postprocess = True
            # if postprocess:
            #     mask_pred = torch.softmax(mask_pred, dim=1).argmax(dim=1, keepdim=True).float()
            #     for i in range(100):
            #         mask_pred = mask_pred + 0.1 * (F.conv2d(mask_pred, Laplacian_kernel, padding=1, groups=1))
            #     mask_pred = (mask_pred > 0.5).to(torch.int64)
            #     mask_pred = mask_pred.squeeze(1)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                # mask_pred = F.one_hot(mask_pred, net.n_classes).permute(0, 3, 1, 2).float() # if postprocess = True
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches
