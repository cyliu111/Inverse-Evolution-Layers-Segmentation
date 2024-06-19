import torch
from torch import nn
import torch.nn.functional as F


def get_gradient_kernel(n_classes, dim='x'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kernel = torch.zeros(3, 3, device=device)
    if dim == 'x':
        kernel[1, 0], kernel[1, 1] = -1, 1
    else:
        kernel[0, 1], kernel[1, 1] = -1, 1
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return kernel.repeat(n_classes, 1, 1, 1)


class SmoothingLoss(nn.Module):
    def __init__(self, lambda_value, n_classes):
        super(SmoothingLoss, self).__init__()
        self.lambda_value = lambda_value
        self.n_classes = n_classes
        self.kernel_x = get_gradient_kernel(self.n_classes, dim='x')
        self.kernel_y = get_gradient_kernel(self.n_classes, dim='y')

    def forward(self, pred, target):
        ce_loss = nn.CrossEntropyLoss()(pred, target)
        gradient_penalty_x = F.conv2d(pred, self.kernel_x, padding=1, groups=self.n_classes)
        gradient_penalty_y = F.conv2d(pred, self.kernel_y, padding=1, groups=self.n_classes)
        loss = ce_loss + self.lambda_value * \
               torch.mean(torch.square(gradient_penalty_x) + torch.square(gradient_penalty_y), dim=[0, 1, 2, 3])
        return loss