import kornia.augmentation as K
import torch
import torch.nn as nn


def get_mask_transform(train :bool):
    if train:
        return nn.Sequential(
            K.RandomResizedCrop((256, 256)),
            K.RandomVerticalFlip(),
            K.RandomHorizontalFlip(),
            Binarize(),
        )
    else:
        return nn.Sequential(
            torch.nn.Upsample(size=256),
            Inverse(),
            Binarize(),
        )


class Binarize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.where(x > 0.5, torch.ones_like(x), torch.zeros_like(x))


class Inverse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1.0 - x