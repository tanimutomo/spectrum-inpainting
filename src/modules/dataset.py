from glob import glob
from itertools import repeat
import os
import random

from hydra.utils import get_original_cwd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_dataloader(cfg, train :bool):
    data_root = os.path.join(get_original_cwd(), cfg.root)
    img_dir = cfg.train_img_dir if train else cfg.test_img_dir
    mask_dir = cfg.train_mask_dir if train else cfg.test_mask_dir

    img_trainsform = transforms.ToTensor()
    mask_transform = transforms.ToTensor()
    dataset = Places2(data_root, img_dir, mask_dir,
                      img_trainsform, mask_transform, train=train)
    data_loader = DataLoader(dataset, batch_size=cfg.batch_size,
                             shuffle=train, num_workers=8)
    if train:
        data_loader = _repeater(data_loader)

    return data_loader


def _repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data


class Places2(Dataset):
    def __init__(self, root, img_dir, mask_dir, img_transform=None,
                 mask_transform=None, train=True):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        self.img_paths = glob(f"{root}/{img_dir}/**/*.jpg", recursive=True)
        self.mask_paths = glob(f"{root}/{mask_dir}/*.png", recursive=True)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = self._load_img(self.img_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[
            random.randint(0, len(self.mask_paths) - 1)
        ]).convert("L")

        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return img, mask

    def _load_img(self, path):
        """
        For dealing with the error of loading image which is occured by the loaded image has no data.
        """
        try:
            img = Image.open(path)
        except:
            extension = path.split('.')[-1]
            for i in range(10):
                new_path = path.split('.')[0][:-1] + str(i) + '.' + extension
                try:
                    img = Image.open(new_path)
                    break
                except:
                    continue
        return img
