import random
from glob import glob
from itertools import repeat

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


DATASETS = [
    "places2"
]

def get_dataloader(data_root :str, dataset_name :str, batch_size :int, train :bool):
    if dataset_name == "places2":
        dataset = Places2(data_root, transforms.ToTensor(),
                          transforms.ToTensor(), train=train)
    else:
        raise NotImplementedError(f"Implemented Datasets are {DATASETS}")
    data_loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=train, num_workers=8)
    if train:
        data_loader = repeater(data_loader)
    return data_loader


def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data


class Places2(Dataset):
    def __init__(self, root, img_transform, mask_transform, train=True):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # get the list of image paths
        # img_dir = "data_256" if train else "val_256"
        img_dir = "data_256_tmp" if train else "val_256"
        mask_dir = "mask" if train else "val_mask"
        self.img_paths = glob(f"{root}/places2/{img_dir}/**/*.jpg", recursive=True)
        self.mask_paths = glob(f"{root}/places2/{mask_dir}/*.png", recursive=True)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = self._load_img(self.img_paths[index])
        img = self.img_transform(img.convert('RGB'))
        mask = Image.open(self.mask_paths[
            random.randint(0, len(self.mask_paths) - 1)
        ])
        mask = self.mask_transform(mask.convert('L'))
        return img * mask, mask, img

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