import os

import hydra
from hydra.utils import get_original_cwd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from modules.experiment import ExperimentController
from modules.trainer import Trainer
from modules.dataset import get_dataloader
from modules.transform import get_mask_transform
from modules.model import get_model
from modules.loss import get_loss
from modules.fourier import (
    FourierTransform,
    InverseFourierTransform,
)
from modules.misc import set_seed


@hydra.main(config_path="config/test.yaml")
def main(cfg):
    is_config_valid(cfg)
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Device
    device = torch.device(f'cuda:{cfg.gpu_id}')

    # model
    model = get_model(
        cfg.model,
        FourierTransform(),
        InverseFourierTransform(),
        test=True,
    )
    model.load_state_dict(torch.load(os.path.join(
        get_original_cwd(), cfg.weight_path,
    ), map_location="cpu"))
    model.to(device)
    model.eval()

    # load inputs
    gt = Image.open(os.path.join(get_original_cwd(), cfg.input.img)).convert("RGB")
    mask = Image.open(os.path.join(get_original_cwd(), cfg.input.mask)).convert("L")
    gt = TF.to_tensor(gt)
    mask = TF.to_tensor(mask)
    inp = gt * mask

    # predict and save
    out, _, _ = model(
        inp.unsqueeze(0).to(device),
        mask.unsqueeze(0).to(device),
        gt=gt.unsqueeze(0).to(device),
    )
    out = out.squeeze().cpu().detach()
    out = torch.clamp(out, 0, 1)
    TF.to_pil_image(out).save("output_raw.png")
    TF.to_pil_image(mask * gt + (1 - mask) * out).save("output.png")


def is_config_valid(cfg):
    print(cfg.pretty())


if __name__ == "__main__":
    main()
