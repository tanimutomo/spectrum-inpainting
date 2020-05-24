from comet_ml import Experiment

import os

import hydra
from hydra.utils import get_original_cwd
import torch
import torch.nn as nn

from modules.experiment import ExperimentController
from modules.trainer import Trainer
from modules.dataset import get_dataloader
from modules.model import get_model
from modules.loss import get_loss
from modules.fourier import (
    FourierTransform,
    InverseFourierTransform,
)
from modules.misc import set_seed


@hydra.main(config_path="config/train.yaml")
def main(cfg):
    is_config_valid(cfg)
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Experiment and Set comet
    experiment = ExperimentController(cfg)
    if cfg.comet.use:
        experiment.set_comet(os.path.join(get_original_cwd(), '.comet'))

    # Device
    device = torch.device(f'cuda:{cfg.gpu_ids[0]}')

    train_loader = get_dataloader(cfg.data, train=True)
    test_loader = get_dataloader(cfg.data, train=False)

    # resume training
    last_iter, model_sd, optimizer_sd = experiment.load_ckpt()

    # model
    model = get_model(cfg.model, FourierTransform(), InverseFourierTransform())
    if model_sd is not None:
        model.load_state_dict(model_sd)
    model.to(device)
    if len(cfg.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=cfg.gpu_ids)

    # optimizer
    if cfg.optim.method == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.optim.initial_lr,
            weight_decay=cfg.optim.weight_decay,
        )
    else:
        raise NotImplementedError(f"{cfg.optim.method} is not implemented.")
    if optimizer_sd:
        optimizer.load_state_dict(optimizer_sd)

    # Loss fucntion
    criterion = get_loss(cfg.loss).to(device)

    trainer = Trainer(device, model, experiment)
    trainer.prepare_training(train_loader, test_loader, criterion, optimizer)
    trainer.training(last_iter+1, cfg.optim.max_iter,
                     cfg.img_interval, cfg.test_interval)

    experiment.save_model(trainer.model)


def is_config_valid(cfg):
    if cfg.model.spec_weight:
        cfg.model.spec_weight = os.path.join(get_original_cwd(),
                                             cfg.model.spec_weight)
    print(cfg.pretty())


if __name__ == "__main__":
    main()