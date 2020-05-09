from comet_ml import Experiment

import hydra

from modules.model import get_model()
from modules.dataset import get_dataloader()


@hydra.main(config_path="config/train.yaml")
def main(cfg):
    is_config_valid(cfg)
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Experiment and Set comet
    experiment = ExperimentController(cfg)
    if cfg.comet.project:
        experiment.set_comet(os.path.join(get_original_cwd(), '.comet'))

    # Device
    device = torch.device(f'cuda:{cfg.gpu_ids[0]}')

    train_loader = get_dataloader(
        cfg.data.data_root, cfg.data.dataset,
        cfg.data.batch_size, train=True
    )
    test_loader = get_dataloader(
        cfg.data.data_root, cfg.data.dataset,
        cfg.data.batch_size, train=True
    )

    # resume training
    last_iter, model_sd, optimizer_sd = experiment.load_ckpt()

    # model
    model = get_model()
    if model_sd is not None:
        model.load_state_dict(model_sd)
    model.to(device)
    if len(cfg.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=cfg.gpu_ids)

    # optimizer
    if cfg.optim.method == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.optim.initial_lr,
            weight_decay=cfg.optim.weight_decay,
        )
    else:
        raise NotImplementedError(f"{cfg.optim.method} is not implemented.")
    if optimizer_sd:
        optimizer.load_state_dict(optimizer_sd)

    # Loss fucntion
    criterion = InpaintingLoss(VGG16FeatureExtractor(), cfg.loss).to(device)

    trainer = Trainer(device, model, experiment)
    trainer.prepare_training(train_loader, val_loader, criterion, optimizer)
    trainer.training(last_iter+1)

    experiment.save_model(trainer.model)