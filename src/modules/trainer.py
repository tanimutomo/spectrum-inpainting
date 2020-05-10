import torch

from modules.misc import AverageMeter


class Trainer(object):
    def __init__(self, device, model, fourier_transform, experiment):
        self.device = device
        self.model = model
        self.experiment = experiment

    def prepare_training(self, train_loader, test_loader, criterion, optimizer):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimzier = optimizer

    def training(self, begin_iter :int, max_iter :int, test_interval :int):
        for itr, (inp, mask, gt) in enumerate(self.train_loader):
            self.experiemnt.iter = itr
            loss_dict, save_inp = train_one(inp, mask, gt)
            if itr % (max_iter//100) == 0:
                self.experiment.report("train", loss_dict)
                self.experiment.save_image(save_inp, f"train_{itr}.png")
            
            if itr % test_interval == 0:
                loss_meters, save_inp = self.test(self.test_loader)
                self.experiment.report("test", loss_meters)
                self.experiment.save_image(save_inp, f"test_{itr}.png")

                self.experiemnt.save_ckpt(self.model, self.optimizer)

    def train_one(self, inp, mask, gt) -> dict:
        # set the model to training mode
        self.model.train()

        inp = inp.to(self.device)
        mask = mask.to(self.device)
        gt = gt.to(self.device)

        out_spectrum, out_image = self.model(inp, mask)
        loss_dict = self.criterion(
            inp, mask, out_spectrum, out_image, gt,
        )
        loss = sum(lise(loss_dict.values()))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict['total'] = loss
        out_comp = (mask * inp + (1 - mask) * out_image).cpu().detach()
        return (
            {k: v.item() for k, v in loss_dict.items()},
            torch.stack([inp[0], mask[0], out_comp[0], gt[0]], dim=0)
        )

    def test(self, test_loader) -> dict:
        loss_meters = {
            "spectrum": AverageMeter(),
            "valid": AverageMeter(),
            "hole": AverageMeter(),
            "perc": AverageMeter(),
            "style": AverageMeter(),
            "tv": AverageMeter(),
            "total": AverageMeter(),
        }
        self.model.eval()
        with torch.no_grad():
            for itr, (inp, mask, gt) in enumerate(test_loader):
                # set the model to training mode

                inp = inp.to(self.device)
                mask = mask.to(self.device)
                gt = gt.to(self.device)

                out_spectrum, out_image = self.model(inp, mask)
                loss_dict = self.criterion(
                    inp, mask, out_spectrum, out_image, gt
                )
                loss = sum(lise(loss_dict.values()))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_dict['total'] = loss
                for name, loss in loss_dict.items():
                    loss_meters[name].update(loss.item())

        out_comp = (mask * inp + (1 - mask) * out_image).cpu().detach()
        return loss_meters, torch.stack([inp[0], mask[0], out_comp[0], gt[0]], dim=0)
