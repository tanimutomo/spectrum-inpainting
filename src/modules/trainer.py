import torch
from tqdm import tqdm

from modules.misc import AverageMeter


class Trainer(object):
    def __init__(self, device, model, experiment):
        self.device = device
        self.model = model
        self.experiment = experiment

    def prepare_training(self, train_loader, test_loader,
                         train_mask_tf, test_mask_tf, criterion, optimizer):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_mask_tf = train_mask_tf
        self.test_mask_tf = test_mask_tf
        self.criterion = criterion
        self.optimizer = optimizer

    def training(self, begin_iter :int, max_iter :int, img_interval :int, test_interval :int):
        print("Start Training...")
        for itr, (img, mask) in enumerate(self.train_loader, begin_iter):
            self.experiment.iter = itr
            loss_dict, save_inp = self.train_one(img, mask)
            self.experiment.report("train", loss_dict)

            if itr % img_interval == 0:
                self.experiment.save_image(save_inp, f"train_{itr}.png")
            
            if itr % test_interval == 0:
                loss_meters, save_inp = self.test(self.test_loader, self.test_mask_tf)
                self.experiment.report("test", loss_meters)
                self.experiment.save_image(save_inp, f"test_{itr}.png")

                self.experiment.save_ckpt(self.model, self.optimizer)

            if itr == max_iter:
                print("End Training.")
                return

    def train_one(self, gt, mask) -> dict:
        self.model.train()

        gt = gt.to(self.device)
        mask = mask.to(self.device)
        mask = self.train_mask_tf(mask)
        inp = gt*mask

        out_img, out_spec, gt_spec = self.model(inp, mask, gt=gt)
        loss_dict = self.criterion(
            inp, mask, gt, out_img, out_spec, gt_spec,
        )
        loss = sum(list(loss_dict.values()))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict['total'] = loss
        out_comp = mask * inp + (1 - mask) * out_img
        mask_ = torch.cat([mask]*out_img.shape[1], dim=1)
        return (
            {k: v.item() for k, v in loss_dict.items() if isinstance(v, torch.Tensor)},
            torch.stack([inp[0], mask_[0], out_img[0], out_comp[0], gt[0]],
                        dim=0).cpu().detach(),
        )

    def test(self, test_loader, test_mask_tf) -> dict:
        loss_meters = {name: AverageMeter() for name in 
                       ["spec", "valid", "hole", "perc", "style", "tv", "total"]}
        self.model.eval()
        with torch.no_grad():
            with tqdm(test_loader, ncols=80, leave=False) as pbar:
                for itr, (gt, mask) in enumerate(pbar):
                    gt = gt.to(self.device)
                    mask = mask.to(self.device)
                    mask = self.train_mask_tf(mask)
                    inp = gt*mask

                    out_img, out_spec, gt_spec = self.model(inp, mask, gt=gt)
                    loss_dict = self.criterion(
                        inp, mask, gt, out_img, out_spec, gt_spec,
                    )
                    loss_dict['total'] = sum(list(loss_dict.values()))
                    for name, loss in loss_dict.items():
                        if isinstance(loss, torch.Tensor):
                            loss = loss.item()
                        loss_meters[name].update(loss)

                    pbar.set_postfix_str(f'loss={loss_dict["total"].item():.4f}')

                    break

        out_comp = mask * inp + (1 - mask) * out_img
        mask_ = torch.cat([mask]*out_img.shape[1], dim=1)
        return (
            loss_meters,
            torch.stack([inp[0], mask_[0], out_img[0], out_comp[0], gt[0]],
                        dim=0).cpu().detach(),
        )
