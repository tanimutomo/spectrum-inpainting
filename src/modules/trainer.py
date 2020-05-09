

class Trainer(object):
    def __init__(self, device, model, experiment):
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
            loss_dict, output = train_one(inp, mask, gt)
            if itr % (max_iter//100) == 0:
                self.experiment.report("train", loss_dict)
                self.experiment.save_image(
                    torch.stack([inp[0], mask[0], output[0], gt[0]], dim=0),
                    f"train_{itr}.png"
                )
            
            if itr % test_interval == 0:
                loss_dict, output = self.test(self.test_loader)
                self.experiment.report("test", loss_dict)
                self.experiment.save_image(
                    torch.stack([inp[0], mask[0], output[0], gt[0]], dim=0),
                    f"test_{itr}.png"
                )

                self.experiemnt.save_ckpt(self.model, self.optimizer)

    def train_one(self, inp, mask, gt) -> dict:
        # set the model to training mode
        self.model.train()

        inp = inp.to(self.device)
        mask = mask.to(self.device)
        gt = gt.to(self.device)

        out, _ = self.model(inp, mask)
        loss_dict = self.criterion(inp, mask, out, gt)
        loss = sum(lise(loss_dict.values()))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict['total'] = loss
        return (
            {k: v.item() for k, v in loss_dict.items()},
            (mask * inp + (1 - mask) * out).cpu().detach(),
        )

    def test(self, test_loader) -> dict:
        self.model.eval()
        with torch.no_grad():
            for itr, (inp, mask, gt) in enumerate(test_loader):
                # set the model to training mode

                inp = inp.to(self.device)
                mask = mask.to(self.device)
                gt = gt.to(self.device)

                out, _ = self.model(inp, mask)
                loss_dict = self.criterion(inp, mask, out, gt)
                loss = sum(lise(loss_dict.values()))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_dict['total'] = loss
                return (
                    {k: v.item() for k, v in loss_dict.items()},
                    (mask * inp + (1 - mask) * out).cpu().detach(),
                )
