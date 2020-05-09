from comet_ml import Experiment, ExistingExperiment

import csv
import os
import shutil
import sys

import moment
import torch
from torchvision import transforms 
from torchvision.utils import save_image
import yaml


class ExperimentController(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.comet = None
        self.iter = 0

    def set_comet(self, comet_config_path):
        with open(comet_config_path) as f:
            comet = yaml.safe_load(f)

        exp_args = dict(
            project_name=self.cfg.comet.project,
            workspace=comet['workspace'],
            api_key=comet['api_key'],
            auto_param_logging=False,
            auto_metric_logging=False,
            log_env_host=True,
            parse_args=False
        )
        if self.cfg.comet.resume_key:
            exp_args['previous_experiment'] = self.cfg.comet.resume_key
            comet = ExistingExperiment(**exp_args)
        else:
            comet = Experiment(**exp_args)
            comet.set_name(self.cfg.experiment.name)

        comet.log_parameters(to_flat_dict(self.cfg.to_container()))
        if isinstance(self.cfg.comet.tags, str):
            comet.add_tag(self.cfg.comet.tags)
        else:
            comet.add_tags(self.cfg.comet.tags.to_container())
        comet.send_notification(self.cfg.experiment.name)

        print('[Experiment Name]', self.cfg.experiment.name)
        self.comet = comet

    def report(self, mode, meters, test=False):
        max_iter = self.cfg.optim.max_iter if not test else 0
        report = f'{mode.upper()} [{self.iter:d}/{max_iter:d}]  '
        for name, meter in meters.items():
            report += f'{name} = {meter.avg:.2f} / '
            log = {'iter': self.iter,
                   'ts': moment.now().format('YYYY-MMDD-HHmm-ss')}
            log[mode+name] = meter.avg
            if not self.comet:
                continue
            self.comet.log_metric(
                f'{mode}-{name}',
                meter.avg, step=self.iter
            )
        self.save_log(log)
        print(report)

    def save_log(self, log: dict):
        # save logs as csv and send to comet-ml
        mode = 'a' if os.path.exists('metrics.csv') else 'w'
        with open('metrics.csv', mode) as f:
            keys = list(log.keys())
            writer = csv.DictWriter(f, fieldnames=keys)
            if mode == 'w':
                writer.writeheader()
            writer.writerow(log)
        # send to comet-ml
        self.send_file('metrics.csv', overwrite=True)

    def send_file(self, path :str, overwrite=False):
        if self.comet:
            self.comet.log_asset(path, overwrite=overwrite, step=self.iter)

    def save_ckpt(self, model, optimizer): #, scheduler):
        checkpoint = {
            'iter': self.iter,
            'model': model.module.state_dict() \
                     if isinstance(model, torch.nn.DataParallel) \
                     else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),
        }
        torch.save(checkpoint, 'checkpoint.pth')
        self.send_file('checkpoint.pth', overwrite=True)

    def load_ckpt(self):
        if not self.cfg.comet.resume_key:
            return 0, None, None, None
        if not os.path.exists(self.cfg.comet.resume_key):
            raise ValueError(f'The specified experiment ({self.cfg.experiment.name}) is not existed.')
        ckpt = torch.load('checkpoint.pth', map_location='cpu')
        shutil.copy('checkpoint.pth', f'checkpoint.pth.bkup.{moment.now().format("YYYY-MMDD-HHmm-ss")}')
        return ckpt['iter']+1, ckpt['model'], ckpt['optimizer'] #, ckpt['scheduler']

    def save_model(self, model):
        torch.save(
            model.module.state_dict() \
                if isinstance(model, torch.nn.DataParallel) \
                else model.state_dict(),
            'model.pth'
        )
        self.send_file('model.pth')

    def save_image(self, img, filename):
        os.makedirs('images', exist_ok=True)
        path = os.path.join('images', filename)
        save_image(img, path)
        self.send_image(path)
    
    def send_image(self, path :str) -> None:
        if self.comet:
            self.comet.log_image(path, step=self.iter)


def to_flat_dict(target :dict, output :dict =dict()):
    if len(list(target.keys())) == 0: 
        return output 
    next_target = dict() 
    for k, v in target.items(): 
        if isinstance(v, dict):
            for k_, v_ in v.items():
                next_target[k+'-'+k_] = v_ 
        else: 
            output[k] = v 
    return to_flat_dict(next_target, output)