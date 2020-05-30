import torch
import torch.nn as nn


class FourierTransform(object):
    def __init__(self, cutidx=None):
        self.cutidx = cutidx

    def __call__(self, x):
        z = torch.fft(torch.stack([x, x], dim=-1), 2)
        z = zshift(z)
        z = torch.cat([z[..., 0], z[..., 1]], dim=1)
        if not self.cutidx:
            return z

        b, c, h, w = z.shape
        ci = self.cutidx
        z_c = z[..., h//2-(ci-1):h//2+ci, w//2-(ci-1):w//2+ci]
        z = torch.zeros(*z.shape, device=z.device)
        z[..., h//2-(ci-1):h//2+ci, w//2-(ci-1):w//2+ci] = z_c
        return z


class InverseFourierTransform(object):
    def __init__(self, cutidx=None):
        self.cutidx = cutidx

    def __call__(self, z):
        b, c, h, w = z.shape
        if not self.cutidx:
            ci = self.cutidx
            z_c = z[..., h//2-(ci-1):h//2+ci, w//2-(ci-1):w//2+ci]
            z = torch.zeros(*z.shape, device=z.device)
            z[..., h//2-(ci-1):h//2+ci, w//2-(ci-1):w//2+ci] = z_c

        z = torch.stack([z[:, :c//2, ...], z[:, c//2:, ...]], dim=-1)
        z = zshift(z)
        x = torch.ifft(z, 2)[..., 0]
        return batch_standadize(x)


def zshift(z :torch.FloatTensor) -> torch.FloatTensor:
    assert z.ndim == 5 and z.shape[-1] == 2 and z.shape[-2] == z.shape[-3]
    resol = z.shape[-2]
    return torch.cat([
        torch.cat([
            z[..., resol//2:, resol//2:, :], # bottom right -> top left
            z[..., resol//2:, :resol//2, :], # bottom left -> top right
        ], dim=-2),
        torch.cat([
            z[..., :resol//2, resol//2:, :], # top right -> bottom left
            z[..., :resol//2, :resol//2, :], # top left -> bottom right
        ], dim=-2),
    ], dim=-3)


def batch_standadize(x):
    assert x.ndim in [4, 5]

    mn, _ = x.view(x.shape[0], -1).min(dim=1)
    mx, _ = x.view(x.shape[0], -1).max(dim=1)

    if x.ndim == 4:
        return (x - mn[:, None, None, None]) / (mx - mn)[:, None, None, None]
    return (x - mn[:, None, None, None, None]) / (mx - mn)[:, None, None, None, None]
