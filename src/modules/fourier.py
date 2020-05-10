import torch
import torch.nn as nn


class FourierTransform(object):
    def __init__(self):
        pass

    def __call__(self, x):
        z = torch.fft(torch.stack([x, x], dim=-1), 2)
        z = zshift(z)
        return batch_standadize(z[..., 0])


class InverseFourierTransform(object):
    def __init__(self):
        pass

    def __call__(self, z):
        z = torch.stack([z, z], dim=-1)
        z = zshift(z)
        x = torch.ifft(z, 2)[..., 0]
        return batch_standadize(x)


def zshift(z :torch.FloatTensor) -> torch.FloatTensor:
    assert z.ndim == 5 and z.shape[-1] == 2 and z.shape[-2] == z.shape[-3]
    resol = z.shape[-2]
    return torch.cat([
        torch.cat([
            z[..., resol//2:, resol//2:, :], # bottom right
            z[..., resol//2:, :resol//2, :], # bottom left
        ], dim=-2),
        torch.cat([
            z[..., :resol//2, resol//2:, :], # top right
            z[..., :resol//2, :resol//2, :], # top left
        ], dim=-2),
    ], dim=-3)


def batch_standadize(x):
    mn, _ = x.view(x.shape[0], -1).min(dim=1)
    mx, _ = x.view(x.shape[0], -1).max(dim=1)
    return (x - mn[:, None, None, None]) / (mx - mn)[:, None, None, None]