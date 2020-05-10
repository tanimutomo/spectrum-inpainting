import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_CHOICES = [
    "spectrum_unet"
]

def get_model(model_name, ft, ift, use_image :bool):
    if model_name == "spectrum_unet":
        return SpectrumUNet(ft, ift, use_image=use_image)
    else:
        raise NotImplementedError(f"Model choices are {MODEL_CHOICES}")


class SpectrumUNet(nn.Module):
    enc_in_ch = 4
    ft_ch = 64
    gen_out_ch = 3

    def __init__(self, fourier_transform, inverse_fourier_transform, use_image=False):
        super().__init__()
        self.use_image = use_image
        self.image_encoder = ImageEncoder(self.enc_in_ch, self.ft_ch)
        self.spectrum_encoder = SpectrumEncoder(self.enc_in_ch, self.ft_ch)

        gen_in_ch = self.ft_ch * 24 if use_image else self.ft_ch * 16
        self.spectrum_generator = SpectrumGenerator(
            gen_in_ch, self.ft_ch, self.enc_in_ch, self.gen_out_ch,
        )

        self.to_spectrum = fourier_transform
        self.to_image = inverse_fourier_transform
    
    def forward(self, inp, mask):
        features = self.spectrum_encoder(
            self.to_spectrum(torch.cat([inp, mask], dim=1))
        )
        if self.use_image:
            image_feature = self.image_encoder(
                torch.cat([inp, mask], dim=1),
            )[-1]
            features[-1] = torch.cat([features[-1], image_feature], dim=1)
        out_spectrum = self.spectrum_generator(features)
        out_image = self.to_image(out_spectrum)
        return out_spectrum, out_image


class BaseEncoder(nn.Module):
    def __init__(self, in_ch, ft_ch):
        super().__init__()
        self.encoder = nn.ModuleList([
            ConvLayer(  in_ch,   ft_ch, 7, padding=3, bn=False),
            ConvLayer(  ft_ch, ft_ch*2, 5, padding=2),
            ConvLayer(ft_ch*2, ft_ch*4, 5, padding=2),
            ConvLayer(ft_ch*4, ft_ch*8, 3),
            ConvLayer(ft_ch*8, ft_ch*8, 3),
            ConvLayer(ft_ch*8, ft_ch*8, 3),
            ConvLayer(ft_ch*8, ft_ch*8, 3),
            ConvLayer(ft_ch*8, ft_ch*8, 3),
        ])
    
    def forward(self, x):
        features = [x]
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        return features


class ImageEncoder(BaseEncoder):
    def __init__(self, in_ch, ft_ch):
        super().__init__(in_ch, ft_ch)


class SpectrumEncoder(BaseEncoder):
    def __init__(self, in_ch, ft_ch):
        super().__init__(in_ch, ft_ch)


class SpectrumGenerator(nn.Module):
    def __init__(self, in_ch, ft_ch, enc_in_ch, out_ch):
        super().__init__()
        self.generator = nn.ModuleList([
            ConvLayer(in_ch, ft_ch*8, 3, stride=1, act="leaky_relu"),
            ConvLayer(        ft_ch*8*2, ft_ch*8, 3, stride=1, act="leaky_relu"),
            ConvLayer(        ft_ch*8*2, ft_ch*8, 3, stride=1, act="leaky_relu"),
            ConvLayer(        ft_ch*8*2, ft_ch*8, 3, stride=1, act="leaky_relu"),
            ConvLayer(ft_ch*8 + ft_ch*4, ft_ch*4, 3, stride=1, act="leaky_relu"),
            ConvLayer(ft_ch*4 + ft_ch*2, ft_ch*2, 3, stride=1, act="leaky_relu"),
            ConvLayer(ft_ch*2 +   ft_ch,   ft_ch, 3, stride=1, act="leaky_relu"),
            ConvLayer(ft_ch + enc_in_ch,  out_ch, 3, stride=1, act=None, bn=False, bias=True),
        ])

    def forward(self, enc_features :list):
        y = enc_features.pop()
        for layer in self.generator:
            y = F.interpolate(y, scale_factor=2.0)
            y = layer(torch.cat([y, enc_features.pop()], dim=1))
        return y


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=2, padding=1,
                 bn=True, bias=False, act="relu"):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Conv2d(in_ch, out_ch, k_size, stride, padding, bias=False))
        if bn:
            self.layers.append(nn.BatchNorm2d(out_ch))
        if act == "relu":
            self.layers.append(nn.ReLU())
        elif act == "leaky_relu":
            self.layers.append(nn.LeakyReLU(negative_slope=0.2))
        elif act is None:
            pass
        else:
            raise NotImplementedError("activation functions should be in [relu, leaky_relu]")
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)
