import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model(cfg, ft, ift):
    if cfg.training == "all":
        return WNet(ft, ift, use_image=cfg.use_image)
    elif cfg.training == "spec":
        return SpectrumUNet(ft, ift, use_image=cfg.use_image)
    elif cfg.training == "refine":
        model = WNet(ft, ift, use_image=cfg.use_image,
                     freeze_spec=True)
        model.spectrum_unet.load_state_dict(
            torch.load(cfg.spec_weight, map_location="cpu")
        )
        return model
    raise NotImplementedError(f"Invalid cfg.model.training: {cfg.training}")


class WNet(nn.Module):
    def __init__(self, ft, ift, use_image=False, freeze_spec=False):
        super().__init__()
        self.spectrum_unet = SpectrumUNet(ft, ift, use_image=use_image)
        self.refinement_unet = RefinementUNet()
        self.freeze_spec = freeze_spec

    def forward(self, inp, mask, gt=None):
        out_img, out_spec, gt_spec = self.spectrum_unet(inp, mask, gt)
        out_img = self.refinement_unet(out_img, inp, mask)
        return out_img, out_spec, gt_spec

    def train(self):
        self.training = True
        for module in self.children():
            module.train(True)
        if self.freeze_spec:
            for module in self.spectrum_unet.children():
                module.train(False)
        return self
    
    def parameters(self):
        if self.freeze_spec:
            named_params = self.refinement_unet.named_parameters(recurse=True)
        else:
            named_params = self.named_parameters(recurse=True)
        for _, param in named_params:
            yield param


class SpectrumUNet(nn.Module):
    spec_enc_in_ch = 8
    img_enc_in_ch = 8
    ft_ch = 64
    dec_out_ch = 6

    def __init__(self, ft, ift, use_image=False):
        super().__init__()
        self.use_image = use_image
        self.image_encoder = ImageEncoder(self.img_enc_in_ch, self.ft_ch)
        self.spectrum_encoder = SpectrumEncoder(self.spec_enc_in_ch, self.ft_ch)

        dec_in_ch = self.ft_ch * 24 if use_image else self.ft_ch * 16
        self.spectrum_decoder = SpectrumDecoder(
            dec_in_ch, self.ft_ch, self.spec_enc_in_ch, self.dec_out_ch,
        )

        self.to_spectrum = ft
        self.to_image = ift
    
    def forward(self, inp, mask, gt=None):
        inp = torch.where(
            torch.cat([mask, mask, mask], dim=1) == 1, inp,
            torch.mean(inp.view(inp.shape[0], -1), dim=1)[:, None, None, None],
        )
        inp_spec = self.to_spectrum(inp)
        mask_spec = self.to_spectrum(mask)

        features = self.spectrum_encoder(torch.cat([inp_spec, mask_spec], dim=1))
        if self.use_image:
            image_feature = self.image_encoder(torch.cat([inp, mask], dim=1))[-1]
            features[-1] = torch.cat([features[-1], image_feature], dim=1)
        out_spec = self.spectrum_decoder(features)
        out_img = self.to_image(out_spec)

        if gt is None:
            return out_img, out_spec
        return out_img, out_spec, self.to_spectrum(gt)


class RefinementUNet(nn.Module):
    enc_in_ch = 7
    ft_ch = 64
    dec_out_ch = 3

    def __init__(self):
        super().__init__()
        self.encoder = RefinementEncoder(self.enc_in_ch, self.ft_ch)
        self.decoder = RefinementDecoder(
            self.ft_ch*16, self.ft_ch, self.enc_in_ch, self.dec_out_ch,
        )
    
    def forward(self, sout, inp, mask):
        inp = torch.where(
            torch.cat([mask, mask, mask], dim=1) == 1, inp,
            torch.mean(inp.view(inp.shape[0], -1), dim=1)[:, None, None, None],
        )
        features = self.encoder(torch.cat([sout, inp, mask], dim=1))
        out = self.decoder(features)
        return out


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


class BaseDecoder(nn.Module):
    def __init__(self, in_ch, ft_ch, enc_in_ch, out_ch):
        super().__init__()
        self.decoder = nn.ModuleList([
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
        for layer in self.decoder:
            y = F.interpolate(y, scale_factor=2.0)
            y = layer(torch.cat([y, enc_features.pop()], dim=1))
        return y


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=2, padding=1,
                 dilation=1, groups=1, bn=True, bias=False, act="relu"):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Conv2d(
            in_ch, out_ch, k_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=False,
        ))
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


class ImageEncoder(BaseEncoder):
    def __init__(self, in_ch, ft_ch):
        super().__init__(in_ch, ft_ch)


class SpectrumEncoder(BaseEncoder):
    def __init__(self, in_ch, ft_ch):
        super().__init__(in_ch, ft_ch)


class RefinementEncoder(BaseEncoder):
    def __init__(self, in_ch, ft_ch):
        super().__init__(in_ch, ft_ch)


class SpectrumDecoder(BaseDecoder):
    def __init__(self, in_ch, ft_ch, enc_in_ch, out_ch):
        super().__init__(in_ch, ft_ch, enc_in_ch, out_ch)


class RefinementDecoder(BaseDecoder):
    def __init__(self, in_ch, ft_ch, enc_in_ch, out_ch):
        super().__init__(in_ch, ft_ch, enc_in_ch, out_ch)
