import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model(cfg, ft, ift, test=False):
    if test:
        if cfg.arch == "wnet":
            return WNet(cfg.num_layers, ft, ift, use_image=cfg.use_image, unite_method=cfg.unite_method)
        elif cfg.arch == "spec_unet":
            return SpectrumUNet(cfg.num_layers, ft, ift, use_image=cfg.use_image, unite_method=cfg.unite_method)
        elif cfg.arch == "mlp":
            return MLP(cfg.in_fs, cfg.gray, ft, ift)
        else:
            raise NotImplementedError(f"Invalid cfg.model.arch: {cfg.arch}")

    if cfg.training == "all":
        return WNet(cfg.num_layers, ft, ift, use_image=cfg.use_image, unite_method=cfg.unite_method)
    elif cfg.training == "spec":
        return SpectrumUNet(cfg.num_layers, ft, ift, use_image=cfg.use_image, unite_method=cfg.unite_method)
    elif cfg.training == "refine":
        model = WNet(cfg.num_layers, ft, ift, use_image=cfg.use_image,
                     unite_method=cfg.unite_method, freeze_spec=True)
        model.spectrum_unet.load_state_dict(
            torch.load(cfg.spec_weight, map_location="cpu")
        )
        return model
    elif cfg.training == "mlp":
        return MLP(cfg.in_fs, cfg.gray, ft, ift)
    raise NotImplementedError(f"Invalid cfg.model.training: {cfg.training}")


class WNet(nn.Module):
    def __init__(self, num_layers, ft, ift, use_image=False, unite_method="cat", freeze_spec=False):
        super().__init__()
        self.spectrum_unet = SpectrumUNet(num_layers, ft, ift, use_image=use_image, unite_method=unite_method)
        self.refinement_unet = RefinementUNet(num_layers)
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

    def eval(self):
        self.training = False
        for module in self.children():
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
    img_enc_in_ch = 4
    ft_ch = 64
    dec_out_ch = 6

    def __init__(self, num_layers, ft, ift, use_image=False, unite_method="cat"):
        super().__init__()
        self.ft = ft
        self.ift = ift
        self.use_image = use_image
        self.unite_method = unite_method
    
        dec_in_ch = self.ft_ch * 8*2
        img_enc_out_ch = self.ft_ch * 8
        spec_enc_out_ch = self.ft_ch * 8
        if use_image:
            if "cat" in unite_method:
                dec_in_ch = self.ft_ch * 8*3
            if "ft" in unite_method:
                img_enc_out_ch = self.ft_ch * 4

        if self.use_image:
            self.image_encoder = BaseEncoder(num_layers, self.img_enc_in_ch, self.ft_ch, img_enc_out_ch)
        self.spectrum_encoder = BaseEncoder(num_layers, self.spec_enc_in_ch, self.ft_ch, spec_enc_out_ch)
        self.spectrum_decoder = BaseDecoder(
            num_layers, dec_in_ch, self.ft_ch, self.spec_enc_in_ch, self.dec_out_ch,
        )

    def forward(self, inp, mask, gt=None):
        inp = torch.where(
            torch.cat([mask, mask, mask], dim=1) == 1, inp,
            torch.mean(inp.view(inp.shape[0], -1), dim=1)[:, None, None, None],
        )
        inp_spec = self.ft(inp)
        mask_spec = self.ft(mask)

        features = self.spectrum_encoder(torch.cat([inp_spec, mask_spec], dim=1))
        if self.use_image:
            image_feature = self.image_encoder(torch.cat([inp, mask], dim=1))[-1]
            image_feature = self.ft(image_feature) if "ft" in self.unite_method else image_feature
            features[-1] = unite_enc_features(features[-1], image_feature, self.unite_method)
        out_spec = self.spectrum_decoder(features)
        out_img = self.ift(out_spec)

        if gt is None:
            return out_img, out_spec
        return out_img, out_spec, self.ft(gt)


class RefinementUNet(nn.Module):
    enc_in_ch = 7
    ft_ch = 64
    enc_out_ch = ft_ch * 8
    dec_out_ch = 3

    def __init__(self, num_layers):
        super().__init__()
        self.encoder = BaseEncoder(num_layers, self.enc_in_ch, self.ft_ch, self.enc_out_ch)
        self.decoder = BaseDecoder(
            num_layers, self.ft_ch*16, self.ft_ch, self.enc_in_ch, self.dec_out_ch,
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
    def __init__(self, num_layers, in_ch, ft_ch, out_ch):
        super().__init__()
        encoder = [
            ConvLayer(  in_ch,   ft_ch, 7, padding=3, bn=False),
            ConvLayer(  ft_ch, ft_ch*2, 5, padding=2),
            ConvLayer(ft_ch*2, ft_ch*4, 5, padding=2),
            ConvLayer(ft_ch*4,  out_ch, 3),
            ConvLayer( out_ch,  out_ch, 3),
        ]
        encoder.extend([ConvLayer(out_ch, out_ch, 3)]*(num_layers - 5))
        self.encoder = nn.ModuleList(encoder)
    
    def forward(self, x):
        features = [x]
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        return features


class BaseDecoder(nn.Module):
    def __init__(self, num_layers, in_ch, ft_ch, enc_in_ch, out_ch):
        super().__init__()
        decoder = [ConvLayer(in_ch, ft_ch*8, 3, stride=1, act="leaky_relu")]
        decoder.extend([ConvLayer(ft_ch*8*2, ft_ch*8, 3, stride=1, act="leaky_relu")]*(num_layers - 5))
        decoder.extend([
            ConvLayer(ft_ch*8 + ft_ch*4, ft_ch*4, 3, stride=1, act="leaky_relu"),
            ConvLayer(ft_ch*4 + ft_ch*2, ft_ch*2, 3, stride=1, act="leaky_relu"),
            ConvLayer(ft_ch*2 +   ft_ch,   ft_ch, 3, stride=1, act="leaky_relu"),
            ConvLayer(ft_ch + enc_in_ch,  out_ch, 3, stride=1, act=None, bn=False, bias=True),
        ])
        self.decoder = nn.ModuleList(decoder)

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


def unite_enc_features(spec_feat, img_feat, unite_method):
    if "pool" in unite_method:
        d = spec_feat.shape[2]
        img_feat = F.adaptive_avg_pool2d(img_feat, 1).repeat(1, 1, d, d)

    if "cat" in unite_method:
        return torch.cat([spec_feat, img_feat], dim=1)
    elif "prod" in unite_method:
        return spec_feat * img_feat
    else:
        raise NotImplementedError()


class MLP(nn.Module):
    def __init__(self, in_fs :int, gray :bool, ft, ift):
        super().__init__()
        fs = in_fs**2 * 4 if gray else in_fs**2 * 8
        out_fs = fs//2 if gray else fs//4*3
        self.layers = nn.Sequential(
            LinearLayer(fs, fs//8),
            LinearLayer(fs//8, fs//16),
            LinearLayer(fs//16, fs//16),
            LinearLayer(fs//16, fs//8),
            LinearLayer(fs//8, out_fs, bn=False, relu=False),
        )
        self.ft = ft
        self.ift = ift

    def forward(self, inp, mask, gt=None):
        inp_spec = self.ft(inp)
        mask_spec = self.ft(mask)
        b, c, w, h = inp_spec.shape

        out_spec_flat = self.layers(
            torch.cat([
                inp_spec.view(b, c*w*h),
                mask_spec.view(b, 2*w*h),
            ], dim=1),
        )
        out_spec = out_spec_flat.view(b, c, w, h)
        out_img = self.ift(out_spec)

        if gt is None:
            return out_img, out_spec
        return out_img, out_spec, self.ft(gt)


class LinearLayer(nn.Module):
    def __init__(self, in_c :int, out_c :int, bias :bool =True,
                 bn :bool =True, relu :bool =True):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Linear(in_c, out_c, bias))
        if bn:
            self.layers.append(nn.BatchNorm1d(out_c))
        if relu:
            self.layers.append(nn.ReLU())
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)