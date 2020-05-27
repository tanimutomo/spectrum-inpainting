from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def get_loss(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    return InpaintingLoss(
        cfg["spec"], cfg["valid"], cfg["hole"], cfg["perc"], cfg["style"], cfg["tv"],
    )


class InpaintingLoss(nn.Module):
    def __init__(self, spec, valid, hole, perc, style, tv):
        super().__init__()
        self.extractor = VGG16FeatureExtractor()
        self.spec_criterion = SpectrumLoss(spec["cut_idx"], spec["norm"], spec["coef"])
        self.valid_criterion = NormLoss(valid["norm"], valid["coef"])
        self.hole_criterion = NormLoss(hole["norm"], hole["coef"])
        self.tv_criterion = TotalVariationLoss(tv["coef"])
        self.perc_style_criterion = PercStyleLoss(
            perc["norm"], perc["coef"], style["norm"], style["coef"],
        )

    def forward(self, inp, mask, gt, out_img, out_spec, gt_spec) -> dict:
        comp = mask * inp + (1 - mask) * out_img
        loss_dict = dict(spec=0.0, valid=0.0, hole=0.0,
                         tv=0.0, perc=0.0, style=0.0)

        loss_dict["spec"] = self.spec_criterion(out_spec, gt_spec)
        loss_dict["valid"] = self.valid_criterion(mask * out_img, mask * gt)
        loss_dict["hole"] = self.hole_criterion((1-mask) * out_img, (1-mask) * gt)
        loss_dict["tv"] = self.tv_criterion(comp, mask)
        loss_dict["perc"], loss_dict["style"] = \
            self.perc_style_criterion(out_img, comp, gt)

        return loss_dict


class PercStyleLoss(nn.Module):
    def __init__(self, perc_norm, perc_coef, style_norm, style_coef):
        super().__init__()
        self.extractor = VGG16FeatureExtractor()
        self.perc_criterion = NormLoss(perc_norm, perc_coef)
        self.style_criterion = NormLoss(style_norm, style_coef)
        self.perc, self.style = perc_coef == 0, style_coef == 0

    def forward(self, out, comp, gt):
        if not self.perc and not self.style:
            return 0.0, 0.0

        feats_out = self.extractor(out)
        feats_comp = self.extractor(comp)
        feats_gt = self.extractor(gt)

        loss_perc, loss_style = 0, 0
        for i in range(len(feats_gt)):
            if self.perc:
                loss_perc += self.perc_criterion(feats_out[i], feats_gt[i])
                loss_perc += self.perc_criterion(feats_comp[i], feats_gt[i])
            if self.style:
                loss_style += self.style_criterion(self.gram_matrix(feats_out[i]),
                                                   self.gram_matrix(feats_gt[i]))
                loss_style += self.style_criterion(self.gram_matrix(feats_comp[i]),
                                                   self.gram_matrix(feats_gt[i]))
        return loss_perc, loss_style

    def gram_matrix(self, feat):
        (b, ch, h, w) = feat.size()
        feat = feat.view(b, ch, h * w)
        feat_t = feat.transpose(1, 2)
        gram = torch.bmm(feat, feat_t) / (ch * h * w)
        return gram


# The network of extracting the feature for perceptual and style loss
class VGG16FeatureExtractor(nn.Module):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        normalization = Normalization(self.MEAN, self.STD)
        # Define the each feature exractor
        self.enc_1 = nn.Sequential(normalization, *vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{}'.format(i+1)).parameters():
                param.requires_grad = False

    def forward(self, inp):
        feature_maps = [inp]
        for i in range(3):
            feature_map = getattr(self, 'enc_{}'.format(i+1))(feature_maps[-1])
            feature_maps.append(feature_map)
        return feature_maps[1:]


# Normalization Layer for VGG
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, inp):
        # normalize img
        if self.mean.type() != inp.type():
            self.mean = self.mean.to(inp)
            self.std = self.std.to(inp)
        return (inp - self.mean) / self.std


class SpectrumLoss(nn.Module):
    def __init__(self, cut_idx :int, norm :int, coef :float):
        super().__init__()
        self.cut_idx = cut_idx
        self.criterion = NormLoss(norm, coef)

    def forward(self, output, target):
        if not self.cut_idx:
            return self.criterion(output, target)
        ci = self.cut_idx
        _, _, h, w = output.shape
        return self.criterion(
            output[..., h//2-(ci-1):h//2+ci, w//2-(ci-1):w//2+ci],
            target[..., h//2-(ci-1):h//2+ci, w//2-(ci-1):w//2+ci],
        )


class NormLoss(nn.Module):
    def __init__(self, norm :int, coef :float):
        super().__init__()
        if norm == 1:
            self.criterion = nn.L1Loss()
        elif norm == 2:
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError("norm should be 0 or 1.")
        self.coef = coef

    def forward(self, output, target):
        if not self.coef:
            return 0.0
        return self.criterion(output, target) * self.coef


class TotalVariationLoss(nn.Module):
    def __init__(self, coef):
        super().__init__()
        self.coef = coef

    def forward(self, image, mask):
        if not self.coef: return 0.0
        hole_mask = 1 - mask
        dilated_holes = self._dilation_holes(hole_mask)
        colomns_in_Pset = dilated_holes[:, :, :, 1:] * dilated_holes[:, :, :, :-1]
        rows_in_Pset = dilated_holes[:, :, 1:, :] * dilated_holes[:, :, :-1:, :]
        loss = torch.mean(torch.abs(colomns_in_Pset*(
                    image[:, :, :, 1:] - image[:, :, :, :-1]))) + \
            torch.mean(torch.abs(rows_in_Pset*(
                    image[:, :, :1, :] - image[:, :, -1:, :])))
        return loss

    def _dilation_holes(self, hole_mask):
        b, ch, h, w = hole_mask.shape
        dilation_conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False).to(hole_mask)
        torch.nn.init.constant_(dilation_conv.weight, 1.0)
        with torch.no_grad():
            output_mask = dilation_conv(hole_mask)
        updated_holes = output_mask != 0
        return updated_holes.float()

