import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class InpaintingLoss(nn.Module):
    def __init__(self, coef, extractor,
                 fourier_transform, inverse_fourier_transform):
        super().__init__()
        self.coef = coef
        self.extractor = extractor
        self.to_spectrum = fourier_transform
        self.to_image = inverse_fourier_transform

    def forward(self, inp, mask, out_spectrum, out_image, gt) -> dict:
        # Non-hole pixels directly set to ground truth
        comp = mask * inp + (1 - mask) * out_image

        # Spectrum Loss
        gt_spectrum = self.to_spectrum(gt)
        spectrum_loss = F.l1_loss(out_spectrum, gt_spectrum)

        # Total Variation Regularization
        tv_loss = total_variation_loss(comp, mask)

        # Hole Pixel Loss
        hole_loss = F.l1_loss((1-mask) * out_image, (1-mask) * gt)

        # Valid Pixel Loss
        valid_loss = F.l1_loss(mask * out_image, mask * gt)

        # Perceptual Loss and Style Loss
        feats_out = self.extractor(out_image)
        feats_comp = self.extractor(comp)
        feats_gt = self.extractor(gt)
        perc_loss = 0.0
        style_loss = 0.0
        # Calculate the L1Loss for each feature map
        for i in range(3):
            perc_loss += F.l1_loss(feats_out[i], feats_gt[i])
            perc_loss += F.l1_loss(feats_comp[i], feats_gt[i])
            style_loss += F.l1_loss(gram_matrix(feats_out[i]),
                                    gram_matrix(feats_gt[i]))
            style_loss += F.l1_loss(gram_matrix(feats_comp[i]),
                                    gram_matrix(feats_gt[i]))

        return {
            'spectrum': spectrum_loss * self.coef.spectrum,
            'valid': valid_loss * self.coef.valid,
            'hole': hole_loss * self.coef.hole,
            'perc': perc_loss * self.coef.perc,
            'style': style_loss * self.coef.style,
            'tv': tv_loss * self.coef.tv
        }


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


# Calcurate the Gram Matrix of feature maps
def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def dialation_holes(hole_mask):
    b, ch, h, w = hole_mask.shape
    dilation_conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False).to(hole_mask)
    torch.nn.init.constant_(dilation_conv.weight, 1.0)
    with torch.no_grad():
        output_mask = dilation_conv(hole_mask)
    updated_holes = output_mask != 0
    return updated_holes.float()


def total_variation_loss(image, mask):
    hole_mask = 1 - mask
    dilated_holes = dialation_holes(hole_mask)
    colomns_in_Pset = dilated_holes[:, :, :, 1:] * dilated_holes[:, :, :, :-1]
    rows_in_Pset = dilated_holes[:, :, 1:, :] * dilated_holes[:, :, :-1:, :]
    loss = torch.mean(torch.abs(colomns_in_Pset*(
                image[:, :, :, 1:] - image[:, :, :, :-1]))) + \
           torch.mean(torch.abs(rows_in_Pset*(
                image[:, :, :1, :] - image[:, :, -1:, :])))
    return loss
