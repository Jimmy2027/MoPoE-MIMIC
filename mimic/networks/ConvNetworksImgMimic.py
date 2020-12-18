import torch
import torch.nn as nn

from mimic.networks.DataGeneratorImg import DataGeneratorImg
from mimic.networks.FeatureCompressor import LinearFeatureCompressor
from mimic.networks.FeatureExtractorImg import FeatureExtractorImg
from mimic.networks.CheXNet import DenseNetFeatureExtractor


def get_feature_extractor_img(flags):
    if flags.feature_extractor_img == 'resnet':
        feature_extractor = FeatureExtractorImg(flags)
    elif flags.feature_extractor_img == 'densenet':
        feature_extractor = DenseNetFeatureExtractor(flags)
    else:
        raise NotImplementedError
    return feature_extractor


class EncoderImg(nn.Module):
    def __init__(self, flags, style_dim):
        super(EncoderImg, self).__init__()
        self.flags = flags
        self.feature_extractor = get_feature_extractor_img(flags)
        self.feature_compressor = LinearFeatureCompressor(5 * flags.DIM_img,
                                                          style_dim,
                                                          flags.class_dim)

    def forward(self, x_img):
        h_img = self.feature_extractor(x_img)
        if self.feature_compressor.style_mu and self.feature_compressor.style_logvar:
            mu_style, logvar_style, mu_content, logvar_content = self.feature_compressor(h_img)
            return mu_content, logvar_content, mu_style, logvar_style
        else:
            mu_content, logvar_content = self.feature_compressor(h_img)
            return mu_content, logvar_content


class DecoderImg(nn.Module):
    def __init__(self, flags, style_dim):
        super(DecoderImg, self).__init__()
        self.flags = flags
        self.feature_generator = nn.Linear(style_dim + flags.class_dim, 5 * flags.DIM_img, bias=True)
        self.img_generator = DataGeneratorImg(flags)

    def forward(self, z_style, z_content):
        if self.flags.factorized_representation:
            z = torch.cat((z_style, z_content), dim=1).squeeze(-1)
        else:
            z = z_content
        img_feat_hat = self.feature_generator(z)
        img_feat_hat = img_feat_hat.view(img_feat_hat.size(0), img_feat_hat.size(1), 1, 1)
        img_hat = self.img_generator(img_feat_hat)
        return img_hat, torch.tensor(0.75).to(z.device)
