import torch
import torch.nn as nn

from mimic.networks.DataGeneratorText import DataGeneratorText
from mimic.networks.FeatureCompressor import LinearFeatureCompressor
from mimic.networks.char_encoding.FeatureExtractorText import FeatureExtractorText


class EncoderText(nn.Module):
    def __init__(self, flags, style_dim):
        super(EncoderText, self).__init__();
        self.feature_extractor = FeatureExtractorText(flags)
        self.feature_compressor = LinearFeatureCompressor(5 * flags.DIM_text,
                                                          style_dim,
                                                          flags.class_dim)

    def forward(self, x_text):
        # d_model must be divisible by nhead
        # encoder_layer = nn.TransformerEncoderLayer(d_model=x_text.shape[-1], nhead=8)
        # transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)
        # h_text = transformer_encoder(x_text)
        # todo is this better?
        h_text = self.feature_extractor(x_text)
        if self.feature_compressor.style_mu and self.feature_compressor.style_logvar:
            mu_style, logvar_style, mu_content, logvar_content = self.feature_compressor(h_text);
            return mu_content, logvar_content, mu_style, logvar_style
        else:
            mu_content, logvar_content = self.feature_compressor(h_text)
            return mu_content, logvar_content;


class DecoderText(nn.Module):
    def __init__(self, flags, style_dim):
        super(DecoderText, self).__init__();
        self.flags = flags;
        self.feature_generator = nn.Linear(style_dim + flags.class_dim,
                                           5 * flags.DIM_text, bias=True);
        self.text_generator = DataGeneratorText(flags)

    def forward(self, z_style, z_content):
        if self.flags.factorized_representation:
            z = torch.cat((z_style, z_content), dim=1).squeeze(-1)
        else:
            z = z_content;
        text_feat_hat = self.feature_generator(z);
        text_feat_hat = text_feat_hat.unsqueeze(-1);
        text_hat = self.text_generator(text_feat_hat)
        text_hat = text_hat.transpose(-2, -1);
        return [text_hat];
