import torch
import torch.nn as nn

from mimic.networks.FeatureCompressor import LinearFeatureCompressor
from mimic.networks.char_encoding import DataGeneratorText as DataGeneratorText_CharEnc
from mimic.networks.char_encoding import FeatureExtractorText as FeatureExtractorText_CharEnc
from mimic.networks.word_encoding import DataGeneratorText as DataGeneratorText_WordEnc
from mimic.networks.word_encoding.mmvae_text_enc import FeatureExtractorText as FeatureExtractorText_WordEnc


class EncoderText(nn.Module):
    def __init__(self, flags, style_dim):
        super(EncoderText, self).__init__()
        self.args = flags
        if flags.text_encoding == 'char':
            self.feature_extractor = FeatureExtractorText_CharEnc(flags)
        elif flags.text_encoding == 'word':
            self.feature_extractor = FeatureExtractorText_WordEnc(flags)
        self.feature_compressor = LinearFeatureCompressor(5 * flags.DIM_text,
                                                          style_dim,
                                                          flags.class_dim)

    def forward(self, x_text):
        # d_model must be divisible by nhead
        # text_in = nn.functional.one_hot(x_text.to(torch.int64), num_classes=self.args.vocab_size)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=x_text.shape[-1], nhead=8)
        # transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)
        # h_text = transformer_encoder(text_in)
        # todo is this better?
        h_text = self.feature_extractor(x_text)
        if self.feature_compressor.style_mu and self.feature_compressor.style_logvar:
            mu_style, logvar_style, mu_content, logvar_content = self.feature_compressor(h_text)
            return mu_content, logvar_content, mu_style, logvar_style
        else:
            mu_content, logvar_content = self.feature_compressor(h_text)
            return mu_content, logvar_content


class DecoderText(nn.Module):
    def __init__(self, flags, style_dim):
        super(DecoderText, self).__init__()
        self.flags = flags
        self.feature_generator = nn.Linear(style_dim + flags.class_dim,
                                           5 * flags.DIM_text, bias=True)
        if flags.text_encoding == 'char':
            self.text_generator = DataGeneratorText_CharEnc(flags)
        elif flags.text_encoding == 'word':
            self.text_generator = DataGeneratorText_WordEnc(flags)
            # self.text_generator = Dec(flags)

    def forward(self, z_style, z_content):
        if self.flags.factorized_representation:
            z = torch.cat((z_style, z_content), dim=1).squeeze(-1)
            # z.shape = [100, 64]
        else:
            z = z_content
        text_feat_hat = self.feature_generator(z)
        text_feat_hat = text_feat_hat.unsqueeze(-1)
        # predict in batches to spare GPU memory
        if text_feat_hat.shape[0] > self.flags.batch_size:
            dl = torch.utils.data.DataLoader(text_feat_hat, batch_size=self.flags.batch_size)
            text_hat = torch.Tensor().to(self.flags.device)
            for batch in dl:
                text_hat = torch.cat(tensors=(text_hat, self.text_generator(batch)))
        else:
            text_hat = self.text_generator(text_feat_hat)
        text_hat = text_hat.transpose(-2, -1)
        return [text_hat]
