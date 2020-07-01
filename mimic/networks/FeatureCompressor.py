
import torch.nn as nn

class LinearFeatureCompressor(nn.Module):
    def __init__(self, in_channels, out_channels_style, out_channels_content):
        super(LinearFeatureCompressor, self).__init__();
        self.style_mu = nn.Linear(in_channels, out_channels_style, bias=True);
        self.style_logvar = nn.Linear(in_channels, out_channels_style, bias=True);
        self.content_mu = nn.Linear(in_channels, out_channels_content, bias=True);
        self.content_logvar = nn.Linear(in_channels, out_channels_content, bias=True);

    def forward(self, feats):
        feats = feats.view(feats.size(0), -1);
        mu_style, logvar_style = self.style_mu(feats), self.style_logvar(feats);
        mu_content, logvar_content = self.content_mu(feats), self.content_logvar(feats);
        return mu_style, logvar_style, mu_content, logvar_content;
