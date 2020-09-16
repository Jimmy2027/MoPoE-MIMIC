import torch
import torch.nn as nn

from mimic.networks.FeatureExtractorImg import make_res_block_feature_extractor

class ClfImg(nn.Module):
    def __init__(self, flags, labels, a=2.0, b=0.3):
        super(ClfImg, self).__init__();
        self.flags = flags;
        self.labels = labels;
        self.a = a;
        self.b = b;
        modules = [];
        modules.append(nn.Conv2d(flags.image_channels, 128,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              dilation=1,
                              bias=False))
        modules.append(make_res_block_feature_extractor(128, 256, kernelsize=4, stride=2,
            padding=1, dilation=1, a_val=a, b_val=b));
        modules.append(make_res_block_feature_extractor(256, 384, kernelsize=4, stride=2,
            padding=1, dilation=1, a_val=self.a, b_val=self.b))
        modules.append(make_res_block_feature_extractor(384, 512, kernelsize=4, stride=2,
            padding=1, dilation=1, a_val=self.a, b_val=self.b));
        if flags.img_size == 64:
            modules.append(make_res_block_feature_extractor(512, 640, kernelsize=4, stride=2,
                padding=0, dilation=1, a_val=self.a, b_val=self.b));
        elif flags.img_size == 128:
            modules.append(make_res_block_feature_extractor(512, 640, kernelsize=4, stride=2,
                padding=1, dilation=1, a_val=self.a, b_val=self.b));
            modules.append(make_res_block_feature_extractor(640, 640, kernelsize=4, stride=2,
                padding=0, dilation=1, a_val=self.a, b_val=self.b));
        else:
            print('please choose a different img size..exit')
        self.enc = nn.Sequential(*modules);
        self.dropout = nn.Dropout(p=0.5, inplace=False);
        self.linear = nn.Linear(in_features=640, out_features=len(self.labels), bias=True);
        self.sigmoid = nn.Sigmoid();

    def forward(self, x_img):
        h = self.enc(x_img);
        h = h.view(h.size(0), -1);
        h = self.linear(h);
        out = self.sigmoid(h)
        return out;

    def get_activations(self, x_img):
        h = self.feature_extractor(x_img);
        return h;
