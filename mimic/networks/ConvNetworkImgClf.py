import torch.nn as nn

from mimic.networks.FeatureExtractorImg import make_res_block_feature_extractor
from typing import Protocol


class ClfImgProto(Protocol):
    image_channels: int
    img_size: int


class ClfImg(nn.Module):
    def __init__(self, flags: ClfImgProto, labels, a=2.0, b=0.3):
        super(ClfImg, self).__init__()
        self.flags = flags
        self.labels = labels
        self.a = a
        self.b = b
        modules = []
        self.conv1 = nn.Conv2d(flags.image_channels, 128,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               dilation=1,
                               bias=False)
        self.resblock_1 = make_res_block_feature_extractor(128, 256, kernelsize=4, stride=2,
                                                           padding=1, dilation=1, a_val=a, b_val=b)
        self.resblock_2 = make_res_block_feature_extractor(256, 384, kernelsize=4, stride=2,
                                                           padding=1, dilation=1, a_val=self.a, b_val=self.b)
        self.resblock_3 = make_res_block_feature_extractor(384, 512, kernelsize=4, stride=2,
                                                           padding=1, dilation=1, a_val=self.a, b_val=self.b)
        if flags.img_size == 64:
            self.resblock_4 = make_res_block_feature_extractor(512, 640, kernelsize=4, stride=2,
                                                               padding=0, dilation=1, a_val=self.a, b_val=self.b)
        elif flags.img_size == 128:
            self.resblock_4 = make_res_block_feature_extractor(512, 640, kernelsize=4, stride=2,
                                                               padding=1, dilation=1, a_val=self.a, b_val=self.b)
            self.resblock_5 = make_res_block_feature_extractor(640, 640, kernelsize=4, stride=2,
                                                               padding=0, dilation=1, a_val=self.a, b_val=self.b)
        elif flags.img_size == 256:
            self.resblock_4 = make_res_block_feature_extractor(512, 576, kernelsize=4, stride=2,
                                                               padding=1, dilation=1, a_val=self.a, b_val=self.b)
            self.resblock_5 = make_res_block_feature_extractor(576, 640, kernelsize=4, stride=2,
                                                               padding=1, dilation=1, a_val=self.a, b_val=self.b)
            self.resblock_6 = make_res_block_feature_extractor(640, 640, kernelsize=4, stride=2,
                                                               padding=0, dilation=1, a_val=self.a, b_val=self.b)
        else:
            NotImplementedError('please choose a different img size... exit')

        self.enc = nn.Sequential(*modules)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.linear = nn.Linear(in_features=640, out_features=len(self.labels), bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_img):
        """
        x_img: tensor of size [bs, 1, img_size, img_size]
        Example:
            x_img: torch.Size([50, 1, 128, 128])
            conv1: torch.Size([50, 128, 64, 64])
            resblock_1: torch.Size([50, 256, 32, 32])
            resblock_2: torch.Size([50, 384, 16, 16])
            resblock_3: torch.Size([50, 512, 8, 8])
            resblock_4: torch.Size([50, 640, 4, 4])
            resblock_5: torch.Size([50, 640, 4, 4])
            torch.Size([50, 640])
            torch.Size([50, 3])
            torch.Size([50, 3])
        """
        # encoding:
        out = self.conv1(x_img)
        out = self.resblock_1(out)
        out = self.resblock_2(out)
        out = self.resblock_3(out)
        out = self.resblock_4(out)
        if self.flags.img_size == 128:
            out = self.resblock_5(out)
        elif self.flags.img_size == 256:
            out = self.resblock_5(out)
            out = self.resblock_6(out)
        # out.shape: [bs, 640, 1, 1]
        h = out.view(out.size(0), -1)
        h = self.linear(h)
        out = self.sigmoid(h)
        return out

    def get_activations(self, x_img):
        return self.feature_extractor(x_img)
