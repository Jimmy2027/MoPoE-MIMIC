import torch.nn as nn

from mimic.networks.ResidualBlocks import ResidualBlock2dConv


def make_res_block_feature_extractor(in_channels, out_channels, kernelsize, stride, padding, dilation, a_val=2.0,
                                     b_val=0.3):
    downsample = None;
    if (stride != 2) or (in_channels != out_channels) or padding == 0:
        downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                             kernel_size=kernelsize,
                                             padding=padding,
                                             stride=stride,
                                             dilation=dilation),
                                   nn.BatchNorm2d(out_channels))
    layers = [];
    layers.append(
        ResidualBlock2dConv(in_channels, out_channels, kernelsize, stride, padding, dilation, downsample, a=a_val,
                            b=b_val))
    return nn.Sequential(*layers)


class FeatureExtractorImg(nn.Module):
    def __init__(self, args, a=2.0, b=0.3):
        super(FeatureExtractorImg, self).__init__();
        self.args = args;
        self.a = a;
        self.b = b;
        self.conv1 = nn.Conv2d(self.args.image_channels, self.args.DIM_img,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               dilation=1,
                               bias=False)
        self.resblock_1 = make_res_block_feature_extractor(args.DIM_img, 2 * args.DIM_img, kernelsize=4, stride=2,
                                                           padding=1, dilation=1, a_val=a, b_val=b)
        self.resblock_2 = make_res_block_feature_extractor(2 * args.DIM_img, 3 * args.DIM_img, kernelsize=4, stride=2,
                                                           padding=1, dilation=1, a_val=self.a, b_val=self.b)
        self.resblock_3 = make_res_block_feature_extractor(3 * args.DIM_img, 4 * args.DIM_img, kernelsize=4, stride=2,
                                                           padding=1, dilation=1, a_val=self.a, b_val=self.b)
        if args.img_size == 64:
            self.resblock_4 = make_res_block_feature_extractor(4 * args.DIM_img, 5 * args.DIM_img, kernelsize=4,
                                                               stride=2,
                                                               padding=0, dilation=1, a_val=self.a, b_val=self.b)
        elif args.img_size == 128:
            self.resblock_4 = make_res_block_feature_extractor(4 * args.DIM_img, 5 * args.DIM_img, kernelsize=4,
                                                               stride=2,
                                                               padding=1, dilation=1, a_val=self.a, b_val=self.b)
            self.resblock_5 = make_res_block_feature_extractor(5 * args.DIM_img, 5 * args.DIM_img, kernelsize=4,
                                                               stride=2,
                                                               padding=0, dilation=1, a_val=self.a, b_val=self.b)
        elif args.img_size == 256:
            self.resblock_4 = make_res_block_feature_extractor(4 * args.DIM_img, 5 * args.DIM_img, kernelsize=4,
                                                               stride=4,
                                                               padding=1, dilation=1, a_val=self.a, b_val=self.b)

            self.resblock_5 = make_res_block_feature_extractor(5 * args.DIM_img, 5 * args.DIM_img, kernelsize=4,
                                                               stride=2,
                                                               padding=0, dilation=1, a_val=self.a, b_val=self.b)

    def forward(self, x):
        """
        Example:
            x_shape: torch.Size([10, 1, 128, 128])
            torch.Size([10, 64, 64, 64])
            torch.Size([10, 128, 32, 32])
            torch.Size([10, 192, 16, 16])
            torch.Size([10, 256, 8, 8])
            torch.Size([10, 320, 4, 4])
            torch.Size([10, 320, 1, 1])
            torch.Size([10, 320, 1])
        """
        out = self.conv1(x)
        out = self.resblock_1(out)
        out = self.resblock_2(out)
        out = self.resblock_3(out)
        out = self.resblock_4(out)
        if self.args.img_size != 64:
                out = self.resblock_5(out)
        out = out.view(out.shape[0], out.shape[1], out.shape[2])
        return out
