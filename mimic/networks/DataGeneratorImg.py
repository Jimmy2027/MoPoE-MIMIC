import torch.nn as nn

from mimic.networks.ResidualBlocks import ResidualBlock2dTransposeConv


def make_res_block_data_generator(in_channels, out_channels, kernelsize, stride, padding, o_padding, dilation,
                                  a_val=1.0, b_val=1.0):
    upsample = None
    if (kernelsize != 1 and stride != 1) or (in_channels != out_channels):
        upsample = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels,
                                                    kernel_size=kernelsize,
                                                    stride=stride,
                                                    padding=padding,
                                                    dilation=dilation,
                                                    output_padding=o_padding),
                                 nn.BatchNorm2d(out_channels))
    layers = []
    layers.append(ResidualBlock2dTransposeConv(in_channels, out_channels,
                                               kernelsize=kernelsize,
                                               stride=stride,
                                               padding=padding,
                                               dilation=dilation,
                                               o_padding=o_padding,
                                               upsample=upsample,
                                               a=a_val, b=b_val))
    return nn.Sequential(*layers)


class DataGeneratorImg(nn.Module):
    def __init__(self, args, a=2.0, b=0.3):
        super(DataGeneratorImg, self).__init__()
        self.args = args
        modules = [
            make_res_block_data_generator(5 * self.args.DIM_img,
                                          4 * self.args.DIM_img,
                                          kernelsize=4,
                                          stride=1,
                                          padding=0,
                                          dilation=1,
                                          o_padding=0,
                                          a_val=a,
                                          b_val=b,
                                          ),

            make_res_block_data_generator(4 * self.args.DIM_img,
                                          3 * self.args.DIM_img,
                                          kernelsize=4, stride=2,
                                          padding=1, dilation=1,
                                          o_padding=0, a_val=a,
                                          b_val=b),
            make_res_block_data_generator(3 * self.args.DIM_img,
                                          2 * self.args.DIM_img,
                                          kernelsize=4, stride=2,
                                          padding=1, dilation=1,
                                          o_padding=0, a_val=a,
                                          b_val=b),
            make_res_block_data_generator(2 * self.args.DIM_img,
                                          1 * self.args.DIM_img,
                                          kernelsize=4, stride=2,
                                          padding=1, dilation=1,
                                          o_padding=0, a_val=a,
                                          b_val=b)]

        if args.img_size == 128:
            modules.append(make_res_block_data_generator(1 * self.args.DIM_img,
                                                         1 * self.args.DIM_img,
                                                         kernelsize=4, stride=2,
                                                         padding=1, dilation=1,
                                                         o_padding=0, a_val=a,
                                                         b_val=b))
        if args.img_size == 256:
            modules.append(make_res_block_data_generator(1 * self.args.DIM_img,
                                                         1 * self.args.DIM_img,
                                                         kernelsize=4, stride=2,
                                                         padding=1, dilation=1,
                                                         o_padding=0, a_val=a,
                                                         b_val=b))
            modules.append(make_res_block_data_generator(1 * self.args.DIM_img,
                                                         1 * self.args.DIM_img,
                                                         kernelsize=4, stride=2,
                                                         padding=1, dilation=1,
                                                         o_padding=0, a_val=a,
                                                         b_val=b))
        modules.append(nn.ConvTranspose2d(self.args.DIM_img,
                                          self.args.image_channels,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          dilation=1,
                                          output_padding=1))
        self.generator = nn.Sequential(*modules)

    def forward(self, feats):
        """
        Example:
            feats.shape = [bs, 320, 1, 1]
        """
        return self.generator(feats)
