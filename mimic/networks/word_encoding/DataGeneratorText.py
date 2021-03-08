import torch.nn as nn

from mimic.networks.ResidualBlocks import ResidualBlock1dTransposeConv


def make_res_block_decoder(in_channels, out_channels, kernelsize, stride, padding, o_padding, dilation, a_val=2.0,
                           b_val=0.3):
    upsample = None

    if (kernelsize != 1 or stride != 1) or (in_channels != out_channels) or dilation != 1:
        upsample = nn.Sequential(nn.ConvTranspose1d(in_channels, out_channels,
                                                    kernel_size=kernelsize,
                                                    stride=stride,
                                                    padding=padding,
                                                    dilation=dilation,
                                                    output_padding=o_padding),
                                 nn.BatchNorm1d(out_channels))
    layers = [
        ResidualBlock1dTransposeConv(in_channels, out_channels, kernelsize, stride, padding, dilation, o_padding,
                                     upsample=upsample,
                                     a=a_val,
                                     b=b_val,
                                     )
    ]

    return nn.Sequential(*layers)


class DataGeneratorText(nn.Module):
    def __init__(self, args, a=2.0, b=0.3):
        super(DataGeneratorText, self).__init__()
        self.args = args
        modules = [make_res_block_decoder(5 * args.DIM_text, 5 * args.DIM_text,
                                          kernelsize=4, stride=1, padding=0, dilation=1, o_padding=0,
                                          ),
                   make_res_block_decoder(5 * args.DIM_text, 5 * args.DIM_text,
                                          kernelsize=4, stride=2, padding=1, dilation=1,
                                          o_padding=0),
                   make_res_block_decoder(5 * args.DIM_text, 5 * args.DIM_text,
                                          kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0),
                   make_res_block_decoder(5 * args.DIM_text, 4 * args.DIM_text,
                                          kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0),
                   make_res_block_decoder(4 * args.DIM_text, 4 * args.DIM_text,
                                          kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0),
                   ]

        if args.len_sequence >= 512:
            modules.append(make_res_block_decoder(4 * args.DIM_text, 3 * args.DIM_text,
                                                  kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0))
            modules.append(make_res_block_decoder(3 * args.DIM_text, 2 * args.DIM_text,
                                                  kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0))
            modules.append(make_res_block_decoder(2 * args.DIM_text, args.DIM_text,
                                                  kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0))

            modules.append(nn.ConvTranspose1d(args.DIM_text, self.args.vocab_size,
                                              kernel_size=4,
                                              stride=2,
                                              padding=1,
                                              dilation=1,
                                              output_padding=0))
        elif args.len_sequence == 128:
            modules.append(make_res_block_decoder(4 * args.DIM_text, 1 * args.DIM_text,
                                                  kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0))
            modules.append(nn.Conv1d(args.DIM_text, self.args.vocab_size,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     dilation=1))
        else:
            raise NotImplementedError(
                f'The output shapes of this network will not work for len_sequence: {args.len_sequence}.'
                f' Please adapt them.')

        if args.text_gen_lastlayer == "sigmoid":
            modules.append(nn.Sigmoid())
        elif args.text_gen_lastlayer == "softmax":
            modules.append(nn.LogSoftmax(dim=1))
        elif args.text_gen_lastlayer != "none":
            raise NotImplementedError(
                f"{args.text_gen_lastlayer} not implemented, chose between softmax and sigmoid for last data gen layer.")
        self.generator = nn.Sequential(*modules)

    def forward(self, feats):
        """
        Example:
        feats.shape: torch.Size([200, 640, 4])
        torch.Size([200, 640, 8])
        torch.Size([200, 640, 16])
        torch.Size([200, 512, 32])
        torch.Size([200, 512, 64])
        torch.Size([200, 384, 128])
        if sequ_length > 128:
            torch.Size([200, 256, 256])
            torch.Size([200, 128, 512])
        conv2: torch.Size([bs, vocab_size, sequ_length])
        """

        return self.generator(feats)
