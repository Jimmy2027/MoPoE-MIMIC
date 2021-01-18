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
    layers = []
    layers.append(
        ResidualBlock1dTransposeConv(in_channels, out_channels, kernelsize, stride, padding, dilation, o_padding,
                                     upsample=upsample, a=a_val, b=b_val))
    return nn.Sequential(*layers)


class DataGeneratorText(nn.Module):
    def __init__(self, args, a=2.0, b=0.3):
        super(DataGeneratorText, self).__init__()
        self.args = args
        self.resblock_1 = make_res_block_decoder(5 * args.DIM_text, 5 * args.DIM_text,
                                                 kernelsize=4, stride=1, padding=0, dilation=1, o_padding=0)
        self.resblock_2 = make_res_block_decoder(5 * args.DIM_text, 5 * args.DIM_text,
                                                 kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.resblock_3 = make_res_block_decoder(5 * args.DIM_text, 5 * args.DIM_text,
                                                 kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.resblock_4 = make_res_block_decoder(5 * args.DIM_text, 4 * args.DIM_text,
                                                 kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.resblock_5 = make_res_block_decoder(4 * args.DIM_text, 4 * args.DIM_text,
                                                 kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.resblock_6 = make_res_block_decoder(4 * args.DIM_text, 3 * args.DIM_text,
                                                 kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.resblock_7 = make_res_block_decoder(3 * args.DIM_text, 2 * args.DIM_text,
                                                 kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.resblock_8 = make_res_block_decoder(2 * args.DIM_text, args.DIM_text,
                                                 kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        # if the output dimension of the resblock_8 is bigger than the sequence length, a simple convolution is needed
        # to reduce the output dim to the sequence length. Otherwise a transposed conv is used.
        if self.args.len_sequence > 500:
            self.conv2 = nn.ConvTranspose1d(args.DIM_text, self.args.vocab_size,
                                            kernel_size=4,
                                            stride=2,
                                            padding=1,
                                            dilation=1,
                                            output_padding=0)
        else:
            self.conv2 = nn.Conv1d(self.args.DIM_text, self.args.vocab_size, stride=4, kernel_size=4)

        # inverts the 'embedding' module upto one-hotness
        # needs input shape of (batch_size, self.args.DIM_text)
        self.toVocabSize = nn.Linear(self.args.DIM_text, self.args.vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats):
        """
        Example:
        feats.shape: torch.Size([200, 640, 4])
        torch.Size([200, 640, 8])
        torch.Size([200, 640, 16])
        torch.Size([200, 512, 32])
        torch.Size([200, 512, 64])
        torch.Size([200, 384, 128])
        torch.Size([200, 256, 256])
        torch.Size([200, 128, 512])
        conv2: torch.Size([bs, vocab_size, sequ_length])
        """
        d = self.resblock_1(feats)
        d = self.resblock_2(d)
        d = self.resblock_3(d)
        d = self.resblock_4(d)
        d = self.resblock_5(d)
        d = self.resblock_6(d)
        d = self.resblock_7(d)
        d = self.resblock_8(d)
        d = self.conv2(d)
        # d = d.view(-1, self.args.DIM_text)
        # d = self.toVocabSize(d)
        # d = self.softmax(d)
        d = self.sigmoid(d)
        return d
