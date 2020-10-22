"""
Data generator from mmvae:
"""

import torch.nn as nn


class Dec(nn.Module):
    """ Generate a sentence given a sample from the latent space. """

    def __init__(self, args):
        super(Dec, self).__init__()
        self.args = args
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(5 * args.DIM_text, args.DIM_text * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.DIM_text * 4),
            nn.ReLU(True),
            # size: (args.DIM_text * 8) x 4 x 4
            nn.ConvTranspose2d(args.DIM_text * 4, args.DIM_text * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(args.DIM_text * 4),
            nn.ReLU(True),
            # size: (args.DIM_text * 8) x 4 x 8
            nn.ConvTranspose2d(args.DIM_text * 4, args.DIM_text * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(args.DIM_text * 4),
            nn.ReLU(True),
            # size: (args.DIM_text * 4) x 8 x 32
            nn.ConvTranspose2d(args.DIM_text * 4, args.DIM_text * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.DIM_text * 2),
            nn.ReLU(True),
            # size: (args.DIM_text * 2) x 16 x 64
            nn.ConvTranspose2d(args.DIM_text * 2, args.DIM_text, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.DIM_text),
            nn.ReLU(True),
            # size: (args.DIM_text) x 32 x 128
            nn.ConvTranspose2d(args.DIM_text, 1, 4, 2, 1, bias=False),
            nn.ReLU(True)
            # Output size: 1 x 64 x 256
        )
        # inverts the 'embedding' module upto one-hotness
        self.toVocabSize = nn.Linear(args.DIM_text, self.args.vocab_size)

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:])).view(-1, self.args.DIM_text)
        # out.shape = (320,128)
        # toVocabSize(out).shape = [320, 1590]
        return self.toVocabSize(out).view(*z.size()[:-3], self.args.sentence_length, self.args.vocab_size),