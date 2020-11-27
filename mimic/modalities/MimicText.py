import torch

import mimic.modalities.utils
from mimic.modalities.Modality import Modality
from mimic.utils import plot
from mimic.utils.save_samples import write_samples_text_to_file
from mimic.utils.text import tensor_to_text


class MimicText(Modality):
    def __init__(self, enc, dec, len_sequence, plotImgSize, font, args):
        self.name = 'text'
        self.args = args
        self.likelihood_name = 'categorical'
        self.len_sequence = len_sequence
        if args.text_encoding == 'char':
            self.alphabet = args.alphabet
            self.data_size = torch.Size((len(args.alphabet), len_sequence))
        elif args.text_encoding == 'word':
            self.data_size = torch.Size((args.vocab_size, len_sequence))
        self.plot_img_size = plotImgSize
        self.font = font
        self.gen_quality_eval = False
        self.file_suffix = '.txt'
        self.encoder = enc
        self.decoder = dec
        self.likelihood: torch.distributions = mimic.modalities.utils.get_likelihood(self.likelihood_name)

    def save_data(self, exp, d, fn, args):
        write_samples_text_to_file(tensor_to_text(exp, d.unsqueeze(0)), fn)

    def plot_data(self, exp, d):
        if exp.flags.text_encoding == 'word':
            d = torch.nn.functional.one_hot(d, um_classes=self.args.vocab_size)
        return plot.text_to_pil(exp, d.unsqueeze(0), self.plot_img_size, self.font)

    def calc_log_prob(self, out_dist: torch.distributions, target: torch.Tensor, norm_value: int):
        if self.args.text_encoding == 'word':
            target = torch.nn.functional.one_hot(target.to(torch.int64), num_classes=self.args.vocab_size)
        return Modality.calc_log_prob(self, out_dist, target, norm_value)
