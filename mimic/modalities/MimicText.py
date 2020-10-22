import torch

from mimic.modalities.Modality import Modality
from mimic.utils import plot
from mimic.utils import utils
from mimic.utils.save_samples import write_samples_text_to_file
from mimic.utils.text import tensor_to_text


class MimicText(Modality):
    def __init__(self, enc, dec, len_sequence, alphabet, plotImgSize, font, args):
        self.name = 'text'
        self.likelihood_name = 'categorical'
        self.alphabet = alphabet
        self.len_sequence = len_sequence
        if args.text_encoding == 'char':
            self.data_size = torch.Size((len(alphabet), len_sequence))
        elif args.text_encoding == 'word':
            self.data_size = torch.Size((args.vocab_size, len_sequence))
        self.plot_img_size = plotImgSize
        self.font = font
        self.gen_quality_eval = False
        self.file_suffix = '.txt'
        self.encoder = enc
        self.decoder = dec
        self.likelihood = utils.get_likelihood(self.likelihood_name)

    def save_data(self, exp, d, fn, args):
        write_samples_text_to_file(tensor_to_text(exp,
                                                  d.unsqueeze(0)),
                                   fn)

    def plot_data(self, exp, d):
        out = plot.text_to_pil(exp, d.unsqueeze(0), self.plot_img_size, self.font)
        return out
