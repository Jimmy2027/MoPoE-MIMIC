import torch

from mimic.modalities.Modality import Modality
from mimic.utils import utils
from mimic.utils.save_samples import write_samples_img_to_file


class MimicPA(Modality):
    def __init__(self, enc, dec):
        self.name = 'PA';
        self.likelihood_name = 'laplace';
        self.data_size = torch.Size((1, 128, 128));
        self.gen_quality_eval = True;
        self.file_suffix = '.png';
        self.encoder = enc;
        self.decoder = dec;
        self.likelihood = utils.get_likelihood(self.likelihood_name);

    def save_data(self, d, fn, args):
        img_per_row = args['img_per_row'];
        write_samples_img_to_file(d, fn, img_per_row);

    def plot_data(self, d):
        p = d.repeat(1, 1, 1, 1);
        return p;
