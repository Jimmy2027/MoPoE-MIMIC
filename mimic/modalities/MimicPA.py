import torch

import mimic.modalities.utils
from mimic.modalities.Modality import ModalityIMG


class MimicPA(ModalityIMG):
    def __init__(self, enc, dec):
        self.name = 'PA'
        self.likelihood_name = 'laplace'
        self.data_size = torch.Size((1, 128, 128))
        super().__init__(data_size=self.data_size)
        self.gen_quality_eval = True
        self.file_suffix = '.png'
        self.encoder = enc
        self.decoder = dec
        self.likelihood = mimic.modalities.utils.get_likelihood(self.likelihood_name)
