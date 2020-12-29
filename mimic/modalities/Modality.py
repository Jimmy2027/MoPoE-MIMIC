from abc import ABC, abstractmethod
import torch
import typing

import PIL.Image as Image
import torch
from torch import Tensor
from torchvision import transforms

import mimic.modalities.utils
from mimic.utils.save_samples import write_samples_img_to_file
from typing import Optional


class Modality(ABC):

    @abstractmethod
    def save_data(self, exp, d, fn, args):
        pass;

    @abstractmethod
    def plot_data(self, exp, d):
        pass;

    def calc_log_prob(self, out_dist, target: torch.Tensor, norm_value: int):
        log_prob = out_dist.log_prob(target).sum()
        return log_prob / norm_value


class ModalityIMG(Modality):
    def __init__(self, data_size):
        self.data_size = data_size

    def save_data(self, exp, d, fn, args):
        img_per_row = args['img_per_row']
        write_samples_img_to_file(d, fn, img_per_row)

    def plot_data(self, exp, d: Tensor, log_tag: Optional[str] = None):
        if d.shape != self.data_size:
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Grayscale(),
                                            transforms.Resize(size=self.data_size[1:], interpolation=Image.BICUBIC),
                                            transforms.ToTensor()])
            d = transform(d.cpu())
        return d.repeat(1, 1, 1, 1)
