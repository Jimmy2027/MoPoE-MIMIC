from abc import ABC, abstractmethod
import torch


class Modality(ABC):

    @abstractmethod
    def save_data(self, exp, d, fn, args):
        pass;

    @abstractmethod
    def plot_data(self, exp, d):
        pass;

    def calc_log_prob(self, out_dist, target: torch.Tensor, norm_value: int):
        log_prob = out_dist.log_prob(target).sum()
        mean_val_logprob = log_prob / norm_value
        return mean_val_logprob
