
from abc import ABC, abstractmethod


class Modality(ABC):

    @abstractmethod
    def save_data(self, d, fn, args):
        pass;

    @abstractmethod
    def plot_data(self, d):
        pass;


    def calc_log_prob(self, out_dist, target, norm_value):
        log_prob = out_dist.log_prob(target).sum();
        mean_val_logprob = log_prob/norm_value;
        return mean_val_logprob;
