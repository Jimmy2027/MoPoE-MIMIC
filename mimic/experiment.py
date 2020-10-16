import os
import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from PIL import ImageFont
from sklearn.metrics import average_precision_score

from mimic.MimicDataset import Mimic, Mimic_testing
from mimic.networks.ConvNetworkImgClf import ClfImg as ClfImg
from mimic.networks.ConvNetworkTextClf import ClfText as ClfText
from mimic.networks.ConvNetworksImgMimic import EncoderImg, DecoderImg
from mimic.networks.ConvNetworksTextMimic import EncoderText, DecoderText
from mimic.networks.VAEtrimodalMimic import VAEtrimodalMimic
from modalities.MimicLateral import MimicLateral
from modalities.MimicPA import MimicPA
from modalities.MimicText import MimicText
from utils.BaseExperiment import BaseExperiment


class MimicExperiment(BaseExperiment):
    def __init__(self, flags, alphabet, **kwargs):
        super().__init__(flags)
        self.labels = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']
        self.flags = flags
        self.experiment_uid = flags.str_experiment
        self.dataset = flags.dataset
        self.plot_img_size = torch.Size((1, 128, 128))
        # some fonts are not found on some devices and need to be passed as argument
        if 'font' in kwargs.keys():
            self.font = kwargs['font']
        else:
            self.font = ImageFont.truetype('FreeSerif.ttf', 38)

        self.alphabet = alphabet
        self.flags.num_features = len(alphabet)

        self.modalities = self.set_modalities()
        self.num_modalities = len(self.modalities.keys())
        self.subsets = self.set_subsets()
        self.dataset_train = None
        self.dataset_test = None
        self.set_dataset()

        self.mm_vae = self.set_model()
        self.clfs = self.set_clfs()
        self.optimizer = None
        self.rec_weights = self.set_rec_weights()
        self.style_weights = self.set_style_weights()

        self.test_samples = self.get_test_samples();
        self.eval_metric = average_precision_score;
        self.paths_fid = self.set_paths_fid()
        self.experiments_dataframe = self.get_experiments_dataframe()

    def set_model(self):
        print('setting model')
        model = VAEtrimodalMimic(self.flags, self.modalities, self.subsets)
        model = model.to(self.flags.device);
        return model;

    def set_modalities(self):
        print('setting modalities')
        mod1 = MimicPA(EncoderImg(self.flags, self.flags.style_pa_dim),
                       DecoderImg(self.flags, self.flags.style_pa_dim))
        mod2 = MimicLateral(EncoderImg(self.flags, self.flags.style_lat_dim),
                            DecoderImg(self.flags, self.flags.style_lat_dim))
        mod3 = MimicText(EncoderText(self.flags, self.flags.style_text_dim),
                         DecoderText(self.flags, self.flags.style_text_dim),
                         self.flags.len_sequence,
                         self.alphabet,
                         self.plot_img_size,
                         self.font)
        mods = {mod1.name: mod1, mod2.name: mod2, mod3.name: mod3}
        return mods;

    def set_dataset(self):
        print('setting dataset')
        if self.dataset == 'testing':
            print('using testing dataset')
            d_train = Mimic_testing()
            d_eval = Mimic_testing()
        else:
            d_train = Mimic(self.flags, self.labels, self.alphabet, dataset=1)
            d_eval = Mimic(self.flags, self.labels, self.alphabet, dataset=2)
        self.dataset_train = d_train
        self.dataset_test = d_eval

    def set_clfs(self):
        print('setting clfs')
        model_clf_m1 = None;
        model_clf_m2 = None;
        model_clf_m3 = None;
        if self.flags.use_clf:
            model_clf_m1 = ClfImg(self.flags, self.labels);
            model_clf_m1.load_state_dict(torch.load(os.path.join(self.flags.dir_clf,
                                                                 self.flags.clf_save_m1)))
            model_clf_m1 = model_clf_m1.to(self.flags.device);

            model_clf_m2 = ClfImg(self.flags, self.labels);
            model_clf_m2.load_state_dict(torch.load(os.path.join(self.flags.dir_clf,
                                                                 self.flags.clf_save_m2)))
            model_clf_m2 = model_clf_m2.to(self.flags.device);

            model_clf_m3 = ClfText(self.flags, self.labels);
            model_clf_m3.load_state_dict(torch.load(os.path.join(self.flags.dir_clf,
                                                                 self.flags.clf_save_m3)))
            model_clf_m3 = model_clf_m3.to(self.flags.device);

        clfs = {'PA': model_clf_m1,
                'Lateral': model_clf_m2,
                'text': model_clf_m3}
        return clfs;

    def set_optimizer(self):
        print('setting optimizer')
        # optimizer definition
        optimizer = optim.Adam(
            list(self.mm_vae.parameters()),
            lr=self.flags.initial_learning_rate,
            betas=(self.flags.beta_1, self.flags.beta_2))
        self.optimizer = optimizer

    def set_rec_weights(self):
        print('setting rec_weights')
        rec_weights = dict()
        ref_mod_d_size = self.modalities['PA'].data_size.numel()
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key]
            numel_mod = mod.data_size.numel()
            rec_weights[mod.name] = float(ref_mod_d_size / numel_mod)
        return rec_weights

    def set_style_weights(self):
        weights = dict()
        weights['PA'] = self.flags.beta_m1_style
        weights['Lateral'] = self.flags.beta_m2_style
        weights['text'] = self.flags.beta_m3_style
        return weights

    def get_prediction_from_attr(self, values):
        return values.ravel()

    def get_test_samples(self, num_images=10):
        n_test = self.dataset_test.__len__()
        samples = []
        for i in range(num_images):
            sample, target = self.dataset_test.__getitem__(random.randint(0, n_test - 1))
            for k, key in enumerate(sample):
                sample[key] = sample[key].to(self.flags.device)
            samples.append(sample)
        return samples

    def mean_eval_metric(self, values):
        return np.mean(np.array(values))

    def eval_label(self, values, labels, index=None):
        pred = values[:, index]
        gt = labels[:, index]
        return self.eval_metric(gt, pred)

    def get_experiments_dataframe(self) -> pd.DataFrame:
        """
        Gets the experiment results dataframe which contains test results of previous experiments together with their
        parameters
        """
        if os.path.exists('experiments_dataframe.csv'):
            experiments_dataframe = pd.read_csv('experiments_dataframe.csv')
            flags_dict = vars(self.flags)
            flags_dict['experiment_uid'] = self.experiment_uid
            experiments_dataframe = experiments_dataframe.append(flags_dict, ignore_index=True)
        else:
            experiments_dataframe = pd.DataFrame()
            flags_dict = vars(self.flags)
            flags_dict['experiment_uid'] = self.experiment_uid
            experiments_dataframe = experiments_dataframe.append(flags_dict, ignore_index=True)

        return experiments_dataframe

    def update_experiments_dataframe(self, values_dict: dict):
        """
        Updates the values in experiments dataframe with the new values from the values_dict and saves it if the
        experiment is not a test run
        """
        for key in values_dict:
            self.experiments_dataframe.loc[
                self.experiments_dataframe['experiment_uid'] == self.experiment_uid, key] = values_dict[key]
        if not self.flags.dataset == 'test':
            self.experiments_dataframe.to_csv('experiments_dataframe.csv', index=False)
