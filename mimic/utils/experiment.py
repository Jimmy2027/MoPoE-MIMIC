import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from PIL import ImageFont
from sklearn.metrics import average_precision_score
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mimic.dataio.MimicDataset import Mimic, Mimic_testing
from mimic.modalities.MimicLateral import MimicLateral
from mimic.modalities.MimicPA import MimicPA
from mimic.modalities.MimicText import MimicText
from mimic.networks.ConvNetworkImgClf import ClfImg as ClfImg
from mimic.networks.ConvNetworkTextClf import ClfText as ClfText
from mimic.networks.ConvNetworksImgMimic import EncoderImg, DecoderImg
from mimic.networks.ConvNetworksTextMimic import EncoderText, DecoderText
from mimic.networks.VAEtrimodalMimic import VAEtrimodalMimic
from mimic.utils import utils
from mimic.utils.BaseExperiment import BaseExperiment
from mimic.utils.TBLogger import TBLogger
from mimic.utils.utils import get_clf_path, get_alphabet


class MimicExperiment(BaseExperiment):
    def __init__(self, flags):
        super().__init__(flags)
        self.labels = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']
        self.flags = flags
        self.experiment_uid = flags.str_experiment
        self.dataset = flags.dataset
        self.plot_img_size = torch.Size((1, 128, 128))

        self.font = ImageFont.truetype(str(Path(__file__).parent.parent / 'FreeSerif.ttf'),
                                       38) if not flags.distributed else None
        if self.flags.text_encoding == 'char':
            self.alphabet = get_alphabet()
            self.flags.num_features = len(self.alphabet)

        self.dataset_train = None
        self.dataset_test = None
        self.set_dataset()
        self.modalities = self.set_modalities()
        self.num_modalities = len(self.modalities.keys())
        self.subsets = self.set_subsets()

        self.mm_vae = self.set_model()
        self.clfs = self.set_clfs()
        self.optimizer = None
        self.rec_weights = self.set_rec_weights()
        self.style_weights = self.set_style_weights()

        self.test_samples = self.get_test_samples()
        self.eval_metric = average_precision_score
        self.paths_fid = self.set_paths_fid()
        self.experiments_dataframe = self.get_experiments_dataframe()

        self.restart_experiment = False  # if the model returns nans, the workflow gets started again
        self.number_restarts = 0
        self.tb_logger = None

    def set_model(self):
        return VAEtrimodalMimic(self.flags, self.modalities, self.subsets)

    def set_modalities(self):
        print('setting modalities')
        mod1 = MimicPA(EncoderImg(self.flags, self.flags.style_pa_dim),
                       DecoderImg(self.flags, self.flags.style_pa_dim))
        mod2 = MimicLateral(EncoderImg(self.flags, self.flags.style_lat_dim),
                            DecoderImg(self.flags, self.flags.style_lat_dim))
        mod3 = MimicText(EncoderText(self.flags, self.flags.style_text_dim),
                         DecoderText(self.flags, self.flags.style_text_dim), self.flags.len_sequence,
                         self.plot_img_size, self.font, self.flags)
        return {mod1.name: mod1, mod2.name: mod2, mod3.name: mod3}

    def set_dataset(self):
        print('setting dataset')
        if self.dataset == 'testing':
            print('using testing dataset')
            self.flags.vocab_size = 3517
            d_train = Mimic_testing(self.flags)
            d_eval = Mimic_testing(self.flags)
        else:
            d_train = Mimic(self.flags, self.labels, split='train')
            d_eval = Mimic(self.flags, self.labels, split='eval')
        self.dataset_train = d_train
        self.dataset_test = d_eval

    def set_clfs(self):
        print('setting clfs')
        # img_clf_type and feature_extractor_img need to be the same. (for the image transformations of the dataset)
        self.flags.img_clf_type = self.flags.feature_extractor_img

        model_clf_m1 = None
        model_clf_m2 = None
        model_clf_m3 = None
        if self.flags.use_clf:
            # finding the directory of the classifier
            dir_img_clf = os.path.join(self.flags.dir_clf, f'Mimic{self.flags.img_size}_{self.flags.img_clf_type}')
            dir_img_clf = os.path.expanduser(dir_img_clf)
            # finding and loading state dict
            model_clf_m1 = ClfImg(self.flags, self.labels)
            clf_m1_path = get_clf_path(dir_img_clf, self.flags.clf_save_m1)
            model_clf_m1.load_state_dict(torch.load(clf_m1_path))
            model_clf_m1 = model_clf_m1.to(self.flags.device)

            model_clf_m2 = ClfImg(self.flags, self.labels)
            clf_m2_path = get_clf_path(dir_img_clf, self.flags.clf_save_m2)
            model_clf_m2.load_state_dict(torch.load(clf_m2_path))
            model_clf_m2 = model_clf_m2.to(self.flags.device)

            model_clf_m3 = ClfText(self.flags, self.labels)
            clf_m3_path = get_clf_path(self.flags.dir_clf, f'clf_text_{self.flags.text_encoding}_encoding')

            model_clf_m3.load_state_dict(torch.load(clf_m3_path))
            model_clf_m3 = model_clf_m3.to(self.flags.device)

        return {'PA': model_clf_m1,
                'Lateral': model_clf_m2,
                'text': model_clf_m3}

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
        rec_weights = {}
        ref_mod_d_size = self.modalities['PA'].data_size.numel()
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key]
            numel_mod = mod.data_size.numel()
            rec_weights[mod.name] = float(ref_mod_d_size / numel_mod)
        return rec_weights

    def set_style_weights(self):
        return {
            'PA': self.flags.beta_m1_style,
            'Lateral': self.flags.beta_m2_style,
            'text': self.flags.beta_m3_style,
        }

    def get_prediction_from_attr(self, values):
        return values.ravel()

    def get_test_samples(self, num_images=10):
        n_test = self.dataset_test.__len__()
        samples = []
        for _ in range(num_images):
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
            flags_dict['total_epochs'] = 0
            flags_dict['experiment_duration'] = -1
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
        if self.flags.dataset != 'testing':
            self.experiments_dataframe.to_csv('experiments_dataframe.csv', index=False)

    def init_summary_writer(self):
        print(f'setting up summary writer for device {self.flags.device}')
        # initialize summary writer
        writer = SummaryWriter(self.flags.dir_logs)
        tb_logger = TBLogger(self.flags.str_experiment, writer)
        str_flags = utils.save_and_log_flags(self.flags)
        tb_logger.writer.add_text('FLAGS', str_flags, 0)
        # todo find a way to store model graph
        # tb_logger.write_model_graph(exp.mm_vae)
        return tb_logger


class Callbacks:
    def __init__(self, exp: MimicExperiment):
        self.args = exp.flags
        self.exp = exp
        self.logger: TBLogger = exp.tb_logger
        optimizer = exp.optimizer
        self.experiment_df = exp.experiments_dataframe
        self.start_early_stopping_epoch = self.args.start_early_stopping_epoch
        self.max_early_stopping_index = self.args.max_early_stopping_index
        # initialize with huge loss
        self.losses = [1e10]
        self.patience_idx = 1
        self.scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
        self.elapsed_times = []

    def update_epoch(self, epoch, loss, elapsed_time):
        stop_early = False
        self.elapsed_times.append(elapsed_time)
        self.scheduler.step(loss)
        if self.logger:
            self.logger.writer.add_scalars(f'test/mean_loss', {'mean_loss': loss}, epoch)

        print(f'current test loss: {loss}')
        self.save_checkpoint(epoch)

        if epoch > self.start_early_stopping_epoch and loss < min(self.losses):
            print(f'current test loss {loss} improved from {min(self.losses)}'
                  f' at epoch {np.argmin(self.losses)}')
            if self.logger:
                self.exp.update_experiments_dataframe({'total_test_loss': loss, 'total_epochs': epoch})
            self.patience_idx = 1

        elif self.patience_idx > self.max_early_stopping_index:
            print(
                f'stopping early at epoch {epoch} because current test loss {loss} '
                f'did not improve from {min(self.losses)} '
                f'at epoch {np.argmin(self.losses)}')
            stop_early = True

        else:
            if epoch > self.start_early_stopping_epoch:
                print(f'current test loss {loss} did not improve from {min(self.losses)} '
                      f'at epoch {np.argmin(self.losses)}')
                print(f'-- idx_early_stopping = {self.patience_idx} / {self.max_early_stopping_index}')
                self.patience_idx += 1

        self.losses.append(loss)
        return stop_early

    def save_checkpoint(self, epoch):
        # save checkpoints every 5 epochs
        if ((epoch + 1) % 5 == 0 or (epoch + 1) == self.exp.flags.end_epoch) and self.exp.tb_logger:
            dir_network_epoch = os.path.join(self.exp.flags.dir_checkpoints, str(epoch).zfill(4))
            if not os.path.exists(dir_network_epoch):
                os.makedirs(dir_network_epoch)
            if self.args.distributed:
                self.exp.mm_vae.module.save_networks()
            else:
                self.exp.mm_vae.save_networks()
            torch.save(self.exp.mm_vae.state_dict(),
                       os.path.join(dir_network_epoch, self.exp.flags.mm_vae_save))

    def write_mean_epoch_time_to_expdf(self):
        self.exp.update_experiments_dataframe({'mean_epoch_time': np.mean(self.elapsed_times)})
