import os
import random
import typing
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from PIL import ImageFont
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from mimic import log
from mimic.dataio.MimicDataset import Mimic, Mimic_testing
from mimic.dataio.MimicDataset import MimicText as MimicTextDataset
from mimic.dataio.utils import get_transform_img, get_str_labels
from mimic.modalities.MimicLateral import MimicLateral
from mimic.modalities.MimicPA import MimicPA
from mimic.modalities.MimicText import MimicText
from mimic.modalities.Modality import Modality
from mimic.networks.CheXNet import CheXNet
from mimic.networks.ConvNetworkImgClf import ClfImg as ClfImg
from mimic.networks.ConvNetworkTextClf import ClfText as ClfText
from mimic.networks.ConvNetworksImgMimic import EncoderImg, DecoderImg
from mimic.networks.ConvNetworksTextMimic import EncoderText, DecoderText
from mimic.networks.VAEtrimodalMimic import VAEtrimodalMimic, VAETextMimic
from mimic.utils import utils
from mimic.utils.BaseExperiment import BaseExperiment
from mimic.utils.TBLogger import TBLogger
from mimic.utils.text import tensor_to_text
from mimic.utils.utils import get_clf_path, get_alphabet
from mimic.utils.utils import init_twolevel_nested_dict


class MimicExperiment(BaseExperiment):
    def __init__(self, flags):
        super().__init__(flags)
        self.labels = get_str_labels(flags.binary_labels)
        self.flags = flags
        self.experiment_uid = flags.str_experiment
        self.dataset = flags.dataset
        self.plot_img_size = torch.Size((1, 128, 128))

        if self.flags.text_encoding == 'char':
            self.alphabet = get_alphabet()
            self.flags.num_features = len(self.alphabet)

        self.dataset_train, self.dataset_test, self.font = self.set_dataset()
        self.modalities: typing.Mapping[str, Modality] = self.set_modalities()
        self.num_modalities = len(self.modalities.keys())
        self.subsets = self.set_subsets()

        self.mm_vae = self.set_model()
        self.clfs = self.set_clfs()
        self.clf_transforms: dict = self.set_clf_transforms()
        self.optimizer = None
        self.rec_weights = self.set_rec_weights()
        self.style_weights = self.set_style_weights()

        self.test_samples = self.get_test_samples()
        self.eval_metric = average_precision_score
        self.paths_fid = self.set_paths_fid()
        self.experiments_dataframe = self.get_experiments_dataframe()

        self.restart_experiment = False  # if true and the model returns nans, the workflow gets started again
        self.number_restarts = 0
        self.tb_logger = None

    def set_model(self):
        if self.flags.only_text_modality:
            return VAETextMimic(self.flags, self.modalities, self.subsets)
        else:
            return VAEtrimodalMimic(self.flags, self.modalities, self.subsets)

    def set_modalities(self) -> typing.Mapping[str, Modality]:
        log.info('setting modalities')
        mod1 = MimicPA(EncoderImg(self.flags, self.flags.style_pa_dim),
                       DecoderImg(self.flags, self.flags.style_pa_dim), self.flags)
        mod2 = MimicLateral(EncoderImg(self.flags, self.flags.style_lat_dim),
                            DecoderImg(self.flags, self.flags.style_lat_dim), self.flags)
        mod3 = MimicText(EncoderText(self.flags, self.flags.style_text_dim),
                         DecoderText(self.flags, self.flags.style_text_dim), self.flags.len_sequence,
                         self.plot_img_size, self.font, self.flags)
        if self.flags.only_text_modality:
            return {mod3.name: mod3}
        else:
            return {mod1.name: mod1, mod2.name: mod2, mod3.name: mod3}

    def set_dataset(self):
        font = ImageFont.truetype(str(Path(__file__).parent.parent / 'data/FreeSerif.ttf'),
                                  20) if not self.flags.distributed else None
        log.info('setting dataset')
        # used for faster unittests i.e. a dummy dataset
        if self.dataset == 'testing':
            log.info('using testing dataset')
            self.flags.vocab_size = 3517
            d_train = Mimic_testing(self.flags)
            d_eval = Mimic_testing(self.flags)
        else:
            if self.flags.only_text_modality:
                d_train = MimicTextDataset(args=self.flags, str_labels=self.labels, split='train')
                d_eval = MimicTextDataset(self.flags, self.labels, split='eval')
            else:
                d_train = Mimic(self.flags, self.labels, split='train')
                d_eval = Mimic(self.flags, self.labels, split='eval')
        return d_train, d_eval, font

    def set_clf_transforms(self) -> dict:
        if self.flags.text_clf_type == 'word':
            def text_transform(x):
                # converts one hot encoding to indices vector
                return torch.argmax(x, dim=-1)
        else:
            def text_transform(x):
                return x

        # create temporary args to set the number of crops to 1
        temp_args = Namespace(**vars(self.flags))
        temp_args.n_crops = 1
        return {
            'PA': get_transform_img(temp_args, self.flags.img_clf_type),
            'Lateral': get_transform_img(temp_args, self.flags.img_clf_type),
            'text': text_transform
        }

    def set_clfs(self) -> typing.Mapping[str, torch.nn.Module]:
        log.info('setting clfs')

        # mapping clf type to clf_save_m*
        clf_save_names: typing.Mapping[str, str] = {
            'PA': self.flags.clf_save_m1,
            'Lateral': self.flags.clf_save_m2,
            'text': self.flags.clf_save_m3
        }
        clfs = {f'{mod}': None for mod in self.modalities}
        if self.flags.use_clf:
            for mod in self.modalities:
                if mod in ['PA', 'Lateral']:
                    # finding the directory of the classifier
                    dir_img_clf = os.path.join(self.flags.dir_clf,
                                               f'Mimic{self.flags.img_size}_{self.flags.img_clf_type}'
                                               f'{"_bin_label" if self.flags.binary_labels else ""}')
                    dir_img_clf = os.path.expanduser(dir_img_clf)
                    # finding and loading state dict
                    clf = ClfImg(self.flags, self.labels) if self.flags.img_clf_type == 'resnet' else CheXNet(
                        len(self.labels))
                    clf_path = get_clf_path(dir_img_clf, clf_save_names[mod])
                    clf.load_state_dict(torch.load(clf_path, map_location=self.flags.device))
                    clfs[mod] = clf.to(self.flags.device)
                elif mod == 'text':
                    # create temporary args to set the word encoding of the classifier to text_clf_type.
                    # This allows to have a different text encoding setting for the VAE than for the classifier.
                    temp_args = Namespace(**vars(self.flags))
                    temp_args.text_encoding = self.flags.text_clf_type
                    clf = ClfText(temp_args, self.labels)
                    clf_path = get_clf_path(self.flags.dir_clf, clf_save_names[
                        mod] + f'vocabsize_{self.flags.vocab_size}{"_bin_label" if self.flags.binary_labels else ""}')

                    clf.load_state_dict(torch.load(clf_path, map_location=self.flags.device))
                    clfs[mod] = clf.to(self.flags.device)
                else:
                    raise NotImplementedError

        return clfs

    def set_optimizer(self):
        log.info('setting optimizer')
        # optimizer definition
        optimizer = optim.Adam(
            list(self.mm_vae.parameters()),
            lr=self.flags.initial_learning_rate,
            betas=(self.flags.beta_1, self.flags.beta_2))
        self.optimizer = optimizer

    def set_rec_weights(self):
        """
        Sets the weights of the log probs for each modality.
        """
        log.info('setting rec_weights')

        return {
            'PA': self.flags.rec_weight_m1,
            'Lateral': self.flags.rec_weight_m2,
            'text': self.flags.rec_weight_m3
        }

    def set_style_weights(self):
        return {
            'PA': self.flags.beta_m1_style,
            'Lateral': self.flags.beta_m2_style,
            'text': self.flags.beta_m3_style,
        }

    def get_prediction_from_attr(self, values):
        return values.ravel()

    def get_test_samples(self, num_images=10) -> typing.Iterable[typing.Mapping[str, Tensor]]:
        """
        Gets random samples from the test dataset
        """
        n_test = self.dataset_test.__len__()
        samples = []
        for _ in range(num_images):
            sample, _ = self.dataset_test.__getitem__(random.randint(0, n_test - 1))
            sample = utils.dict_to_device(sample, self.flags.device)

            samples.append(sample)

        return samples

    def mean_eval_metric(self, values):
        return np.mean(np.array(values))

    def eval_label(self, values: Tensor, labels: Tensor, index: int = None):
        """
        index: index of the labels
        """
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
        experiments_dataframe.to_csv('experiments_dataframe.csv', index=False)
        return experiments_dataframe

    def update_experiments_dataframe(self, values_dict: dict):
        """
        Updates the values in experiments dataframe with the new values from the values_dict and saves it if the
        experiment is not a test run
        """
        log.info(f"writing to experiment df with uid {self.experiment_uid}: {values_dict}")
        # load dataframe every time in order not to overwrite other writers
        if os.path.exists('experiments_dataframe.csv'):
            self.experiments_dataframe = pd.read_csv('experiments_dataframe.csv')
        for key, value in values_dict.items():
            self.experiments_dataframe.loc[
                self.experiments_dataframe['experiment_uid'] == self.experiment_uid, key] = value

        if self.flags.dataset != 'testing':
            self.experiments_dataframe.to_csv('experiments_dataframe.csv', index=False)


    def init_summary_writer(self):
        log.info(f'setting up summary writer for device {self.flags.device}')
        # initialize summary writer
        writer = SummaryWriter(self.flags.dir_logs)
        tb_logger = TBLogger(self.flags.str_experiment, writer)
        str_flags = utils.save_and_log_flags(self.flags)
        tb_logger.writer.add_text('FLAGS', str_flags, 0)
        # todo find a way to store model graph
        # tb_logger.write_model_graph(exp.mm_vae)
        self.log_text_test_samples(tb_logger)
        return tb_logger

    def log_text_test_samples(self, tb_logger):
        """
        Logs the text test samples to the tb_logger to verify if the text encoding does what it is supposed to do.
        """
        samples = self.test_samples
        one_hot = self.flags.text_encoding != 'word'
        text_test_samples = tensor_to_text(self,
                                           torch.cat(([samples[i]['text'].unsqueeze(0) for i in range(5)]), 0),
                                           one_hot=one_hot)
        tb_logger.write_texts_from_list('test_samples', text_test_samples, text_encoding=self.flags.text_encoding)


class Callbacks:
    def __init__(self, exp: MimicExperiment):
        self.args = exp.flags
        self.exp = exp
        self.logger: TBLogger = exp.tb_logger
        optimizer = exp.optimizer
        self.experiment_df = exp.experiments_dataframe
        self.start_early_stopping_epoch = self.args.start_early_stopping_epoch
        self.max_early_stopping_index = self.args.max_early_stopping_index
        # initialize with infinite loss
        self.losses = [float('inf')]
        self.patience_idx = 1
        self.scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
        self.elapsed_times = []
        self.results_lr = init_twolevel_nested_dict(exp.labels, exp.subsets.keys(), init_val=[], copy_init_val=True)

    def update_epoch(self, epoch, loss, elapsed_time, results_lr):
        self._update_results_lr(results_lr)
        stop_early = False
        self.elapsed_times.append(elapsed_time)
        self.scheduler.step(loss)
        self.logger.writer.add_scalars(f'test/mean_loss', {'mean_loss': loss}, epoch)

        log.info(f'current test loss: {loss}')
        self.save_checkpoint(epoch)

        # evaluate progress
        if epoch > self.start_early_stopping_epoch and loss < min(self.losses):
            log.info(f'current test loss {loss} improved from {min(self.losses)}'
                     f' at epoch {np.argmin(self.losses)}')
            self.exp.update_experiments_dataframe(
                {'total_test_loss': loss, 'total_epochs': epoch, 'mean_epoch_time': np.mean(self.elapsed_times)})
            self.patience_idx = 1

        elif self.patience_idx > self.max_early_stopping_index:
            log.info(
                f'stopping early at epoch {epoch} because current test loss {loss} '
                f'did not improve from {min(self.losses)} '
                f'at epoch {np.argmin(self.losses)}')
            stop_early = True

        else:
            if epoch > self.start_early_stopping_epoch:
                log.info(f'current test loss {loss} did not improve from {min(self.losses)} '
                         f'at epoch {np.argmin(self.losses)}')
                log.info(f'-- idx_early_stopping = {self.patience_idx} / {self.max_early_stopping_index}')
                self.patience_idx += 1

        self.losses.append(loss)

        if (epoch + 1) % self.args.eval_freq == 0 or (epoch + 1) == self.args.end_epoch:
            # plot evolution of metrics every Nth epochs
            self.plot_results_lr()

        return stop_early

    def plot_results_lr(self):
        if not self.exp.flags.dir_experiment_run.is_dir():
            os.mkdir(self.exp.flags.dir_experiment_run)
        for label, d_label in self.results_lr.items():
            for subset, values in d_label.items():
                plt.plot(values, label=subset)
            plt.title(f'{label}, eval freq: {self.args.eval_freq} epochs')
            plt.legend()
            out_path = self.exp.flags.dir_experiment_run / f"{label.replace(' ', '_')}.png"
            if out_path.is_file():
                out_path.unlink()
            plt.savefig(out_path)
            log.info(f"Saving plot to {out_path}")
            plt.close()

    def _update_results_lr(self, results_lr):
        # update values only if results_lr is non None, (the test metrics are only evaluated every Nth epoch)
        if results_lr:
            for label, d_label in results_lr.items():
                for subset in d_label:
                    self.results_lr[label][subset].append(results_lr[label][subset])

    def save_checkpoint(self, epoch):
        # save checkpoints every 5 epochs
        # when using DDP, the model is the same over all devices, only need to save it for one process
        if ((epoch + 1) % 50 == 0 or (
                epoch + 1) == self.exp.flags.end_epoch) and (
                not self.args.distributed or self.exp.flags.device % self.exp.flags.world_size == 0):
            dir_network_epoch = os.path.join(self.exp.flags.dir_checkpoints, str(epoch).zfill(4))
            if not os.path.exists(dir_network_epoch):
                os.makedirs(dir_network_epoch)
            if self.args.distributed:
                self.exp.mm_vae.module.save_networks()
            else:
                self.exp.mm_vae.save_networks()
            torch.save(self.exp.mm_vae.state_dict(),
                       os.path.join(dir_network_epoch, self.exp.flags.mm_vae_save))
