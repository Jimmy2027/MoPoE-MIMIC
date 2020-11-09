import os
from timeit import default_timer as timer

import pandas as pd
import torch

from mimic.networks.CheXNet import CheXNet
from mimic.networks.ConvNetworkImgClf import ClfImg
from mimic.networks.ConvNetworkTextClf import ClfText
from mimic.utils.filehandling import create_dir
from mimic.utils.filehandling import expand_paths
from mimic.utils.filehandling import get_str_experiments
import numpy as np

LABELS = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']


class Callbacks:
    def __init__(self, flags, start_early_stopping_epoch: int, max_early_stopping_index, modality: str,
                 experiment_df: any, logger: any):
        self.modality = modality
        self.logger = logger
        self.flags = flags
        self.experiment_df = experiment_df
        self.start_early_stopping_epoch = start_early_stopping_epoch
        self.max_early_stopping_index = max_early_stopping_index
        self.val_metric_values = [0]
        self.patience_idx = 0

    def update_epoch(self, epoch, loss, mean_AP, model):
        stop_early = False

        self.logger.add_scalars(f'eval_clf_{self.modality}/mean_AP', {self.modality: mean_AP}, epoch)
        self.logger.add_scalars(f'eval_clf_{self.modality}/mean_loss', {self.modality: loss}, epoch)
        self.experiment_df.update_experiments_dataframe(
            {'mean_AP': mean_AP, 'mean_eval_loss': loss, 'total_epochs': epoch})

        print(f'current eval loss: {loss}, mean_AP: {mean_AP}')
        # start saving checkpoints after epoch 10, if the loss improved
        if epoch > self.start_early_stopping_epoch and mean_AP > self.val_metric_values[-1]:
            if self.modality == 'PA':
                filename = self.flags.clf_save_m1
            elif self.modality == 'Lateral':
                filename = self.flags.clf_save_m2
            else:
                filename = self.flags.clf_save_m3
            self.save_and_overwrite_model(model.state_dict(), self.flags.dir_clf, filename, epoch)
            self.patience_idx = 0
        elif self.patience_idx > self.max_early_stopping_index:
            stop_early = True
        else:
            if epoch > self.start_early_stopping_epoch:
                print(f'current mean_AP {mean_AP} did not improve from {max(self.val_metric_values)} '
                      f'at epoch {np.argmax(self.val_metric_values)}')
                print(f'-- idx_early_stopping = {self.patience_idx} / {self.max_early_stopping_index}')
                self.patience_idx += 1
        return stop_early

    def save_and_overwrite_model(self, state_dict, dir_path: str, checkpoint_name: str, epoch: int):
        """
        saves the model and deletes old one
        """
        for file in os.listdir(dir_path):
            if file.startswith(checkpoint_name):
                print(f'deleting old checkpoint: {os.path.join(dir_path, file)}')
                os.remove(os.path.join(dir_path, file))
        print('saving model to {}'.format(os.path.join(dir_path, checkpoint_name + f'_{epoch}')))
        torch.save(state_dict, os.path.join(dir_path, checkpoint_name + f'_{epoch}'))


def get_models(flags, modality):
    """
    Get the wanted classifier for specific modality
    """
    if modality in ['PA', 'Lateral']:
        if flags.img_clf_type == 'cheXnet':
            model = CheXNet(len(LABELS)).cuda()
        elif flags.img_clf_type == 'resnet':
            model = ClfImg(flags, LABELS).to(flags.device)
        else:
            raise NotImplementedError(f'{flags.img_clf_type} is not implemented, chose between "cheXnet" and "resnet"')
    elif modality == 'text':
        model = ClfText(flags, LABELS).to(flags.device)
    if torch.cuda.device_count() > 1:
        print(f'Training with {torch.cuda.device_count()} GPUs')
        model = torch.nn.DataParallel(model)
        # todo pytorch recommends to use DistributedDataParallel instead of DataParallel
    return model


class ExperimentDf:
    """
    clf experiment results dataframe which contains test results of previous experiments together with their
    parameters
    """

    def __init__(self, flags):
        self.testing = flags.dataset == 'testing'
        experiment_uid = flags.experiment_uid
        if os.path.exists('clf_experiments_dataframe.csv'):
            experiments_dataframe = pd.read_csv('clf_experiments_dataframe.csv')
            flags_dict = vars(flags)
            flags_dict['experiment_uid'] = experiment_uid
            flags_dict['total_epochs'] = 0
            flags_dict['experiment_duration'] = -1
            self.experiments_dataframe = experiments_dataframe.append(flags_dict, ignore_index=True)
            self.experiments_dataframe.to_csv('clf_experiments_dataframe.csv', index=False)
        else:
            experiments_dataframe = pd.DataFrame()
            flags_dict = vars(flags)
            flags_dict['clf_experiment_uid'] = experiment_uid
            self.experiments_dataframe = experiments_dataframe.append(flags_dict, ignore_index=True)
            self.experiments_dataframe.to_csv('clf_experiments_dataframe.csv', index=False)
        self.experiment_uid = experiment_uid
        self.start_time = timer()

    def update_experiments_dataframe(self, values_dict: dict):
        """
        Updates the values in experiments dataframe with the new values from the values_dict and saves it if the
        experiment is not a test run
        """
        experiments_dataframe = pd.read_csv('clf_experiments_dataframe.csv')
        for key in values_dict:
            experiments_dataframe.loc[
                experiments_dataframe['experiment_uid'] == self.experiment_uid, key] = values_dict[key]
        if not self.testing:
            experiments_dataframe.to_csv('clf_experiments_dataframe.csv', index=False)

    def write_experiment_time(self):
        self.update_experiments_dataframe({'experiment_duration': (timer() - self.start_time) // 60})


def set_clf_paths(flags):
    """
    dir_clf: path to the directory where the classifier checkpoints will be saved
    clf_save_m{1,2,3}: filename of the classifier checkpoint
    dir_logs_clf: path to the directory where the training logs will be saved
    """
    flags.experiment_uid = get_str_experiments(flags, prefix=f'clf_{flags.modality}')
    flags.dir_logs_clf = os.path.join(os.path.expanduser(flags.dir_clf), 'logs', flags.experiment_uid)
    create_dir(flags.dir_logs_clf)
    # change dir_clf
    if flags.modality in ['PA', 'Lateral']:
        flags.dir_clf = os.path.expanduser(
            os.path.join(flags.dir_clf, f'Mimic{flags.img_size}_{flags.img_clf_type}'))
    else:
        flags.dir_clf = os.path.expanduser(flags.dir_clf)
    if not os.path.exists(flags.dir_clf):
        os.makedirs(flags.dir_clf)

    use_cuda = torch.cuda.is_available()

    flags.device = torch.device('cuda' if use_cuda else 'cpu')

    flags = expand_paths(flags)
    return flags
