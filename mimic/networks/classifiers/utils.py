import os
from timeit import default_timer as timer
from typing import Optional
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mimic.networks.CheXNet import CheXNet
from mimic.networks.ConvNetworkImgClf import ClfImg
from mimic.networks.ConvNetworkTextClf import ClfText
from mimic.utils.filehandling import create_dir
from mimic.utils.filehandling import expand_paths
from mimic.utils.filehandling import get_str_experiments
import typing
from sklearn.metrics import average_precision_score

LABELS = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']


class ExperimentDfProto(Protocol):
    dataset: str
    experiment_uid: str


class ExperimentDf:
    """
    clf experiment results dataframe which contains test results of previous experiments together with their
    parameters
    """

    def __init__(self, flags: ExperimentDfProto):
        self.testing = flags.dataset == 'testing'
        experiment_uid = flags.experiment_uid
        if os.path.exists('clf_experiments_dataframe.csv'):
            experiments_dataframe = pd.read_csv('clf_experiments_dataframe.csv')
            flags_dict = vars(flags)
            flags_dict['experiment_uid'] = experiment_uid
            flags_dict['total_epochs'] = 0
            flags_dict['experiment_duration'] = -1
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
            if isinstance(values_dict[key], list) and len(values_dict[key]) == 1:
                value = values_dict[key][0]
            else:
                value = values_dict[key]
            experiments_dataframe.loc[
                experiments_dataframe['experiment_uid'] == self.experiment_uid, key] = value
        if not self.testing:
            experiments_dataframe.to_csv('clf_experiments_dataframe.csv', index=False)

    def write_experiment_time(self):
        self.update_experiments_dataframe({'experiment_duration': (timer() - self.start_time) // 60})


class CallbacksProto(Protocol):
    clf_save_m1: Optional[str]
    clf_save_m2: Optional[str]
    clf_save_m3: Optional[str]
    dir_clf: str


class Callbacks:
    def __init__(self, flags: CallbacksProto,
                 start_early_stopping_epoch: int, max_early_stopping_index,
                 modality: str,
                 experiment_df: ExperimentDf, logger: any, optimizer: torch.optim):
        self.modality = modality
        self.logger = logger
        self.flags = flags
        self.experiment_df = experiment_df
        self.start_early_stopping_epoch = start_early_stopping_epoch
        self.max_early_stopping_index = max_early_stopping_index
        self.patience_idx = 1
        self.scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
        self.elapsed_times = []
        self.metrics = {}

    def update_epoch(self, epoch: int, loss, val_results: typing.Dict[str, torch.Tensor], model, elapsed_time):
        # calculate metrics
        metrics = Metrics(val_results['predictions'], val_results['ground_truths'])
        metrics_dict = metrics.evaluate()
        mean_AP = metrics_dict['mean_AP'][0]
        self._update_metrics(metrics_dict)

        stop_early = False
        self.elapsed_times.append(elapsed_time)
        self.scheduler.step(loss)

        # update logger
        for k, v in metrics_dict.items():
            self.logger.add_scalars(f'eval_clf_{self.modality}/k', {self.modality: v}, epoch)
        self.logger.add_scalars(f'eval_clf_{self.modality}/mean_loss', {self.modality: loss}, epoch)

        # evaluate progress
        print(f'current eval loss: {loss}, metrics: {metrics_dict}')
        if epoch > self.start_early_stopping_epoch and mean_AP >= max(self.metrics["mean_AP"]):
            print(f'current mean AP {mean_AP} improved from {max(self.metrics["mean_AP"])}'
                  f' at epoch {np.argmax(self.metrics["mean_AP"])}')
            self.experiment_df.update_experiments_dataframe(
                {'mean_eval_loss': loss, 'total_epochs': epoch, **metrics_dict})

            self._save_and_overwrite_model(model.state_dict(), epoch)
            self.patience_idx = 1
        elif self.patience_idx > self.max_early_stopping_index:
            print(
                f'stopping early at epoch {epoch} because current mean_AP {mean_AP} '
                f'did not improve from {max(self.metrics["mean_AP"])} at epoch {np.argmax(self.metrics["mean_AP"])}')
            stop_early = True
        else:
            if epoch > self.start_early_stopping_epoch:
                print(f'current mean_AP {mean_AP} did not improve from {max(self.metrics["mean_AP"])} '
                      f'at epoch {np.argmax(self.metrics["mean_AP"])}')
                print(f'-- idx_early_stopping = {self.patience_idx} / {self.max_early_stopping_index}')
                self.patience_idx += 1

        return stop_early

    def _update_metrics(self, metrics_dict: typing.Dict[str, list]):
        if not self.metrics:
            self.metrics = metrics_dict
        else:
            for k, v in metrics_dict.items():
                self.metrics[k].extend(v)

    def _save_and_overwrite_model(self, state_dict, epoch: int):
        """
        saves the model to flags.dir_clf/flags.clf_save_m[1,2,3] and deletes old one
        """
        if self.modality == 'PA':
            filename = self.flags.clf_save_m1
        elif self.modality == 'Lateral':
            filename = self.flags.clf_save_m2
        else:
            filename = self.flags.clf_save_m3

        for file in os.listdir(self.flags.dir_clf):
            if file.startswith(filename):
                print(f'deleting old checkpoint: {os.path.join(self.flags.dir_clf, file)}')
                os.remove(os.path.join(self.flags.dir_clf, file))
        print('saving model to {}'.format(os.path.join(self.flags.dir_clf, filename + f'_{epoch}')))
        torch.save(state_dict, os.path.join(self.flags.dir_clf, filename + f'_{epoch}'))


class GetModelsProto(Protocol):
    device: any
    img_clf_type: str
    distributed: bool
    fixed_extractor: bool


def get_models(flags: GetModelsProto, modality: str):
    """
    Get the wanted classifier for specific modality
    """
    # argument feature_extractor_img is only used for mimic_main.
    # Need to make sure it is unset when training classifiers
    flags.feature_extractor_img = ''
    assert modality in ['PA', 'Lateral', 'text']
    assert flags.img_clf_type in ['densenet', 'resnet', '']

    if modality in ['PA', 'Lateral']:
        if flags.img_clf_type == 'densenet':
            model = CheXNet(len(LABELS), flags.fixed_extractor).cuda()
        elif flags.img_clf_type == 'resnet':
            model = ClfImg(flags, LABELS).to(flags.device)
        else:
            raise NotImplementedError(f'{flags.img_clf_type} is not implemented, chose between "densenet" and "resnet"')

    elif modality == 'text':
        model = ClfText(flags, LABELS).to(flags.device)
    if flags.distributed and torch.cuda.device_count() > 1:
        print(f'Training with {torch.cuda.device_count()} GPUs')
        model = torch.nn.DataParallel(model)

    return model


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


def get_imgs_from_crops(input: torch.Tensor, device):
    """
    Reshapes the input such that the number of crops and the batch size are multiplied in the first dimension.
    """
    bs, n_crops, c, h, w = input.size()
    imgs = Variable(input.view(-1, c, h, w)).to(device)
    return imgs, bs, n_crops


def get_input(args: any, input: torch.Tensor, modality):
    if args.img_clf_type == 'densenet' and modality != 'text' and args.n_crops in [5, 10]:
        imgs, bs, n_crops = get_imgs_from_crops(input, args.device)
    else:
        imgs = Variable(input).to(args.device)
        bs = None
        n_crops = 1
    return imgs, bs, n_crops


class Metrics(object):
    """
    Defines a set of metrics that are used to evaluate the performance of a model
    Modified version of https://github.com/ParGG/MasterThesisOld/blob/44f7b93214fa16494ebaeef7763ff81943b5ffc3/losses.py#L142
    """

    def __init__(self, prediction: torch.Tensor, groundtruth: torch.Tensor):
        """
        params:
            prediction: Tensor which is given as output of the network
            groundtruth: Tensor which resembles the goundtruth
        """
        self.prediction: torch.Tensor = (prediction > 0.5) * 1
        self.groundtruth: torch.Tensor = (groundtruth > 0.5) * 1

    def evaluate(self) -> typing.Dict[str, list]:
        """
        Computes the different metrics (accuracy, recall, specificity, precision, f1 score, jaccard score, dice score).
        NOTE: f1 and dice are the same
        """
        return {
            'accuracy': [Metrics.accuracy(self)],
            'recall': [Metrics.recall(self)],
            'specificity': [Metrics.specificity(self)],
            'precision': [Metrics.precision(self)],
            'f1': [Metrics.f1(self)],
            'jaccard': [Metrics.jaccard(self)],
            'dice': [Metrics.dice(self)],
            'mean_AP': [self.mean_AP()],
        }

    def accuracy(self) -> float:
        """
        Computes the accuracy
        """
        self.INTER = torch.mul(self.prediction, self.groundtruth).sum()
        self.INTER_NEG = torch.mul(1 - self.prediction, 1 - self.groundtruth).sum()
        self.TOTAL = self.prediction.nelement()
        return float(self.INTER + self.INTER_NEG) / float(self.TOTAL)

    def recall(self) -> float:
        """
        Computes the recall
        """
        self.TP = torch.mul(self.prediction, self.groundtruth).sum()
        self.FN = torch.mul(1 - self.prediction, self.groundtruth).sum()

        self.RC = float(self.TP) / (float(self.TP + self.FN) + 1e-6)

        return self.RC

    def specificity(self):
        self.TN = torch.mul(1 - self.prediction, 1 - self.groundtruth).sum()
        self.FP = torch.mul(self.prediction, 1 - self.groundtruth).sum()

        self.SP = float(self.TN) / (float(self.TN + self.FP) + 1e-6)

        return self.SP

    def precision(self) -> float:
        """
        Computes the precision
        """
        self.PC = float(self.TP) / (float(self.TP + self.FP) + 1e-6)

        return self.PC

    def f1(self) -> float:
        """
        Computes the f1 score (same as dice)
        """
        F1 = 2 * (self.RC * self.PC) / (self.RC + self.PC + 1e-6)

        return F1

    def jaccard(self) -> float:
        """
        Computes the jaccard score
        """
        JS = float(self.INTER) / (float(self.INTER + self.FP + self.FN) + 1e-6)

        return JS

    def dice(self):
        """
        Computes the dice score (same as f1)
        """
        DC = 2 * float(self.INTER) / (float(2 * self.INTER + self.FP + self.FN) + 1e-6)

        return DC

    def mean_AP(self):
        """
        Computes the mean average precision
        """
        return average_precision_score(self.prediction.cpu().data.numpy().ravel(),
                                       self.groundtruth.cpu().data.numpy().ravel())
