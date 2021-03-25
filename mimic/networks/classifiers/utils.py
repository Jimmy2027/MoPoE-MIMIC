import os
import typing
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mimic import log
from mimic.networks.CheXNet import CheXNet
from mimic.networks.ConvNetworkImgClf import ClfImg
from mimic.networks.ConvNetworkTextClf import ClfText
from mimic.utils.filehandling import create_dir
from mimic.utils.filehandling import expand_paths
from mimic.utils.filehandling import get_str_experiments


def save_flags(flags, modality, epoch):
    if modality == 'text':
        save_name = f'flags_clf_{modality}_vocabsize_{flags.vocab_size}' \
                    f'{"_bin_label" if flags.binary_labels else ""}_{epoch}.rar'
    else:
        save_name = f'flags_clf_{modality}_{epoch}.rar'

    torch.save(flags, Path(flags.dir_clf) / save_name)


def get_labels(binary_labels: bool = False):
    if binary_labels:
        return ['Finding']
    else:
        return ['Lung Opacity', 'Pleural Effusion', 'Support Devices']


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
        log.info(f"writing to experiment df: {values_dict}")
        experiments_dataframe = pd.read_csv('clf_experiments_dataframe.csv')
        for key, value_ in values_dict.items():
            if isinstance(values_dict[key], list) and len(value_) == 1:
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
    dir_logs_clf: str
    binary_labels: bool
    vocab_size: int


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
        # metrics is a dict that will contain a list for each metric, containing a value for each epoch
        self.metrics: typing.Mapping[str, list] = {}
        # self.early_stopping_crit = 'mean_AP_total'
        self.early_stopping_crit = 'dice'
        # maximize metric or minimize
        self.early_stopping_mode = 'maximize'

    def plot_metrics(self):
        for k, v in self.metrics.items():
            plt.plot(v)
            plt.title(k)
            plt.savefig(os.path.join(self.flags.dir_logs_clf, f'{k}.png'))
            plt.close()

    def update_epoch(self, epoch: int, loss, val_results: typing.Dict[str, torch.Tensor], model, elapsed_time):
        # calculate metrics
        metrics = Metrics(val_results['predictions'], val_results['ground_truths'],
                          str_labels=get_labels(self.flags.binary_labels))
        metrics_dict = metrics.evaluate()
        metrics_dict['eval_loss'] = [loss]
        early_stop_crit_val = metrics_dict[self.early_stopping_crit][0]

        self._update_metrics(metrics_dict)

        stop_early = False
        self.elapsed_times.append(elapsed_time)
        self.scheduler.step(loss)

        # update logger
        for k, v in metrics_dict.items():
            k = k.replace(' ', '_')
            self.logger.add_scalars(f'eval_clf_{self.modality}/{k}', {self.modality: v}, epoch)
        self.logger.add_scalars(f'eval_clf_{self.modality}/mean_loss', {self.modality: loss}, epoch)

        # evaluate progress
        max_eval_metric = max(self.metrics[self.early_stopping_crit])
        epoch_max_eval_metric = np.argmax(self.metrics[self.early_stopping_crit])

        print(f'current eval loss: {loss}, metrics: {metrics_dict}')
        if epoch >= self.start_early_stopping_epoch and early_stop_crit_val >= max_eval_metric:
            print(
                f'current {self.early_stopping_crit} {early_stop_crit_val} improved from {max_eval_metric}'
                f' at epoch {epoch_max_eval_metric}')
            self.experiment_df.update_experiments_dataframe(
                {'mean_eval_loss': loss, 'total_epochs': epoch, **metrics_dict})

            self._save_and_overwrite_model(model.state_dict(), epoch)
            self.patience_idx = 1
        elif self.patience_idx > self.max_early_stopping_index:
            print(
                f'stopping early at epoch {epoch} because current {self.early_stopping_crit} {early_stop_crit_val} '
                f'did not improve from {max_eval_metric} at epoch {epoch_max_eval_metric}')
            stop_early = True
        else:
            if epoch > self.start_early_stopping_epoch:
                print(
                    f'current {self.early_stopping_crit} {early_stop_crit_val} did not improve from {max_eval_metric} '
                    f'at epoch {epoch_max_eval_metric}')
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
            filename = self.flags.clf_save_m3 + \
                       f'vocabsize_{self.flags.vocab_size}{"_bin_label" if self.flags.binary_labels else ""}'

        for file in os.listdir(self.flags.dir_clf):
            if file.startswith(filename):
                print(f'deleting old checkpoint: {os.path.join(self.flags.dir_clf, file)}')
                os.remove(os.path.join(self.flags.dir_clf, file))
        log.info('saving model to {}'.format(os.path.join(self.flags.dir_clf, filename + f'_{epoch}')))
        torch.save(state_dict, os.path.join(self.flags.dir_clf, filename + f'_{epoch}'))


class GetModelsProto(Protocol):
    device: any
    img_clf_type: str
    distributed: bool
    fixed_extractor: bool
    binary_labels: bool


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
            model = CheXNet(len(get_labels(flags.binary_labels)), flags.fixed_extractor).cuda()
        elif flags.img_clf_type == 'resnet':
            model = ClfImg(flags, get_labels(flags.binary_labels)).to(flags.device)
        else:
            raise NotImplementedError(f'{flags.img_clf_type} is not implemented, chose between "densenet" and "resnet"')

    elif modality == 'text':
        model = ClfText(flags, get_labels(flags.binary_labels)).to(flags.device)
    if flags.distributed and torch.cuda.device_count() > 1:
        print(f'Training with {torch.cuda.device_count()} GPUs')
        model = torch.nn.DataParallel(model)

    return model


def set_clf_paths(flags):
    """
    Used for the trianing of the classifiers.
    dir_clf: path to the directory where the classifier checkpoints will be saved
    clf_save_m{1,2,3}: filename of the classifier checkpoint
    dir_logs_clf: path to the directory where the training logs will be saved
    """
    flags.exp_str_prefix = f'clf_{flags.modality}' + f'{flags.exp_str_prefix}' * bool(flags.exp_str_prefix)
    flags.experiment_uid = get_str_experiments(flags)
    # flags.dir_logs_clf = os.path.join(os.path.expanduser(flags.dir_clf), 'logs', flags.experiment_uid)
    flags.dir_logs_clf = Path(flags.dir_clf).expanduser() / f'logs/{flags.experiment_uid}'
    create_dir(flags.dir_logs_clf)
    # change dir_clf
    if flags.modality in ['PA', 'Lateral']:
        flags.dir_clf = os.path.expanduser(
            os.path.join(flags.dir_clf,
                         f'Mimic{flags.img_size}_{flags.img_clf_type}{"_bin_label" if flags.binary_labels else ""}'))
    else:
        flags.dir_clf = Path(flags.dir_clf).expanduser()
    if not os.path.exists(flags.dir_clf):
        os.makedirs(flags.dir_clf)

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
        imgs = input.to(args.device)
        bs = None
        n_crops = 1
    return imgs, bs, n_crops


class Metrics(object):
    """
    Defines a set of metrics that are used to evaluate the performance of a model
    Modified version of https://github.com/ParGG/MasterThesisOld/blob/44f7b93214fa16494ebaeef7763ff81943b5ffc3/losses.py#L142
    """

    def __init__(self, prediction: torch.Tensor, groundtruth: torch.Tensor, str_labels):
        """
        params:
            prediction: Tensor which is given as output of the network
            groundtruth: Tensor which resembles the goundtruth
        >>> import torch
        >>> metrics = Metrics(torch.ones((10,1)), torch.ones((10,1)), str_labels=['my_labels'])
        """
        self.str_labels = str_labels
        self.prediction = prediction
        self.groundtruth = groundtruth
        self.prediction_bin: torch.Tensor = (prediction > 0.5) * 1
        self.groundtruth_bin: torch.Tensor = (groundtruth > 0.5) * 1
        # classwise binarized predictions
        self.class_pred_bin: dict = {str_labels[i]: self.prediction_bin[:, i] for i in range(len(str_labels))}
        self.class_gt_bin: dict = {str_labels[i]: self.groundtruth_bin[:, i] for i in range(len(str_labels))}

    def evaluate(self) -> typing.Dict[str, list]:
        """
        Computes the different metrics (accuracy, recall, specificity, precision, f1 score, jaccard score, dice score).
        NOTE: f1 and dice are the same
        """

        return {**{
            'accuracy': [Metrics.accuracy(self)],
            'recall': [Metrics.recall(self)],
            'specificity': [Metrics.specificity(self)],
            'precision': [Metrics.precision(self)],
            'f1': [Metrics.f1(self)],
            'jaccard': [Metrics.jaccard(self)],
            'dice': [Metrics.dice(self)],

        },
                **self.mean_AP(), **self.counts()
                }

    def extract_values(self, results: dict):
        """
        Extract first values from list for each metric result.
        >>> import torch
        >>> metrics = Metrics(torch.ones((10,1)), torch.ones((10,1)), str_labels=['my_labels'])
        >>> metrics.extract_values(results={'accuracy':[0.9], 'f1': [0.8], 'recall':[0.6]})
        {'accuracy': 0.9, 'f1': 0.8, 'recall': 0.6}
        """
        return {k: v[0] for k, v in results.items()}

    def accuracy(self) -> float:
        """
        Computes the accuracy
        """
        self.INTER = torch.mul(self.prediction_bin, self.groundtruth_bin).sum()
        self.INTER_NEG = torch.mul(1 - self.prediction_bin, 1 - self.groundtruth_bin).sum()
        self.TOTAL = self.prediction_bin.nelement()
        return float(self.INTER + self.INTER_NEG) / float(self.TOTAL)

    def recall(self) -> float:
        """
        Computes the recall
        """
        self.TP = torch.mul(self.prediction_bin, self.groundtruth_bin).sum()
        self.FN = torch.mul(1 - self.prediction_bin, self.groundtruth_bin).sum()

        self.RC = float(self.TP) / (float(self.TP + self.FN) + 1e-6)

        return self.RC

    def specificity(self):
        self.TN = torch.mul(1 - self.prediction_bin, 1 - self.groundtruth_bin).sum()
        self.FP = torch.mul(self.prediction_bin, 1 - self.groundtruth_bin).sum()

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
        return 2 * (self.RC * self.PC) / (self.RC + self.PC + 1e-6)

    def jaccard(self) -> float:
        """
        Computes the jaccard score
        """
        return float(self.INTER) / (float(self.INTER + self.FP + self.FN) + 1e-6)

    def dice(self):
        """
        Computes the dice score (same as f1)
        """
        return 2 * float(self.INTER) / (float(2 * self.INTER + self.FP + self.FN) + 1e-6)

    def mean_AP(self) -> dict:
        """
        Compute the mean average precision.
        >>> import torch
        >>> metrics = Metrics(torch.tensor([0, 0, 1, 1]).unsqueeze(-1), torch.tensor([0.1, 0.4, 0.35, 0.8]).unsqueeze(-1), str_labels=['my_labels'])
        >>> metrics.mean_AP()
        {'mean_AP_my_labels': [0.8333333333333333], 'mean_AP_total': [0.8333333333333333]}
        """
        ap_values = {
            f'mean_AP_{self.str_labels[i]}': [
                average_precision_score(self.prediction[:, i].numpy().ravel(), self.groundtruth[:, i].numpy().ravel())]
            for i in range(len(self.str_labels))}
        ap_values['mean_AP_total'] = [average_precision_score(self.prediction.cpu().data.numpy().ravel(),
                                                              self.groundtruth.cpu().data.numpy().ravel())]
        return ap_values

    def counts(self) -> dict:
        predicted_counts = {f'pred_count_{label}': [self.class_pred_bin[label].sum().item()] for label in
                            self.str_labels}
        gt_counts = {f'gt_count_{label}': [self.class_gt_bin[label].sum().item()] for label in self.str_labels}

        return {**predicted_counts, **gt_counts}
