import os
from dataclasses import dataclass
from typing import Optional
from sklearn.model_selection import ParameterGrid

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from mimic.dataio.MimicDataset import Mimic
from mimic.dataio.utils import get_data_loaders
from mimic.networks.classifiers.utils import get_input
from mimic.networks.classifiers.utils import set_clf_paths, ExperimentDf, get_models, LABELS, Callbacks
from mimic.utils.flags import parser
from mimic.utils.flags import update_flags_with_config
from mimic.utils.loss import clf_loss
from mimic.utils.filehandling import get_config_path

START_EARLY_STOPPING_EPOCH = 0
MAX_EARLY_STOPPING_IDX = 5


def train_clf(flags, epoch, model, data_loader: DataLoader, log_writer, modality, optimizer) -> torch.nn.Module:
    num_samples_train = data_loader.__len__()
    name_logs = f"train_clf_{modality}"
    model.train()

    num_batches_train = np.floor(num_samples_train / flags.batch_size)
    step = epoch * num_batches_train

    for idx, (batch_d, batch_l) in tqdm(enumerate(data_loader), total=num_batches_train, postfix='training epoch'):
        imgs, bs, n_crops = get_input(flags, batch_d[modality], modality)
        labels = Variable(batch_l).to(flags.device)
        optimizer.zero_grad()
        attr_hat = model(imgs)

        if flags.n_crops > 1 and flags.img_clf_type == 'cheXnet' and not modality == 'text':
            attr_hat = attr_hat.view(bs, n_crops, -1).mean(1)
        loss = clf_loss(flags, attr_hat, labels)
        loss.backward()
        optimizer.step()

        log_writer.add_scalars(f'%s/Loss {modality}' % name_logs, {modality: loss.item()}, step)

        step += 1
    return model


def eval_clf(flags, epoch, model, data_loader: DataLoader, log_writer, modality: str):
    num_samples_test = data_loader.__len__()
    name_logs = f"eval_clf_{modality}"
    model.eval()
    num_batches_eval = np.floor(num_samples_test / flags.batch_size)
    step = epoch * num_batches_eval

    losses = []
    list_predictions = []
    list_gt = []
    with torch.no_grad():
        for idx, (batch_d, batch_l) in tqdm(enumerate(data_loader), total=num_batches_eval, postfix='eval epoch'):
            imgs, bs, n_crops = get_input(flags, batch_d[modality], modality)
            gt = Variable(batch_l).to(flags.device)
            attr_hat = model(imgs)

            if flags.n_crops > 1 and flags.img_clf_type == 'cheXnet' and not modality == 'text':
                attr_hat = attr_hat.view(bs, n_crops, -1).mean(1)
            loss = clf_loss(flags, attr_hat, gt)
            list_predictions.extend(attr_hat.cpu().data.numpy().ravel())
            list_gt.extend(gt.cpu().data.numpy().ravel())
            losses.append(loss.item())
            log_writer.add_scalars('%s/Loss' % name_logs, {modality: loss.item()}, step)

            step += 1
    mean_AP = average_precision_score(list_gt, list_predictions)
    return np.mean(losses), mean_AP


def training_procedure_clf(flags, train_set: Mimic, eval_set: Mimic, modality: str, total_epochs: int = 100):
    experiment_df = ExperimentDf(flags)
    epoch = 0
    logger = SummaryWriter(flags.dir_logs_clf)
    use_cuda = torch.cuda.is_available()
    flags.device = torch.device('cuda' if use_cuda else 'cpu')
    model = get_models(flags, modality)
    # optimizer definition
    optimizer = optim.Adam(
        list(model.parameters()),
        lr=flags.initial_learning_rate)

    callbacks = Callbacks(flags, START_EARLY_STOPPING_EPOCH, MAX_EARLY_STOPPING_IDX, modality,
                          experiment_df, logger, optimizer)

    _, train_loader = get_data_loaders(flags, train_set)
    _, eval_loader = get_data_loaders(flags, eval_set)
    end = time.time()
    for epoch in tqdm(range(0, total_epochs), postfix=f'train_clf_{modality}'):
        print(f'epoch: {epoch}')
        end = time.time()
        model = train_clf(flags, epoch, model, train_loader, logger, modality, optimizer)
        loss, mean_AP = eval_clf(flags, epoch, model, eval_loader, logger, modality)
        if callbacks.update_epoch(epoch, loss, mean_AP, model, time.time() - end):
            break

    torch.save(flags, os.path.join(flags.dir_clf, f'flags_clf_{modality}_{epoch}.rar'))
    experiment_df.update_experiments_dataframe(
        {'mean_epoch_time': np.mean(callbacks.elapsed_times)})
    experiment_df.write_experiment_time()


# todo fill this dataclass with all the arguments needed to run training_procedure_clf
@dataclass
class Args:
    img_clf_type = 'resnet'
    img_size = 128
    image_channels = 10
    clf_save_m1: Optional[str] = 'temp'
    dataset: str = 'testing'
    experiment_uid: str = 'temp'
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    def __init__(self, tmpdirname: str):
        self.tempdirname = tmpdirname
        self.dir_clf = tmpdirname
        self.dir_logs_clf = tmpdirname


if __name__ == '__main__':

    parser.add_argument('--modality', type=str, default='PA', choices=['PA', 'Lateral', 'text'],
                        help="modality on which to train the image classifier, chose between PA and Lateral")
    parser.add_argument('--which_grid', type=str, default='chxnet', choices=['resnet', 'chexnet'],
                        help="modality on which to train the image classifier, chose between PA and Lateral")

    params_seach_space_cheXnet = {
        'n_crops': [1, 5, 10],
        'img_clf_type': ['cheXnet'],
        'clf_loss': ['bce_with_logits', 'crossentropy', 'binary_crossentropy'],
        'img_size': [256],
        'modality': ['PA', 'Lateral', 'text']
    }

    params_seach_space_resnet = {
        'img_clf_type': ['resnet'],
        'clf_loss': ['bce_with_logits', 'crossentropy', 'binary_crossentropy'],
        'img_size': [256, 128],
        'modality': ['PA', 'Lateral', 'text']
    }

    FLAGS = parser.parse_args()
    config_path = get_config_path()

    if FLAGS.which_grid == 'resnet':
        grid = params_seach_space_resnet
    else:
        grid = params_seach_space_cheXnet
    for params in ParameterGrid(grid):
        print(params)
        MODALITY = FLAGS.modality
        FLAGS = update_flags_with_config(FLAGS, config_path, additional_args=params)

        # temp 256
        # FLAGS.img_size = 256
        # temp batch size=30
        FLAGS.batch_size = 50
        # FLAGS.initial_learning_rate = 1e-4
        # FLAGS.n_crops = 5
        assert FLAGS.n_crops in [1, 5, 10]
        # FLAGS.img_clf_type = 'cheXnet'
        assert FLAGS.img_clf_type in ['cheXnet', 'resnet']
        # FLAGS.clf_loss = 'binary_crossentropy'
        assert FLAGS.clf_loss in ['bce_with_logits', 'crossentropy', 'binary_crossentropy']
        # temp
        FLAGS.dir_clf += '_gridsearch'
        FLAGS.distributed = False

        set_clf_paths(FLAGS)

        # This will overwrite old classifiers!!
        mimic_train = Mimic(FLAGS, LABELS, split='train')
        mimic_eval = Mimic(FLAGS, LABELS, split='eval')
        print(f'Training image classifier {FLAGS.img_clf_type} with modality: {FLAGS.modality} '
              f'for images of size {FLAGS.img_size}')
        print(f'clf_log_dir: {FLAGS.dir_logs_clf}')
        training_procedure_clf(FLAGS, mimic_train, mimic_eval, MODALITY, total_epochs=600)
