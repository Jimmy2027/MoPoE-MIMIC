import os
import time
from contextlib import contextmanager
from typing import Union

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from mimic.dataio.MimicDataset import Mimic, Mimic_testing
from mimic.dataio.utils import get_data_loaders
from mimic.networks.classifiers.utils import get_input
from mimic.networks.classifiers.utils import set_clf_paths, ExperimentDf, get_models, LABELS, Callbacks
from mimic.utils.exceptions import CudaOutOfMemory
from mimic.utils.filehandling import get_config_path
from mimic.utils.flags import parser
from mimic.utils.flags import update_flags_with_config
from mimic.utils.loss import clf_loss

START_EARLY_STOPPING_EPOCH = 0
MAX_EARLY_STOPPING_IDX = 5


@contextmanager
def catching_cuda_out_of_memory(batch_size):
    """
    Context that throws CudaOutOfMemory error if GPU is out of memory.
    """
    try:
        yield
    # if the GPU runs out of memory, start the experiment again with a smaller batch size
    except RuntimeError as e:
        if (str(e).startswith('CUDA out of memory.') or str(e).startswith(
                'Caught RuntimeError in replica')) and batch_size > 10:
            raise CudaOutOfMemory(e)
        else:
            raise e


def train_clf(flags, epoch, model, data_loader: DataLoader, log_writer, modality, optimizer) -> torch.nn.Module:
    num_samples_train = data_loader.__len__()
    name_logs = f"train_clf_{modality}"
    model.train()

    num_batches_train = np.floor(num_samples_train / flags.batch_size)
    print(f'total training epoch steps {num_batches_train}')
    step = epoch * num_batches_train

    for idx, (batch_d, batch_l) in tqdm(enumerate(data_loader), total=num_batches_train, postfix='training epoch'):
        imgs, bs, n_crops = get_input(flags, batch_d[modality], modality)
        labels = Variable(batch_l).to(flags.device)
        optimizer.zero_grad()

        with catching_cuda_out_of_memory(flags.batch_size):
            attr_hat = model(imgs)

        if (
                flags.n_crops > 1
                and flags.img_clf_type == 'cheXnet'
                and modality != 'text'
        ):
            attr_hat = attr_hat.view(bs, n_crops, -1).mean(1)
        loss = clf_loss(flags, attr_hat, labels)
        loss.backward()
        optimizer.step()

        log_writer.add_scalars(f'%s/Loss {modality}' % name_logs, {modality: loss.item()}, step)

        step += 1
    return model


def eval_clf(flags, epoch, model, data_loader: DataLoader, log_writer, modality: str):
    num_samples_test = data_loader.__len__()
    model.eval()
    num_batches_eval = np.floor(num_samples_test / flags.batch_size)
    step = epoch * num_batches_eval

    losses = []
    list_predictions = []
    list_gt = []
    with torch.no_grad():
        name_logs = f"eval_clf_{modality}"
        for idx, (batch_d, batch_l) in tqdm(enumerate(data_loader), total=num_batches_eval, postfix='eval epoch'):
            imgs, bs, n_crops = get_input(flags, batch_d[modality], modality)
            gt = Variable(batch_l).to(flags.device)
            attr_hat = model(imgs)

            if (
                    flags.n_crops > 1
                    and flags.img_clf_type == 'cheXnet'
                    and modality != 'text'
            ):
                attr_hat = attr_hat.view(bs, n_crops, -1).mean(1)
            loss = clf_loss(flags, attr_hat, gt)
            list_predictions.extend(attr_hat.cpu().data.numpy().ravel())
            list_gt.extend(gt.cpu().data.numpy().ravel())
            losses.append(loss.item())
            log_writer.add_scalars('%s/Loss' % name_logs, {modality: loss.item()}, step)

            step += 1
    mean_AP = average_precision_score(list_gt, list_predictions)
    return np.mean(losses), mean_AP


def training_procedure_clf(flags, train_set: Union[Mimic, Mimic_testing], eval_set: Union[Mimic, Mimic_testing],
                           modality: str, total_epochs: int = 100) -> bool:
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
    for epoch in tqdm(range(total_epochs), total=total_epochs, postfix=f'train_clf_{modality}'):
        print(f'epoch: {epoch}')
        end = time.time()
        with catching_cuda_out_of_memory(flags.batch_size):
            model = train_clf(flags, epoch, model, train_loader, logger, modality, optimizer)
        loss, mean_AP = eval_clf(flags, epoch, model, eval_loader, logger, modality)
        if callbacks.update_epoch(epoch, loss, mean_AP, model, time.time() - end):
            break

    torch.save(flags, os.path.join(flags.dir_clf, f'flags_clf_{modality}_{epoch}.rar'))
    experiment_df.update_experiments_dataframe(
        {'mean_epoch_time': np.mean(callbacks.elapsed_times)})
    experiment_df.write_experiment_time()

    return True


def run_training_procedure_clf(flags):
    """
    Runs training_procedure_clf, and restarts with smaller batch size if CUDA out of memory occurs.
    """
    set_clf_paths(flags)

    # This will overwrite old classifiers!!
    mimic_train = Mimic(flags, LABELS, split='train')
    mimic_eval = Mimic(flags, LABELS, split='eval')
    print(f'Training image classifier {flags.img_clf_type} with modality: {flags.modality} '
          f'for images of size {flags.img_size} and with batch_size {flags.batch_size}')
    print(f'clf_log_dir: {flags.dir_logs_clf}')

    success = False
    while not success:
        try:
            success = training_procedure_clf(flags, mimic_train, mimic_eval, flags.modality, total_epochs=600)

        except CudaOutOfMemory as e:
            print(e)
            old_bs = flags.batch_size

            flags.batch_size = int(flags.batch_size * 0.7)
            print(f'********  GPU ran out of memory with batch size {old_bs}, '
                  f'trying again with batch size: {flags.batch_size}  ********')


if __name__ == '__main__':
    """
    Script to train classifiers on the MIMIC-database.
    """
    parser.add_argument('--modality', type=str, default='PA', choices=['PA', 'Lateral', 'text'],
                        help="modality on which to train the image classifier, chose between PA and Lateral")

    FLAGS = parser.parse_args()
    config_path = get_config_path()

    FLAGS = update_flags_with_config(FLAGS, config_path)

    run_training_procedure_clf(FLAGS)
