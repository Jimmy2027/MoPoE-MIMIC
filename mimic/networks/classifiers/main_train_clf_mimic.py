import os
import time
import typing
from contextlib import contextmanager
from typing import Union

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from mimic.dataio.MimicDataset import Mimic, Mimic_testing
from mimic.dataio.utils import get_data_loaders
from mimic.networks.classifiers.utils import get_input
from mimic.networks.classifiers.utils import set_clf_paths, ExperimentDf, get_models, get_labels, Callbacks
from mimic.utils.exceptions import CudaOutOfMemory
from mimic.utils.filehandling import get_config_path
from mimic.utils.flags import parser
from mimic.utils.flags import setup_flags
from mimic.utils.flags import update_flags_with_config
from mimic.utils.loss import get_clf_loss
from mimic.utils.utils import at_most_n

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
        if (str(e).startswith('CUDA out of memory.')
            or str(e).startswith('Caught RuntimeError in replica')
            or str(e).startswith('DataLoader worker')) \
                and batch_size > 10:
            raise CudaOutOfMemory(e)
        else:
            raise e


def train_clf(flags, epoch, model, data_loader: DataLoader, log_writer, modality, optimizer,
              criterion) -> torch.nn.Module:
    num_samples_train = data_loader.__len__()
    name_logs = f"train_clf_{modality}"
    model.train()
    running_loss = 0.0

    step = epoch * num_samples_train

    for idx, (batch_d, batch_l) in tqdm(enumerate(at_most_n(data_loader, None)), total=len(data_loader),
                                        postfix='training epoch'):
        imgs, bs, n_crops = get_input(flags, batch_d[modality], modality)
        labels = batch_l.to(flags.device)
        optimizer.zero_grad()

        with catching_cuda_out_of_memory(flags.batch_size):
            attr_hat = model(imgs)

        if (
                flags.n_crops > 1
                and flags.img_clf_type == 'densenet'
                and modality != 'text'
        ):
            attr_hat = attr_hat.view(bs, n_crops, -1).mean(1)
        loss = criterion(attr_hat, labels)

        loss.backward()
        optimizer.step()

        log_writer.add_scalars(f'%s/Loss_{modality}' % name_logs, {modality: loss.item()}, step)

        step += 1

        # print statistics
        running_loss += loss.item()
        N = int(len(data_loader) / 5)
        if idx % N == N - 1:  # print every N mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, idx + 1, running_loss / N))
            running_loss = 0.0
    return model


def eval_clf(flags, epoch, model, data_loader: DataLoader, log_writer, modality: str, criterion) -> typing.Tuple[
    np.ndarray, typing.Dict[str, torch.Tensor]]:
    num_samples_eval = data_loader.__len__()
    model.eval()

    step = epoch * num_samples_eval

    losses = []
    predictions = torch.Tensor()
    gts = torch.Tensor()
    with torch.no_grad():
        name_logs = f"eval_clf_{modality}"
        for idx, (batch_d, batch_l) in tqdm(enumerate(data_loader), total=len(data_loader), postfix='eval epoch'):
            imgs, bs, n_crops = get_input(flags, batch_d[modality], modality)
            gt = Variable(batch_l).to(flags.device)
            attr_hat = model(imgs)

            if (
                    flags.n_crops > 1
                    and flags.img_clf_type == 'densenet'
                    and modality != 'text'
            ):
                attr_hat = attr_hat.view(bs, n_crops, -1).mean(1)
            loss = criterion(attr_hat, gt)

            predictions = torch.cat((predictions, attr_hat.cpu()), 0)
            gts = torch.cat((gts, gt.cpu()), 0)

            losses.append(loss.item())
            log_writer.add_scalars('%s/Loss' % name_logs, {modality: loss.item()}, step)

            step += 1
    val_results = {'predictions': predictions, 'ground_truths': gts}
    return np.mean(losses), val_results


def training_procedure_clf(flags, train_set: Union[Mimic, Mimic_testing], eval_set: Union[Mimic, Mimic_testing],
                           modality: str, total_epochs: int = 100) -> bool:
    experiment_df = ExperimentDf(flags)
    epoch = 0
    logger = SummaryWriter(flags.dir_logs_clf)

    model = get_models(flags, modality)
    # optimizer definition
    optimizer = optim.Adam(
        list(model.parameters()),
        lr=flags.initial_learning_rate)

    criterion = get_clf_loss(flags.clf_loss)

    callbacks = Callbacks(flags, flags.start_early_stopping_epoch, flags.max_early_stopping_index, modality,
                          experiment_df, logger, optimizer)

    _, train_loader = get_data_loaders(flags, train_set, weighted_sampler=flags.weighted_sampler, which_set='train')
    _, eval_loader = get_data_loaders(flags, eval_set, weighted_sampler=flags.weighted_sampler, which_set='eval')

    end = time.time()
    for epoch in tqdm(range(total_epochs), total=total_epochs, postfix=f'train_clf_{modality}'):
        print(f'epoch: {epoch}')
        end = time.time()
        with catching_cuda_out_of_memory(flags.batch_size):
            model = train_clf(flags, epoch, model, train_loader, logger, modality, optimizer, criterion)

        loss, val_results = eval_clf(flags, epoch, model, eval_loader, logger, modality, criterion)
        if callbacks.update_epoch(epoch, loss, val_results, model, time.time() - end):
            break

        if epoch % 5 == 4:
            # plot evolution of metrics every 5 epochs
            callbacks.plot_metrics()

    # after training:
    torch.save(flags, os.path.join(flags.dir_clf, f'flags_clf_{modality}_{epoch}.rar'))
    experiment_df.update_experiments_dataframe({'mean_epoch_time': np.mean(callbacks.elapsed_times)})
    experiment_df.write_experiment_time()

    return True


def run_training_procedure_clf(flags):
    """
    Runs training_procedure_clf, and restarts with smaller batch size if CUDA out of memory occurs.
    """
    flags = setup_flags(flags)

    set_clf_paths(flags)

    # This will overwrite old classifiers!!
    mimic_train = Mimic(flags, get_labels(flags.binary_labels), split='train', clf_training=True)
    mimic_eval = Mimic(flags, get_labels(flags.binary_labels), split='eval', clf_training=True)
    print(f'Training image classifier {flags.img_clf_type} with modality: {flags.modality} '
          f'for images of size {flags.img_size} and with batch_size {flags.batch_size}')
    print(f'clf_log_dir: {flags.dir_logs_clf}')

    success = False
    while not success:
        try:
            success = training_procedure_clf(flags, train_set=mimic_train, eval_set=mimic_eval, modality=flags.modality,
                                             total_epochs=100)

        except CudaOutOfMemory as e:
            print(e)
            old_bs = flags.batch_size

            flags.batch_size = int(flags.batch_size * 0.7)
            print(f'********  GPU ran out of memory with batch size {old_bs}, '
                  f'trying again with batch size: {flags.batch_size}  ********')
    return flags.experiment_uid


if __name__ == '__main__':
    """
    Script to train classifiers on the MIMIC-database.
    """
    parser.add_argument('--modality', type=str, default='PA', choices=['PA', 'Lateral', 'text'],
                        help="modality on which to train the image classifier, chose between PA and Lateral")
    parser.add_argument('--fixed_extractor', type=bool, default=True,
                        help="if the layers of the encoder part of the pretrained classifier are frozen.")
    parser.add_argument('--normalization', type=bool, default=False,
                        help="if true, the images are normalized using the training set statistics.")
    parser.add_argument('--clf_loss', type=str, default='dice',
                        help="Which loss to use for the classifier training.")

    FLAGS = parser.parse_args()
    config_path = get_config_path(FLAGS)

    FLAGS = update_flags_with_config(config_path)

    run_training_procedure_clf(FLAGS)
