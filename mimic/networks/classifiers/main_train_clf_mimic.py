import os

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from mimic.dataio.MimicDataset import Mimic
from mimic.networks.classifiers.utils import set_clf_paths, ExperimentDf, get_models, LABELS, Callbacks
from mimic.utils.flags import parser
from mimic.utils.flags import update_flags_with_config
from mimic.utils.loss import clf_loss

START_EARLY_STOPPING_EPOCH = 2
MAX_EARLY_STOPPING_IDX = 5


def train_clf(flags, epoch, model, dataset: Mimic, log_writer, modality, optimizer) -> torch.nn.Module():
    num_samples_train = dataset.__len__()
    name_logs = f"train_clf_{modality}"
    model.train()

    num_batches_train = np.floor(num_samples_train / flags.batch_size)
    step = epoch * num_batches_train
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size, shuffle=True,
                                             num_workers=flags.dataloader_workers,
                                             drop_last=True)
    for idx, (batch_d, batch_l) in tqdm(enumerate(dataloader), total=num_batches_train, postfix='steps epoch'):
        if flags.img_clf_type == 'cheXnet' and not modality == 'text':
            bs, n_crops, c, h, w = batch_d[modality].size()
            imgs = torch.autograd.Variable(batch_d[modality].view(-1, c, h, w).cuda())
        else:
            imgs = Variable(batch_d[modality]).to(flags.device)
        labels = Variable(batch_l).to(flags.device)
        optimizer.zero_grad()
        attr_hat = model(imgs)

        if flags.img_clf_type == 'cheXnet' and not modality == 'text':
            attr_hat = attr_hat.view(bs, n_crops, -1).mean(1)
        loss = clf_loss(attr_hat, labels)
        loss.backward()
        optimizer.step()

        log_writer.add_scalars(f'%s/Loss {modality}' % name_logs, {modality: loss.item()}, step)

        step += 1;
    return model;


def eval_clf(flags, epoch, model, dataset, log_writer, modality: str):
    num_samples_test = dataset.__len__()
    name_logs = f"eval_clf_{modality}"
    model.eval()
    step = epoch * np.floor(num_samples_test / flags.batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size, shuffle=True, num_workers=0)
    losses = []
    list_predictions = []
    list_gt = []
    with torch.no_grad():
        for idx, (batch_d, batch_l) in enumerate(dataloader):
            if flags.img_clf_type == 'cheXnet' and not modality == 'text':
                bs, n_crops, c, h, w = batch_d[modality].size()
                imgs = Variable(batch_d[modality].view(-1, c, h, w)).to(flags.device)
            else:
                imgs = Variable(batch_d[modality]).to(flags.device)
            gt = Variable(batch_l).to(flags.device)
            attr_hat = model(imgs)
            if flags.img_clf_type == 'cheXnet' and not modality == 'text':
                attr_hat = attr_hat.view(bs, n_crops, -1).mean(1)
            loss = clf_loss(attr_hat, gt)
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
    for epoch in tqdm(range(0, total_epochs), postfix=f'train_clf_{modality}'):
        print(f'epoch: {epoch}')
        model = train_clf(flags, epoch, model, train_set, logger, modality, optimizer)
        loss, mean_AP = eval_clf(flags, epoch, model, eval_set, logger, modality)
        if callbacks.update_epoch(epoch, loss, mean_AP, model):
            break

    torch.save(flags, os.path.join(flags.dir_clf, f'flags_clf_{modality}_{epoch}.rar'))
    experiment_df.write_experiment_time()


if __name__ == '__main__':
    parser.add_argument('--modality', type=str, default='PA', choices=['PA', 'Lateral', 'text'],
                        help="modality on which to train the image classifier, chose between PA and Lateral")
    FLAGS = parser.parse_args()
    MODALITY = FLAGS.modality
    FLAGS = update_flags_with_config(FLAGS)
    # temp 256
    FLAGS.img_size = 256
    # temp batch size=30
    FLAGS.batch_size = 35
    # FLAGS.initial_learning_rate = 1e-4
    FLAGS.n_crops = 10
    FLAGS.img_clf_type = 'cheXnet'
    # temp
    FLAGS.dir_clf += '_new'

    set_clf_paths(FLAGS)

    # This will overwrite old classifiers!!
    mimic_train = Mimic(FLAGS, LABELS, split='train')
    mimic_eval = Mimic(FLAGS, LABELS, split='eval')
    print(f'Training image classifier {FLAGS.img_clf_type} with modality: {FLAGS.modality} '
          f'for images of size {FLAGS.img_size}')
    print(f'clf_log_dir: {FLAGS.dir_logs_clf}')
    training_procedure_clf(FLAGS, mimic_train, mimic_eval, MODALITY, total_epochs=600)
