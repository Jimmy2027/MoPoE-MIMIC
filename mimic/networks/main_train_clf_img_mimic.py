import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from mimic.dataio.MimicDataset import Mimic
from mimic.networks.CheXNet import CheXNet
from mimic.networks.ConvNetworkImgClf import ClfImg
from mimic.utils.filehandling import create_dir_structure, get_config_path, expand_paths
from mimic.utils.filehandling import get_str_experiments
from mimic.utils.flags import parser
from mimic.utils.loss import clf_loss

LABELS = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']


def train_clf(flags, epoch, model, dataset, log_writer, modality) -> torch.nn.Module():
    # optimizer definition
    optimizer = optim.Adam(
        list(model.parameters()),
        lr=flags.initial_learning_rate,
        betas=(flags.beta_1, flags.beta_2))

    num_samples_train = dataset.__len__()
    name_logs = "train_clf_img"
    model.train()

    num_batches_train = np.floor(num_samples_train / flags.batch_size)
    step = epoch * num_batches_train
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size, shuffle=True,
                                             num_workers=flags.dataloader_workers,
                                             drop_last=True)
    for idx, (batch_d, batch_l) in tqdm(enumerate(dataloader), total=num_batches_train, postfix='steps epoch'):
        if flags.img_clf_type == 'cheXnet':
            bs, n_crops, c, h, w = batch_d[modality].size()
            imgs = torch.autograd.Variable(batch_d[modality].view(-1, c, h, w).cuda(), volatile=True)
        else:
            imgs = Variable(batch_d[modality]).to(flags.device)
        labels = Variable(batch_l).to(flags.device)

        attr_hat = model(imgs)
        if flags.img_clf_type == 'cheXnet':
            attr_hat = attr_hat.view(bs, n_crops, -1).mean(1)
        loss = clf_loss(attr_hat, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_writer.add_scalars(f'%s/Loss {modality}' % name_logs, {modality: loss.item()}, step)

        step += 1;
    return model;


def eval_clf(flags, epoch, model, dataset, log_writer, modality: str):
    num_samples_test = dataset.__len__()
    name_logs = "eval_clf_img"
    model.eval()
    step = epoch * np.floor(num_samples_test / flags.batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size, shuffle=True, num_workers=0,
                                             drop_last=True)
    for idx, (batch_d, batch_l) in enumerate(dataloader):
        if flags.img_clf_type == 'cheXnet':
            bs, n_crops, c, h, w = batch_d[modality].size()
            imgs = torch.autograd.Variable(batch_d[modality].view(-1, c, h, w).cuda(), volatile=True)
        else:
            imgs = Variable(batch_d[modality]).to(flags.device)
        labels = Variable(batch_l).to(flags.device)

        attr_hat = model(imgs)
        if flags.img_clf_type == 'cheXnet':
            attr_hat = attr_hat.view(bs, n_crops, -1).mean(1)
        loss = clf_loss(attr_hat, labels)

        log_writer.add_scalars('%s/Loss' % name_logs, {modality: loss.item()}, step)

        step += 1


def get_models(flags):
    if flags.img_clf_type == 'cheXnet':
        model = CheXNet(len(LABELS)).cuda()
        # todo pytorch recommends to use DistributedDataParallel instead of DataParallel
        model = torch.nn.DataParallel(model).cuda()
    elif flags.img_clf_type == 'resnet':
        model = ClfImg(flags, LABELS).to(flags.device)
    else:
        raise NotImplementedError(f'{flags.img_clf_type} is not implemented, chose between "cheXnet" and "resnet"')
    return model


def training_procedure_clf(flags, train_set, eval_set, modality: str, total_epochs: int = 100):
    epoch = 0
    logger = SummaryWriter(flags.dir_logs_clf)

    print(eval_set.__len__())
    use_cuda = torch.cuda.is_available()
    flags.device = torch.device('cuda' if use_cuda else 'cpu')
    model = get_models(flags)

    for epoch in tqdm(range(0, total_epochs), postfix='train_clf_img'):
        print(f'epoch: {epoch}')
        model = train_clf(flags, epoch, model, train_set, logger, modality)
        eval_clf(flags, epoch, model, eval_set, logger, modality)
        save_and_overwrite_model(model.state_dict(), flags.dir_clf, flags.clf_save_m1, epoch)

    torch.save(flags, os.path.join(flags.dir_clf, f'flags_clf_img_{epoch}.rar'))


def save_and_overwrite_model(state_dict, dir_path: str, checkpoint_name: str, epoch: int):
    """
    saves the model and deletes old one
    """
    for file in os.listdir(dir_path):
        if file.startswith(checkpoint_name):
            print(f'deleting old checkpoint: {os.path.join(dir_path, file)}')
            os.remove(os.path.join(dir_path, file))
    print('saving model to {}'.format(os.path.join(dir_path, checkpoint_name + f'_{epoch}')))
    torch.save(state_dict, os.path.join(dir_path, checkpoint_name + f'_{epoch}'))


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    MODALITY = 'PA'
    config_path = get_config_path()
    with open(config_path, 'rt') as json_file:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(json_file))
        FLAGS = parser.parse_args(namespace=t_args)
    FLAGS.img_size = 256
    FLAGS.batch_size = 150
    FLAGS.img_clf_type = 'cheXnet'
    FLAGS.dir_clf = os.path.expanduser(os.path.join(FLAGS.dir_clf, f'Mimic{FLAGS.img_size}_{FLAGS.img_clf_type}_new'))
    FLAGS = expand_paths(FLAGS)
    print(f'Training image classifier {FLAGS.img_clf_type} for images of size {FLAGS.img_size}')
    print(os.path.join(FLAGS.dir_clf, FLAGS.clf_save_m1))
    use_cuda = torch.cuda.is_available()
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu')
    FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content,
                              FLAGS.div_weight_m1_content,
                              FLAGS.div_weight_m2_content,
                              FLAGS.div_weight_m3_content]
    create_dir_structure(FLAGS, train=False)
    FLAGS.dir_logs_clf = os.path.join(os.path.expanduser(FLAGS.dir_clf),
                                      get_str_experiments(FLAGS, prefix='clf_img'))
    # This will overwrite old classifiers!!
    mimic_train = Mimic(FLAGS, LABELS, split='train')
    mimic_eval = Mimic(FLAGS, LABELS, split='eval')
    training_procedure_clf(FLAGS, mimic_train, mimic_eval, MODALITY)
