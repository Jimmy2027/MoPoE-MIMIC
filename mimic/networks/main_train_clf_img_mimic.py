import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mimic.dataio.MimicDataset import Mimic
from mimic.utils.flags import parser
from mimic.networks.ConvNetworkImgClf import ClfImg
from mimic.utils.filehandling import create_dir_structure, get_config_path, expand_paths
from mimic.utils.loss import clf_loss
from mimic.utils.utils import printProgressBar

LABELS = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']


def train_clf(flags, epoch, models, dataset, log_writer):
    model_pa = models['pa'];
    model_lat = models['lateral'];
    # optimizer definition
    optimizer_pa = optim.Adam(
        list(model_pa.parameters()),
        lr=flags.initial_learning_rate,
        betas=(flags.beta_1, flags.beta_2))
    optimizer_lat = optim.Adam(
        list(model_lat.parameters()),
        lr=flags.initial_learning_rate,
        betas=(flags.beta_1, flags.beta_2))

    num_samples_train = dataset.__len__()
    name_logs = "train_clf_img"
    model_pa.train()
    model_lat.train()
    num_batches_train = np.floor(num_samples_train / flags.batch_size);
    step = epoch * num_batches_train;
    step_print_progress = 0;
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size, shuffle=True,
                                             num_workers=flags.dataloader_workers,
                                             drop_last=True)
    for idx, batch in enumerate(dataloader):
        batch_d = batch[0]
        batch_l = batch[1]
        imgs_pa = Variable(batch_d['PA']).to(flags.device);
        imgs_lat = Variable(batch_d['Lateral']).to(flags.device);
        labels = Variable(batch_l).to(flags.device);

        attr_hat_pa = model_pa(imgs_pa);
        loss_pa = clf_loss(attr_hat_pa, labels);
        optimizer_pa.zero_grad()
        loss_pa.backward()
        optimizer_pa.step()

        attr_hat_lat = model_lat(imgs_lat);
        loss_lat = clf_loss(attr_hat_lat, labels);
        optimizer_lat.zero_grad()
        loss_lat.backward()
        optimizer_lat.step()

        log_writer.add_scalars('%s/Loss PA' % name_logs, {'pa': loss_pa.item()}, step)
        log_writer.add_scalars('%s/Loss Lateral' % name_logs, {'lateral': loss_lat.item()}, step)

        step += 1;
        step_print_progress += 1;
        printProgressBar(step_print_progress, num_batches_train)
    models = {'pa': model_pa, 'lateral': model_lat};
    return models;


def test_clf(flags, epoch, models, dataset, log_writer):
    model_pa = models['pa'];
    model_lat = models['lateral'];

    num_samples_test = dataset.__len__()
    name_logs = "eval_clf_img"
    model_pa.eval()
    model_lat.eval()
    step = epoch * np.floor(num_samples_test / flags.batch_size);
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size, shuffle=True, num_workers=0,
                                             drop_last=True)
    for idx, batch in enumerate(dataloader):
        batch_d = batch[0]
        batch_l = batch[1]
        imgs_pa = Variable(batch_d['PA']).to(flags.device);
        imgs_lat = Variable(batch_d['Lateral']).to(flags.device);
        labels = Variable(batch_l).to(flags.device);

        attr_hat_pa = model_pa(imgs_pa);
        loss_pa = clf_loss(attr_hat_pa, labels)
        attr_hat_lat = model_pa(imgs_lat);
        loss_lat = clf_loss(attr_hat_lat, labels)

        log_writer.add_scalars('%s/Loss' % name_logs, {'PA': loss_pa.item()}, step)
        log_writer.add_scalars('%s/Loss' % name_logs, {'LATERAL': loss_lat.item()}, step)

        step += 1;


def training_procedure_clf(FLAGS):
    logger = SummaryWriter(FLAGS.dir_logs_clf)

    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json');
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    FLAGS.num_features = len(alphabet)
    mimic_train = Mimic(FLAGS, LABELS, alphabet, dataset=1)
    mimic_eval = Mimic(FLAGS, LABELS, alphabet, dataset=2)
    print(mimic_eval.__len__())
    use_cuda = torch.cuda.is_available();
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu');

    model_pa = ClfImg(FLAGS, LABELS).to(FLAGS.device);
    model_lat = ClfImg(FLAGS, LABELS).to(FLAGS.device);
    models = {'pa': model_pa, 'lateral': model_lat};

    for epoch in range(0, 100):
        print('epoch: ' + str(epoch))
        models = train_clf(FLAGS, epoch, models, mimic_train, logger);
        test_clf(FLAGS, epoch, models, mimic_eval, logger);
        torch.save(models['pa'].state_dict(), os.path.join(FLAGS.dir_clf, FLAGS.clf_save_m1))
        torch.save(models['lateral'].state_dict(), os.path.join(FLAGS.dir_clf, FLAGS.clf_save_m2))


if __name__ == '__main__':
    FLAGS = parser.parse_args()

    config_path = get_config_path()
    with open(config_path, 'rt') as json_file:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(json_file))
        FLAGS = parser.parse_args(namespace=t_args)
    FLAGS = expand_paths(FLAGS)

    use_cuda = torch.cuda.is_available()
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu')
    FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content,
                              FLAGS.div_weight_m1_content,
                              FLAGS.div_weight_m2_content,
                              FLAGS.div_weight_m3_content];
    create_dir_structure(FLAGS, train=False);
    training_procedure_clf(FLAGS)
