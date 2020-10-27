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
from mimic.utils.flags import parser
from mimic.networks.ConvNetworkTextClf import ClfText
from mimic.utils.filehandling import create_dir_structure, expand_paths, get_config_path
from mimic.utils.loss import clf_loss
from mimic.utils.utils import printProgressBar

LABELS = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']


def train_clf(flags, epoch, model, dataset, log_writer):
    # optimizer definition
    optimizer = optim.Adam(
        list(model.parameters()),
        lr=flags.initial_learning_rate,
        betas=(flags.beta_1, flags.beta_2))

    num_samples_train = dataset.__len__()
    name_logs = "train_clf_text"
    model.train()
    num_batches_train = np.floor(num_samples_train / flags.batch_size);
    step = epoch * num_batches_train;
    step_print_progress = 0;
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size, shuffle=True,
                                             num_workers=flags.dataloader_workers,
                                             drop_last=True)
    for idx, batch in enumerate(dataloader):
        batch_d = batch[0]
        batch_l = batch[1]
        txts = Variable(batch_d['text']).to(flags.device);
        labels = Variable(batch_l).to(flags.device);

        attr_hat = model(txts);
        loss = clf_loss(attr_hat, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log_writer.add_scalars('%s/Loss' % name_logs, {'loss': loss.item()}, step)

        step += 1;
        step_print_progress += 1;
        printProgressBar(step_print_progress, num_batches_train)
    return model;


def test_clf(flags, epoch, model, dataset, log_writer):
    num_samples_test = dataset.__len__()
    name_logs = "eval_clf_text"
    model.eval()
    step = epoch * np.floor(num_samples_test / flags.batch_size);
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size, shuffle=True,
                                             num_workers=flags.dataloader_workers,
                                             drop_last=True)
    for idx, batch in enumerate(dataloader):
        batch_d = batch[0]
        batch_l = batch[1]
        txts = Variable(batch_d['text']).to(flags.device);
        labels = Variable(batch_l).to(flags.device);

        attr_hat = model(txts);
        loss = clf_loss(attr_hat, labels)
        log_writer.add_scalars('%s/Loss' % name_logs, {'loss': loss.item()}, step)
        step += 1;


def training_procedure_clf(FLAGS):
    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json');
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    FLAGS.num_features = len(alphabet)
    mimic_train = Mimic(FLAGS, LABELS, alphabet, split='train')
    mimic_eval = Mimic(FLAGS, LABELS, alphabet, split='eval')
    print(mimic_eval.__len__())
    use_cuda = torch.cuda.is_available();
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu');

    logger = SummaryWriter(FLAGS.dir_logs_clf)
    model = ClfText(FLAGS, LABELS).to(FLAGS.device);
    for epoch in tqdm(range(0, 100), postfix='train_clf_text'):
        print('epoch: ' + str(epoch))
        model = train_clf(FLAGS, epoch, model, mimic_train, logger)
        test_clf(FLAGS, epoch, model, mimic_eval, logger)
        print('saving text classifier to {}'.format(
            os.path.join(FLAGS.dir_clf, f'clf_text_{FLAGS.text_encoding}_encoding')))
        torch.save(model.state_dict(), os.path.join(FLAGS.dir_clf, f'clf_text_{FLAGS.text_encoding}_encoding'))


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
