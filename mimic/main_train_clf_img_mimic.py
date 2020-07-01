
import os
import json

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from mimic.networks.ConvNetworkImgClf import ClfImg
from mimic import MimicDataset

from flags.flags_celeba import parser
from utils.filehandling import create_dir_structure
from utils.utils import printProgressBar
from utils.loss import clf_loss


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
    model.train()
    num_batches_train = np.floor(num_samples_train/flags.batch_size);
    step = epoch*num_batches_train;
    step_print_progress = 0;
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size, shuffle=True, num_workers=8, drop_last=True)
    for idx, (imgs_pa, imgs_lat, txts, labels) in enumerate(dataloader):
        imgs_pa = Variable(imgs_pa).to(flags.device);
        imgs_lat = Variable(imgs_lat).to(flags.device);
        labels = Variable(labels).to(flags.device);

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
    model_lat = models['lat'];

    num_samples_test = dataset.__len__()
    name_logs = "eval_clf_img"
    model.eval()
    step = epoch*np.floor(num_samples_test/flags.batch_size);
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size, shuffle=True, num_workers=0, drop_last=True)
    for idx, (imgs_pa, imgs_lat, txts, labels) in enumerate(dataloader):
        imgs_pa = Variable(imgs_pa).to(flags.device);
        imgs_lat = Variable(imgs_lat).to(flags.device);
        labels = Variable(labels).to(flags.device);

        attr_hat_pa = model(imgs_pa);
        loss_pa = clf_loss(attr_hat_pa, labels)
        attr_hat_lat = model(imgs_lat);
        loss_lat = clf_loss(attr_hat_lat, labels)

        log_writer.add_scalars('%s/Loss' % name_logs, {'PA': loss_pa.item()}, step)
        log_writer.add_scalars('%s/Loss' % name_logs, {'LATERAL': loss_lat.item()}, step)

        step += 1;

def training_procedure_clf(FLAGS):
    logger = SummaryWriter(FLAGS.dir_logs_clf)
    if FLAGS.cuda:
        model.cuda();

    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json');
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    FLAGS.num_features = len(alphabet)
    mimic_train = Mimic(FLAGS, alphabet, dataset=1)
    mimic_eval = Mimic(FLAGS, alphabet, dataset=2)
    print(mimic_eval.__len__())
    use_cuda = torch.cuda.is_available();
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu');

    model_pa = ClfImg(FLAGS).to(FLAGS.device);
    model_lat = ClfImg(FLAGS).to(FLAGS.device);
    models = {'pa': model_pa, 'lateral': model_lat};
    
    for epoch in range(0, 100):
        print('epoch: ' + str(epoch))
        models = train_clf(FLAGS, epoch, models, mimic_train, logger);
        test_clf(FLAGS, epoch, models, mimic_eval, logger);
        torch.save(model['pa'].state_dict(), os.path.join(FLAGS.dir_clf, FLAGS.clf_save_m1))
        torch.save(model['lateral'].state_dict(), os.path.join(FLAGS.dir_clf, FLAGS.clf_save_m2))


if __name__ == '__main__':

    FLAGS = parser.parse_args()
    FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content,
            FLAGS.div_weight_m1_content,
            FLAGS.div_weight_m2_content,
            FLAGS.div_weight_m3_content];
    create_dir_structure(FLAGS, train=False);
    training_procedure_clf(FLAGS)
