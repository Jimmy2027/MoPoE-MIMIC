import argparse
import json
import os
from tqdm import tqdm
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
import shutil

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
    epoch = 0
    logger = SummaryWriter(FLAGS.dir_logs_clf)

    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json');
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    FLAGS.num_features = len(alphabet)
    mimic_train = Mimic(FLAGS, LABELS, alphabet, split='train')
    mimic_eval = Mimic(FLAGS, LABELS, alphabet, split='eval')
    print(mimic_eval.__len__())
    use_cuda = torch.cuda.is_available();
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu');

    model_pa = ClfImg(FLAGS, LABELS).to(FLAGS.device);
    model_lat = ClfImg(FLAGS, LABELS).to(FLAGS.device);
    models = {'pa': model_pa, 'lateral': model_lat};
    for epoch in tqdm(range(0, 100), postfix='train_clf_img'):
        models = train_clf(FLAGS, epoch, models, mimic_train, logger);
        test_clf(FLAGS, epoch, models, mimic_eval, logger)
    save_and_overwrite_model(models['pa'].state_dict(), FLAGS.dir_clf, FLAGS.clf_save_m1, epoch)
    save_and_overwrite_model(models['lateral'].state_dict(), FLAGS.dir_clf, FLAGS.clf_save_m1, epoch)


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

    config_path = get_config_path()
    with open(config_path, 'rt') as json_file:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(json_file))
        FLAGS = parser.parse_args(namespace=t_args)
    FLAGS.img_size = 128
    FLAGS.dir_clf = os.path.join(FLAGS.dir_clf, f'Mimic{FLAGS.img_size}')
    FLAGS = expand_paths(FLAGS)
    print(f'Training image classifier for images of size {FLAGS.img_size}')
    print(os.path.join(FLAGS.dir_clf, FLAGS.clf_save_m1))
    use_cuda = torch.cuda.is_available()
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu')
    FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content,
                              FLAGS.div_weight_m1_content,
                              FLAGS.div_weight_m2_content,
                              FLAGS.div_weight_m3_content];
    create_dir_structure(FLAGS, train=False);
    # This will overwrite old classifiers!!
    training_procedure_clf(FLAGS)
