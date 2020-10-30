import warnings

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import gc
import json
import os

import torch

from mimic.utils.experiment import MimicExperiment
from mimic.utils.flags import parser
from mimic.run_epochs import run_epochs
from mimic.utils.filehandling import create_dir_structure, expand_paths, create_dir_structure_testing, get_config_path, \
    get_method
from timeit import default_timer as timer
from mimic.dataio.MimicDataset import Mimic
from mimic.utils.experiment import MimicExperiment


def test_clfs(flags, img_size: int, text_encoding: str, alphabet=''):
    flags.img_size = img_size
    flags.text_encoding = text_encoding
    mimic_experiment = MimicExperiment(flags=flags, alphabet=alphabet)

    mimic_test = Mimic(flags, mimic_experiment.labels, alphabet, split='test')
    print(flags.text_encoding)
    model_pa = mimic_experiment.clfs['PA']
    model_lat = mimic_experiment.clfs['Lateral']
    model_text = mimic_experiment.clfs['text']

    model_pa.eval()
    model_lat.eval()
    model_text.eval()
    dataloader = torch.utils.data.DataLoader(mimic_test, batch_size=flags.batch_size, shuffle=True, num_workers=0,
                                             drop_last=True)
    list_precision_pa = []
    list_precision_lat = []
    list_precision_text = []
    for idx, batch in enumerate(dataloader):
        batch_d = batch[0]
        batch_l = batch[1]
        labels = np.array(np.reshape(batch_l, (batch_l.shape[0], len(mimic_experiment.labels))))

        imgs_pa = Variable(batch_d['PA']).to(flags.device)
        imgs_lat = Variable(batch_d['Lateral']).to(flags.device)
        txts = Variable(batch_d['text']).to(flags.device)

        attr_hat_pa = model_pa(imgs_pa)
        attr_hat_lat = model_pa(imgs_lat)
        attr_hat_text = model_text(txts)

        avg_precision_pa = average_precision_score(labels.ravel(), attr_hat_pa.cpu().data.numpy().ravel())
        avg_precision_lat = average_precision_score(labels.ravel(), attr_hat_lat.cpu().data.numpy().ravel())
        avg_precision_text = average_precision_score(labels.ravel(), attr_hat_text.cpu().data.numpy().ravel())

        if not np.isnan(avg_precision_lat):
            list_precision_lat.append(avg_precision_lat)
        else:
            warnings.warn(
                f'avg_precision_lat has value {avg_precision_lat} with labels: {labels.ravel()} and '
                f'prediction: {attr_hat_lat.cpu().data.numpy().ravel()}')
        if not np.isnan(avg_precision_pa):
            list_precision_pa.append(avg_precision_pa)
        else:
            warnings.warn(
                f'avg_precision_pa has value {avg_precision_pa} with labels: {labels.ravel()} and '
                f'prediction: {attr_hat_pa.cpu().data.numpy().ravel()}')
        if not np.isnan(avg_precision_text):
            list_precision_text.append(avg_precision_text)
        else:
            warnings.warn(
                f'avg_precision_text has value {avg_precision_text} with labels: {labels.ravel()} and '
                f'prediction: {attr_hat_text.cpu().data.numpy().ravel()}')
    return list_precision_pa, list_precision_lat, list_precision_text


def test_clf_pa(flags, mimic_experiment, mimic_test, alphabet=''):


    print(flags.text_encoding)
    model_pa = mimic_experiment.clfs['PA']

    model_pa.eval()

    dataloader = torch.utils.data.DataLoader(mimic_test, batch_size=flags.batch_size, shuffle=True, num_workers=0,
                                             drop_last=True)
    list_precision_pa = []

    for idx, batch in enumerate(dataloader):
        batch_d = batch[0]
        batch_l = batch[1]
        labels = np.array(np.reshape(batch_l, (batch_l.shape[0], len(mimic_experiment.labels))))

        imgs_pa = Variable(batch_d['PA']).to(flags.device)

        attr_hat_pa = model_pa(imgs_pa)

        avg_precision_pa = average_precision_score(labels.ravel(), attr_hat_pa.cpu().data.numpy().ravel())

        if not np.isnan(avg_precision_pa):
            list_precision_pa.append(avg_precision_pa)
        else:
            warnings.warn(
                f'avg_precision_pa has value {avg_precision_pa} with labels: {labels.ravel()} and '
                f'prediction: {attr_hat_pa.cpu().data.numpy().ravel()}')

    return list_precision_pa


def test_clf_lat(flags, mimic_experiment, mimic_test, alphabet=''):


    print(flags.text_encoding)
    model_lat = mimic_experiment.clfs['Lateral']

    model_lat.eval()
    dataloader = torch.utils.data.DataLoader(mimic_test, batch_size=flags.batch_size, shuffle=True, num_workers=0,
                                             drop_last=True)
    list_precision_lat = []
    for idx, batch in enumerate(dataloader):
        batch_d = batch[0]
        batch_l = batch[1]
        labels = np.array(np.reshape(batch_l, (batch_l.shape[0], len(mimic_experiment.labels))))

        imgs_lat = Variable(batch_d['Lateral']).to(flags.device)

        attr_hat_lat = model_lat(imgs_lat)

        avg_precision_lat = average_precision_score(labels.ravel(), attr_hat_lat.cpu().data.numpy().ravel())

        if not np.isnan(avg_precision_lat):
            list_precision_lat.append(avg_precision_lat)
        else:
            warnings.warn(
                f'avg_precision_lat has value {avg_precision_lat} with labels: {labels.ravel()} and '
                f'prediction: {attr_hat_lat.cpu().data.numpy().ravel()}')
    return list_precision_lat


def test_clf_text(flags, mimic_experiment, mimic_test, alphabet=''):


    print(flags.text_encoding)
    model_text = mimic_experiment.clfs['text']

    model_text.eval()
    dataloader = torch.utils.data.DataLoader(mimic_test, batch_size=flags.batch_size, shuffle=True, num_workers=0,
                                             drop_last=True)
    list_precision_text = []
    for idx, batch in enumerate(dataloader):
        batch_d = batch[0]
        batch_l = batch[1]
        labels = np.array(np.reshape(batch_l, (batch_l.shape[0], len(mimic_experiment.labels))))

        txts = Variable(batch_d['text']).to(flags.device)

        attr_hat_text = model_text(txts)
        avg_precision_text = average_precision_score(labels.ravel(), attr_hat_text.cpu().data.numpy().ravel())

        if not np.isnan(avg_precision_text):
            list_precision_text.append(avg_precision_text)
        else:
            warnings.warn(
                f'avg_precision_text has value {avg_precision_text} with labels: {labels.ravel()} and '
                f'prediction: {attr_hat_text.cpu().data.numpy().ravel()}')
    return list_precision_text


if __name__ == '__main__':
    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    config_path = get_config_path()
    flags = parser.parse_args()

    with open(config_path, 'rt') as json_file:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(json_file))
        flags = parser.parse_args(namespace=t_args)
    flags = expand_paths(flags)
    flags.str_experiment = 'temp'
    flags.device = 'cuda'
    flags.dir_gen_eval_fid = ''
    flags.alpha_modalities = [flags.div_weight_uniform_content, flags.div_weight_m1_content,
                              flags.div_weight_m2_content, flags.div_weight_m3_content]
    #temp 256
    list_precision_text = test_clf_text(flags, 128, 'char', alphabet)
    list_precision_pa = test_clf_pa(flags, 256, 'char', alphabet)
    list_precision_lat = test_clf_lat(flags, 256, 'char', alphabet)

    print(list_precision_pa)
    print(list_precision_lat)
    print(list_precision_text)
    print(np.mean(list_precision_pa))
    print(np.mean(list_precision_lat))
    print(np.mean(list_precision_text))
