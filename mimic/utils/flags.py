import argparse
import json

from mimic import log
from mimic.utils.BaseFlags import parser as parser
from mimic.utils.filehandling import expand_paths
import os
import numpy as np
from typing import Union


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser.add_argument('--exp_str_prefix', type=str, default='Mimic', help="prefix of the experiment directory.")
parser.add_argument('--dataset', type=str, default='Mimic', help="name of the dataset")
parser.add_argument('--config_path', type=str, default=None, help="path to the json config")
parser.add_argument('--verbose', type=int, default=0, help="global verbosity level")
parser.add_argument('--load_flags', type=str, default=None, help="overwrite all values with parameters from an old "
                                                                 "experiment. Give the path to the flags.rar "
                                                                 "file as input.")
# Image dependent
parser.add_argument('--fixed_image_extractor', type=str2bool, default=True,
                    help="If the feature extraction layers of the "
                         "pretrained densenet are frozen. "
                         "Only works when img_clf_type classifier "
                         "is densenet.")
# DATA DEPENDENT
parser.add_argument('--only_text_modality', type=str2bool, default=False,
                    help="flag to indicate if only the text modality is to be used")
parser.add_argument('--undersample_dataset', type=str2bool, default=False,
                    help="flag to indicate if the dataset should be undersampled such that there are "
                         "the same number of datapoints that have no label than datapoints that have a label")

# Text Dependent
parser.add_argument('--text_encoding', type=str, default='char',
                    help="encoding of the text, either character or wordwise")
parser.add_argument('--len_sequence', type=int, default=1024, help="length of sequence")
parser.add_argument('--word_min_occ', type=int, default=3,
                    help="min occurence of a word in the dataset such that it is added to the vocabulary.")

parser.add_argument('--style_pa_dim', type=int, default=0, help="dimension of varying factor latent space")
parser.add_argument('--style_lat_dim', type=int, default=0, help="dimension of varying factor latent space")
parser.add_argument('--style_text_dim', type=int, default=0, help="dimension of varying factor latent space")
parser.add_argument('--image_channels', type=int, default=1, help="number of classes on which the data set trained")
parser.add_argument('--img_size', type=int, default=128, help="size of the images on which the model is trained")
parser.add_argument('--DIM_img', type=int, default=128, help="number of classes on which the data set trained")
parser.add_argument('--DIM_text', type=int, default=128, help="number of classes on which the data set trained")
parser.add_argument('--likelihood_m1', type=str, default='laplace', help="output distribution")
parser.add_argument('--likelihood_m2', type=str, default='laplace', help="output distribution")
parser.add_argument('--likelihood_m3', type=str, default='categorical', help="output distribution")
parser.add_argument('--dataloader_workers', type=int, default=8, help="number of workers used for the Dataloader")
parser.add_argument('--use_toy_dataset', type=bool, default=False, help="if true uses small toy dataset")

# paths to save models
parser.add_argument('--encoder_save_m1', type=str, default='encoderM1', help="model save for encoder")
parser.add_argument('--encoder_save_m2', type=str, default='encoderM2', help="model save for encoder")
parser.add_argument('--encoder_save_m3', type=str, default='encoderM3', help="model save for decoder")
parser.add_argument('--decoder_save_m1', type=str, default='decoderM1', help="model save for decoder")
parser.add_argument('--decoder_save_m2', type=str, default='decoderM2', help="model save for decoder")
parser.add_argument('--decoder_save_m3', type=str, default='decoderM3', help="model save for decoder")

# classifiers
parser.add_argument('--text_clf_type', type=str, default='word',
                    help="text classifier type, implemented are 'word' and 'char'")
parser.add_argument('--img_clf_type', type=str, default='resnet',
                    help="image classifier type, implemented are 'resnet' and 'densenet'")
parser.add_argument('--clf_save_m1', type=str, default='clf_m1', help="model save for clf")
parser.add_argument('--clf_save_m2', type=str, default='clf_m2', help="model save for clf")
parser.add_argument('--clf_save_m3', type=str, default='clf_m3', help="model save for clf")
parser.add_argument('--clf_loss', type=str, default='binary_crossentropy',
                    choices=['binary_crossentropy', 'crossentropy', 'bce_with_logits'], help="model save for clf")

# Callbacks
parser.add_argument('--reduce_lr_on_plateau', type=bool, default=False,
                    help="boolean indicating if callback 'reduce lr on plateau' is used")
parser.add_argument('--max_early_stopping_index', type=int, default=5,
                    help="patience of the early stopper. If the target metric did not improve "
                         "for that amount of epochs, training is stopepd")
parser.add_argument('--start_early_stopping_epoch', type=int, default=0,
                    help="epoch on which to start the early stopping callback")

# LOSS TERM WEIGHTS
parser.add_argument('--beta_m1_style', type=float, default=1.0, help="default weight divergence term style modality 1")
parser.add_argument('--beta_m2_style', type=float, default=1.0, help="default weight divergence term style modality 2")
parser.add_argument('--beta_m3_style', type=float, default=1.0, help="default weight divergence term style modality 2")
parser.add_argument('--div_weight_m1_content', type=float, default=0.25,
                    help="default weight divergence term content modality 1")
parser.add_argument('--div_weight_m2_content', type=float, default=0.25,
                    help="default weight divergence term content modality 2")
parser.add_argument('--div_weight_m3_content', type=float, default=0.25,
                    help="default weight divergence term content modality 3")
parser.add_argument('--div_weight_uniform_content', type=float, default=0.25,
                    help="default weight divergence term prior")
parser.add_argument('--rec_weight_m1', default=0.33, type=float,
                    help="weight of the m1 modality for the log probs. Type should be either float or string.")
parser.add_argument('--rec_weight_m2', default=0.33, type=float,
                    help="weight of the m2 modality for the log probs. Type should be either float or string.")
parser.add_argument('--rec_weight_m3', default=0.33, type=float,
                    help="weight of the m3 modality for the log probs. Type should be either float or string.")


def update_flags_with_config(config_path: str, additional_args={}, testing=False):
    """
    If testing is true, no cli arguments will be read.
    """
    with open(config_path, 'rt') as json_file:
        t_args = argparse.Namespace()
        json_config = json.load(json_file)
    t_args.__dict__.update({**json_config, **additional_args})
    if testing:
        return parser.parse_args([], namespace=t_args)
    else:
        return parser.parse_args(namespace=t_args)


def get_freer_gpu():
    """
    Returns the index of the gpu with the most free memory.
    Taken from https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/6
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def setup_flags(flags, testing=False):
    """
    If testing is true, no cli arguments will be read.
    """
    import torch
    from pathlib import Path
    if flags.config_path:
        flags = update_flags_with_config(config_path=flags.config_path, testing=testing)
    flags = expand_paths(flags)
    use_cuda = torch.cuda.is_available()
    flags.device = torch.device('cuda' if use_cuda else 'cpu')
    if str(flags.device) == 'cuda':
        torch.cuda.set_device(get_freer_gpu())
    flags = flags_set_alpha_modalities(flags)
    flags.log_file = log.manager.root.handlers[1].baseFilename
    flags.len_sequence = 128 if flags.text_encoding == 'word' else 1024

    if flags.load_flags:
        old_flags = torch.load(Path(flags.load_flags).expanduser())
        # create param dict from all the params of old_flags that are not paths
        params = {k: v for k, v in old_flags.item() if ('dir' not in v) and ('path' not in v)}
        flags.__dict__.update(params)

    return flags


def flags_set_alpha_modalities(flags):
    flags.alpha_modalities = [flags.div_weight_uniform_content, flags.div_weight_m1_content,
                              flags.div_weight_m2_content, flags.div_weight_m3_content]
    return flags
