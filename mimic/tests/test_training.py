import argparse
import json
import os
import tempfile
import unittest
from unittest import TestCase

import pytest
import torch
import torch.multiprocessing as mp

from mimic.run_epochs import run_epochs
from mimic.utils.experiment import MimicExperiment
from mimic.utils.filehandling import create_dir_structure, expand_paths, create_dir_structure_testing, get_config_path, \
    get_method
from mimic.utils.flags import parser


class TestTraining(TestCase):
    def _setup_distributed(self, flags):
        flags.world_size = torch.cuda.device_count()
        flags.distributed = flags.world_size > 1
        flags.batch_size = int(flags.batch_size / flags.world_size)

    def _run_train_loop(self, text_encoding: str, img_size: int, tmpdirname: str,
                        feature_extractor_img: str = 'resnet', distributed: bool = False):
        """
        General test to see if training loop works
        """
        FLAGS = parser.parse_args([])
        config_path = get_config_path()
        with open(config_path, 'rt') as json_file:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(json_file))
            FLAGS = parser.parse_args([], namespace=t_args)
        FLAGS = expand_paths(t_args)
        use_cuda = torch.cuda.is_available()
        FLAGS.device = torch.device('cuda' if use_cuda else 'cpu')
        FLAGS = get_method(FLAGS)

        FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content, FLAGS.div_weight_m1_content,
                                  FLAGS.div_weight_m2_content, FLAGS.div_weight_m3_content]
        if os.path.exists('/cluster/home/klugh/'):
            FLAGS.dir_experiment = os.path.join(os.path.expandvars('$TMPDIR'), 'experiment_dir')
        else:
            FLAGS.dir_experiment = tmpdirname
        FLAGS.distributed = distributed
        FLAGS.dataset = 'testing'
        FLAGS.use_clf = True
        FLAGS.calc_nll = False
        FLAGS.eval_lr = True
        FLAGS.calc_prd = True
        FLAGS.save_figure = False
        FLAGS.end_epoch = 2
        FLAGS.batch_size = 10
        FLAGS.eval_freq = 1
        FLAGS.vocab_size = 3517
        FLAGS.feature_extractor_img = feature_extractor_img
        FLAGS.text_encoding = text_encoding
        FLAGS.img_size = img_size
        FLAGS.steps_per_training_epoch = 2
        print(
            f'running on {FLAGS.device} with text {FLAGS.text_encoding} encoding with method {FLAGS.method} '
            f'and img size {FLAGS.img_size}')
        FLAGS = create_dir_structure(FLAGS)
        import mimic
        alphabet_path = os.path.join(os.path.dirname(mimic.__file__), 'alphabet.json')
        with open(alphabet_path) as alphabet_file:
            alphabet = str(''.join(json.load(alphabet_file)))
        FLAGS.alphabet = alphabet
        mimic = MimicExperiment(FLAGS)
        create_dir_structure_testing(mimic)
        mimic.set_optimizer()

        if FLAGS.distributed:
            self._setup_distributed(FLAGS)
            mp.spawn(run_epochs, nprocs=FLAGS.world_size, args=(mimic,), join=True)
        else:
            mimic.flags.world_size = 1
            run_epochs(FLAGS.device, mimic)

        return True

    def test_train_loop_charEncoding_128(self):
        with tempfile.TemporaryDirectory() as tmpdirname1:
            _ = self._run_train_loop('char', 128, tmpdirname1)

    def test_train_loop_charEncoding_256(self):
        with tempfile.TemporaryDirectory() as tmpdirname2:
            _ = self._run_train_loop('char', 256, tmpdirname2)

    def test_train_loop_wordEncoding_128(self):
        with tempfile.TemporaryDirectory() as tmpdirname3:
            _ = self._run_train_loop('word', 128, tmpdirname3)

    def test_train_loop_wordEncoding_256(self):
        with tempfile.TemporaryDirectory() as tmpdirname2:
            _ = self._run_train_loop('word', 256, tmpdirname2)

    def test_train_loop_charEncoding_256_densenet(self):
        with tempfile.TemporaryDirectory() as tmpdirname1:
            _ = self._run_train_loop('char', 256, tmpdirname1, 'densenet')

    @pytest.mark.distributed
    def test_train_loop_charEncoding_128_distributed(self):
        with tempfile.TemporaryDirectory() as tmpdirname1:
            _ = self._run_train_loop('char', 128, tmpdirname1, distributed=True)


if __name__ == '__main__':
    unittest.main()
