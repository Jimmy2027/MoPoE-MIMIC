import argparse
import json
import os
import tempfile
import unittest
from unittest import TestCase

import torch

from mimic.run_epochs import run_epochs
from mimic.utils.experiment import MimicExperiment
from mimic.utils.filehandling import create_dir_structure, expand_paths, create_dir_structure_testing, get_config_path, \
    get_method
from mimic.utils.flags import parser


class TestTraining(TestCase):
    def _run_train_loop(self, text_encoding: str, img_size: int, tmpdirname: str):
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
        FLAGS.dataset = 'testing'
        FLAGS.use_clf = True
        FLAGS.calc_nll = True
        FLAGS.eval_lr = True
        FLAGS.calc_prd = True
        FLAGS.save_figure = False
        FLAGS.end_epoch = 2
        FLAGS.batch_size = 100
        FLAGS.eval_freq = 1
        FLAGS.vocab_size = 3517
        FLAGS.text_encoding = text_encoding
        FLAGS.img_size = img_size
        FLAGS.steps_per_training_epoch = 10
        print(
            f'running on {FLAGS.device} with text {FLAGS.text_encoding} encoding with method {FLAGS.method} and img size {FLAGS.img_size}')
        FLAGS = create_dir_structure(FLAGS)
        import mimic
        alphabet_path = os.path.join(os.path.dirname(mimic.__file__), 'alphabet.json')
        with open(alphabet_path) as alphabet_file:
            alphabet = str(''.join(json.load(alphabet_file)))

        mimic = MimicExperiment(FLAGS, alphabet)
        create_dir_structure_testing(mimic)
        mimic.set_optimizer()

        run_epochs(mimic)

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


if __name__ == '__main__':
    unittest.main()
