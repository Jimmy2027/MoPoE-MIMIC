import argparse
import json
import os
import tempfile
import unittest
from unittest import TestCase

import torch

from mimic.dataio.MimicDataset import Mimic_testing
from mimic.networks.main_train_clf_img_mimic import training_procedure_clf
from mimic.utils.filehandling import create_dir_structure, expand_paths, get_config_path
from mimic.utils.filehandling import get_str_experiments
from mimic.utils.flags import parser


class TestTraining(TestCase):
    def _run_train_loop(self, tmpdirname: str, modality: str, img_clf_type: str):
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
        FLAGS.img_size = 128
        FLAGS.batch_size = 5
        FLAGS.img_clf_type = img_clf_type
        FLAGS.dir_clf = tmpdirname
        FLAGS.dir_clf = os.path.expanduser(
            os.path.join(FLAGS.dir_clf, f'Mimic{FLAGS.img_size}_{FLAGS.img_clf_type}_new'))
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

        d_train = Mimic_testing(FLAGS)
        d_eval = Mimic_testing(FLAGS)
        training_procedure_clf(FLAGS, d_train, d_eval, modality, total_epochs=2)
        return True

    def test_clf_train_loop_resnet(self):
        with tempfile.TemporaryDirectory() as tmpdirname1:
            _ = self._run_train_loop(tmpdirname1, modality='PA', img_clf_type='resnet')

    def test_clf_train_loop_cheXnet(self):
        with tempfile.TemporaryDirectory() as tmpdirname2:
            _ = self._run_train_loop(tmpdirname2, modality='PA', img_clf_type='cheXnet')


if __name__ == '__main__':
    unittest.main()
