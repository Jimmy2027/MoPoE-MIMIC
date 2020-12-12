import argparse
import json
import os
import tempfile
import unittest
from unittest import TestCase

import torch

from mimic.dataio.MimicDataset import Mimic_testing
from mimic.networks.classifiers.main_train_clf_mimic import training_procedure_clf
from mimic.utils.filehandling import create_dir_structure, expand_paths, get_config_path
from mimic.utils.filehandling import get_str_experiments
from mimic.utils.flags import parser
import pytest


class TestTraining(TestCase):
    def _run_train_loop(self, tmpdirname: str, modality: str, img_clf_type: str, n_crops=1):
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
        FLAGS.img_size = 256
        FLAGS.batch_size = 5
        FLAGS.img_clf_type = img_clf_type
        FLAGS.dir_clf = tmpdirname
        FLAGS.vocab_size = 3517
        FLAGS.n_crops = n_crops
        FLAGS.fixed_extractor = True
        FLAGS.distributed = False
        FLAGS.clf_loss = 'bce_with_logits'
        FLAGS.experiment_uid = 'test'
        FLAGS.dir_clf = os.path.expanduser(
            os.path.join(FLAGS.dir_clf, f'Mimic{FLAGS.img_size}_{FLAGS.img_clf_type}'))
        FLAGS = expand_paths(FLAGS)
        print(f'Training image classifier {FLAGS.img_clf_type} for images of size {FLAGS.img_size}')
        use_cuda = torch.cuda.is_available()
        FLAGS.device = torch.device('cuda' if use_cuda else 'cpu')
        FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content,
                                  FLAGS.div_weight_m1_content,
                                  FLAGS.div_weight_m2_content,
                                  FLAGS.div_weight_m3_content]
        create_dir_structure(FLAGS, train=False)
        FLAGS.dir_logs_clf = os.path.join(os.path.expanduser(FLAGS.dir_clf),
                                          get_str_experiments(FLAGS, prefix='clf_img'))

        d_train = Mimic_testing(FLAGS, classifier_training=True)
        d_eval = Mimic_testing(FLAGS, classifier_training=True)

        # with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        training_procedure_clf(FLAGS, d_train, d_eval, modality, total_epochs=2)
        # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
        return True

    def test_clf_train_loop_resnet(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            _ = self._run_train_loop(tmpdirname, modality='PA', img_clf_type='resnet')

    def test_clf_train_loop_densenet(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            _ = self._run_train_loop(tmpdirname, modality='PA', img_clf_type='densenet')

    def test_clf_train_loop_text(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            _ = self._run_train_loop(tmpdirname, modality='text', img_clf_type='')

    def test_clf_train_loop_densenet_5ncrops(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            _ = self._run_train_loop(tmpdirname, modality='PA', img_clf_type='densenet', n_crops=5)


if __name__ == '__main__':
    unittest.main()
