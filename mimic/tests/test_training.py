import argparse
import json
import os
import tempfile
import unittest
from unittest import TestCase
import torch

from mimic.run_epochs import run_epochs
from mimic.utils.experiment import MimicExperiment
from mimic.utils.filehandling import create_dir_structure, expand_paths, create_dir_structure_testing, get_config_path
from mimic.utils.flags import parser


class TestTraining(TestCase):
    def _run_train_loop(self, text_encoding):
        """
        General test to see if training loop works
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            FLAGS = parser.parse_args([])
            config_path = get_config_path()
            with open(config_path, 'rt') as json_file:
                t_args = argparse.Namespace()
                t_args.__dict__.update(json.load(json_file))
                FLAGS = parser.parse_args([], namespace=t_args)
            FLAGS = expand_paths(t_args)
            use_cuda = torch.cuda.is_available()
            FLAGS.device = torch.device('cuda' if use_cuda else 'cpu')
            if FLAGS.method == 'poe':
                FLAGS.modality_poe = True
                FLAGS.poe_unimodal_elbos = True
            elif FLAGS.method == 'moe':
                FLAGS.modality_moe = True
            elif FLAGS.method == 'jsd':
                FLAGS.modality_jsd = True
            elif FLAGS.method == 'joint_elbo':
                FLAGS.joint_elbo = True
            else:
                NotImplementedError('method not implemented... exit!')

            FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content, FLAGS.div_weight_m1_content,
                                      FLAGS.div_weight_m2_content, FLAGS.div_weight_m3_content]

            FLAGS.dir_experiment = tmpdirname
            FLAGS.dataset = 'testing'
            FLAGS.use_clf = False
            FLAGS.end_epoch = 2
            FLAGS.batch_size = 10
            FLAGS.eval_freq = 1
            FLAGS.text_encoding = text_encoding

            FLAGS = create_dir_structure(FLAGS)
            import mimic
            alphabet_path = os.path.join(os.path.dirname(mimic.__file__), 'alphabet.json')
            with open(alphabet_path) as alphabet_file:
                alphabet = str(''.join(json.load(alphabet_file)))

            mimic = MimicExperiment(FLAGS, alphabet)
            create_dir_structure_testing(mimic)
            mimic.set_optimizer()

            run_epochs(mimic)

    def test_train_loop(self):
        text_encodings = ['char', 'word']
        for text_encoding in text_encodings:
            self._run_train_loop(text_encoding)


if __name__ == '__main__':
    unittest.main()
