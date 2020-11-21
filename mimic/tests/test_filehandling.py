import argparse
import os
import tempfile
from unittest import TestCase
import unittest
from mimic.utils.filehandling import create_dir_structure
from dataclasses import dataclass
from unittest import TestCase

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from mimic.networks.classifiers.utils import ExperimentDf, get_models, Callbacks, CallbacksProto


@dataclass
class Args:
    dir_fid = None
    dataset = 'mimic'
    config_path = None

    def __init__(self, tmpdirname: str):
        self.dir_experiment = tmpdirname
        self.dir_clf = os.path.join(tmpdirname, 'clf_dir')
        self.dir_logs_clf = tmpdirname


class TestFilehandling(TestCase):
    def test_create_dir_structure(self):
        """
        Checks if function create_dir_structure is running.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            args = Args(tmpdirname)
            _ = create_dir_structure(args)


if __name__ == '__main__':
    unittest.main()
