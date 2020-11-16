from unittest import TestCase
from mimic.networks.classifiers.utils import set_clf_paths, ExperimentDf, get_models, LABELS, Callbacks
from collections import namedtuple
import tempfile
from tensorboardX import SummaryWriter
import torch.optim as optim
from dataclasses import dataclass
import torch
import os


@dataclass
class Args:
    img_clf_type = 'resnet'
    img_size = 128
    image_channels = 10
    clf_save_m1: str = 'temp'
    dataset: str = 'testing'
    experiment_uid: str = 'temp'
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    def __init__(self, tmpdirname: str):
        self.tempdirname = tmpdirname
        self.dir_clf = tmpdirname
        self.dir_logs_clf = tmpdirname


class TestCallbacks(TestCase):
    def test_callbacks_increasing_loss_decreasingAP(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            args = Args(tmpdirname)
            modality = 'PA'
            experiment_df = ExperimentDf(args)
            model = get_models(args, modality)
            optimizer = optim.Adam(list(model.parameters()))
            logger = SummaryWriter(args.dir_logs_clf)
            callbacks = Callbacks(args, 0, 5, modality,
                                  experiment_df, logger, optimizer)
            loss = 10
            mean_AP = 0.9
            for epoch in range(100):
                loss += 1
                mean_AP -= 0.1
                if callbacks.update_epoch(epoch, loss, mean_AP, model, elapsed_time=1):
                    break
            self.assertEqual(epoch, 6)

    def test_callbacks_decreasing_loss_increasingAP(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            args = Args(tmpdirname)
            modality = 'PA'
            experiment_df = ExperimentDf(args)
            model = get_models(args, modality)
            optimizer = optim.Adam(list(model.parameters()))
            logger = SummaryWriter(args.dir_logs_clf)
            callbacks = Callbacks(args, 0, 5, modality,
                                  experiment_df, logger, optimizer)
            loss = 1000
            mean_AP = 0.1
            for epoch in range(10):
                loss -= 1
                mean_AP += 0.1
                if callbacks.update_epoch(epoch, loss, mean_AP, model, elapsed_time=1):
                    break
            self.assertEqual(epoch, 9)
            self.assertTrue(os.path.exists(f'{tmpdirname}/temp_9'))