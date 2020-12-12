import os
import tempfile
from dataclasses import dataclass
from unittest import TestCase

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from mimic.networks.classifiers.utils import ExperimentDf, get_models, Callbacks, CallbacksProto


@dataclass
class Args(CallbacksProto):
    img_clf_type = 'resnet'
    img_size = 128
    image_channels = 10
    clf_save_m1: str = 'temp'
    dataset: str = 'testing'
    experiment_uid: str = 'temp'
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    distributed: bool = False

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

            for epoch in range(100):
                if epoch == 0:
                    val_results = {'predictions': torch.ones((10, 3)), 'ground_truths': torch.ones((10, 3))}
                else:
                    val_results = {'predictions': torch.zeros((10, 3)), 'ground_truths': torch.ones((10, 3))}
                loss += 1
                if callbacks.update_epoch(epoch, loss, val_results, model, elapsed_time=1):
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
            for epoch in range(10):
                loss -= 1
                if epoch == 0:
                    val_results = {'predictions': torch.cat((torch.ones((1, 3)), torch.zeros((9, 3)))),
                                   'ground_truths': torch.ones((10, 3))}
                else:
                    val_results = {'predictions': torch.cat((torch.ones((epoch, 3)), torch.zeros((10 - epoch, 3)))),
                                   'ground_truths': torch.ones((10, 3))}

                if callbacks.update_epoch(epoch, loss, val_results, model, elapsed_time=1):
                    break
            self.assertEqual(epoch, 9)
            self.assertTrue(os.path.exists(f'{tmpdirname}/temp_9'))
