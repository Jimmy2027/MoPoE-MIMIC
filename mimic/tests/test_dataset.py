import os
import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
import torch

from mimic.utils.experiment import MimicExperiment
from mimic.utils.filehandling import expand_paths, get_config_path
from mimic.utils.flags import parser
from mimic.utils.flags import update_flags_with_config, setup_flags
import pytest

COUNT_LABELS = {
    'train': {'Lung Opacity': [53686, 6775], 'Pleural Effusion': [56852, 3609], 'Support Devices': [59555, 906]},
    'eval': {'Lung Opacity': [423, 52], 'Pleural Effusion': [448, 27], 'Support Devices': [463, 12]},
    'test': {'Lung Opacity': [354, 126], 'Pleural Effusion': [413, 67], 'Support Devices': [466, 14]}
}
STR_LABELS = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']


@pytest.mark.dataset
class TestDataset(TestCase):
    def _test_datasets(self, split, img_shape):
        """
        Test to see if datasets have the same shape
        """
        FLAGS = parser.parse_args([])
        config_path = get_config_path()
        FLAGS = update_flags_with_config(FLAGS, config_path, testing=True)

        FLAGS = expand_paths(FLAGS)
        dir_dataset = os.path.join(FLAGS.dir_data, f'files_small_{img_shape}')
        fn_img_pa = os.path.join(dir_dataset, split + f'_pa.pt')
        fn_img_lat = os.path.join(dir_dataset, split + f'_lat.pt')
        fn_findings = os.path.join(dir_dataset, split + '_findings.csv')
        fn_labels = os.path.join(dir_dataset, split + '_labels.csv')
        self.labels = pd.read_csv(fn_labels)[STR_LABELS].fillna(0)
        self.imgs_pa = torch.load(fn_img_pa)
        self.imgs_lat = torch.load(fn_img_lat)
        self.report_findings = pd.read_csv(fn_findings)['findings']

        assert self.imgs_pa.shape[0] == self.imgs_lat.shape[0] == len(self.labels) == len(
            self.report_findings), \
            f'before removing -1 labels: all modalities must have the same length. ' \
            f'len(imgs_pa): {self.imgs_pa.shape[0]}, len(imgs_lat): {self.imgs_lat.shape[0]}, ' \
            f'len(labels): {len(self.labels)}, len(report_findings): {len(self.report_findings)}'
        print(
            f'len(imgs_pa): {self.imgs_pa.shape[0]}, len(imgs_lat): {self.imgs_lat.shape[0]}, '
            f'len(labels): {len(self.labels)}, len(report_findings): {len(self.report_findings)}')
        # need to remove all cases where the labels have 3 classes
        indices = []
        indices += self.labels.index[(self.labels['Lung Opacity'] == -1)].tolist()
        indices += self.labels.index[(self.labels['Pleural Effusion'] == -1)].tolist()
        indices += self.labels.index[(self.labels['Support Devices'] == -1)].tolist()
        indices = list(set(indices))
        self.labels = self.labels.drop(indices).values
        self.report_findings = self.report_findings.drop(indices).values
        self.imgs_pa = torch.tensor(np.delete(self.imgs_pa.numpy(), indices, 0))
        self.imgs_lat = torch.tensor(np.delete(self.imgs_lat.numpy(), indices, 0))

        assert len(np.unique(self.labels)) == 2, \
            'labels should contain 2 classes, might need to remove -1 labels'
        assert self.imgs_pa.shape[0] == self.imgs_lat.shape[0] == len(self.labels) == len(self.report_findings), \
            f'after removing -1 labels: all modalities must have the same length. ' \
            f'len(imgs_pa): {self.imgs_pa.shape[0]}, len(imgs_lat): {self.imgs_lat.shape[0]}, ' \
            f'len(labels): {len(self.labels)}, len(report_findings): {len(self.report_findings)}'

        for idx, label in enumerate(STR_LABELS):
            values, counts = np.unique(self.labels[:, idx], return_counts=True)
            self.assertTrue(all(counts == COUNT_LABELS[split][label]))

    def test_train_dataset_128(self):
        self._test_datasets('train', 128)

    def test_test_dataset_128(self):
        self._test_datasets('test', 128)

    def test_eval_dataset_128(self):
        self._test_datasets('eval', 128)

    def test_train_dataset_256(self):
        self._test_datasets('train', 256)

    def test_test_dataset_256(self):
        self._test_datasets('test', 256)

    def test_eval_dataset_256(self):
        self._test_datasets('eval', 256)

    def _test_lencount_labels(self, img_size: int):
        """
        The dataset should remain the same. This test tests if it was somehow changed.
        """
        FLAGS = parser.parse_args([])
        config_path = get_config_path()
        FLAGS = setup_flags(FLAGS, config_path, testing=True)

        FLAGS.str_experiment = 'temp'
        FLAGS.dir_gen_eval_fid = ''
        FLAGS.img_size = img_size
        mimic = MimicExperiment(FLAGS)

        for idx, label in enumerate(STR_LABELS):
            train_value, train_count = np.unique(mimic.dataset_train.labels[:, idx], return_counts=True)
            test_value, test_count = np.unique(mimic.dataset_test.labels[:, idx], return_counts=True)

            self.assertTrue(all(train_value == [0, 1]))
            self.assertTrue(all(train_count == COUNT_LABELS['train'][label]))

            self.assertTrue(all(test_value == [0, 1]))
            self.assertTrue(all(test_count == COUNT_LABELS['eval'][label]))

    def test_lencount_labels_128(self):
        self._test_lencount_labels(128)

    def test_lencount_labels_256(self):
        self._test_lencount_labels(256)


if __name__ == '__main__':
    unittest.main()
