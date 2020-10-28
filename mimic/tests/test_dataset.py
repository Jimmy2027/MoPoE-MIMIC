import argparse
import json
import os
import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
import torch

from mimic.utils.filehandling import expand_paths, get_config_path
from mimic.utils.flags import parser


class TestDataset(TestCase):
    def _test_datasets(self, split, img_shape):
        """
        Test to see if datasets have the same shape
        """
        FLAGS = parser.parse_args([])
        config_path = get_config_path()
        with open(config_path, 'rt') as json_file:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(json_file))
            FLAGS = parser.parse_args([], namespace=t_args)
        FLAGS = expand_paths(t_args)
        dir_dataset = os.path.join(FLAGS.dir_data, f'files_small_{img_shape}')
        fn_img_pa = os.path.join(dir_dataset, split + f'_pa.pt')
        fn_img_lat = os.path.join(dir_dataset, split + f'_lat.pt')
        fn_findings = os.path.join(dir_dataset, split + '_findings.csv')
        fn_labels = os.path.join(dir_dataset, split + '_labels.csv')
        str_labels = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']
        self.labels = pd.read_csv(fn_labels)[str_labels].fillna(0)
        self.imgs_pa = torch.load(fn_img_pa)
        self.imgs_lat = torch.load(fn_img_lat)
        self.report_findings = pd.read_csv(fn_findings)['findings']

        assert self.imgs_pa.shape[0] == self.imgs_lat.shape[0] == len(self.labels) == len(
            self.report_findings), \
            f'before removing -1 labels: all modalities must have the same length. len(imgs_pa): {self.imgs_pa.shape[0]}, len(imgs_lat): {self.imgs_lat.shape[0]}, len(labels): {len(self.labels)}, len(report_findings): {len(self.report_findings)}'
        print(
            f'len(imgs_pa): {self.imgs_pa.shape[0]}, len(imgs_lat): {self.imgs_lat.shape[0]}, len(labels): {len(self.labels)}, len(report_findings): {len(self.report_findings)}')
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
        assert self.imgs_pa.shape[0] == self.imgs_lat.shape[0] == len(self.labels) == len(
            self.report_findings), f'after removing -1 labels: all modalities must have the same length. len(imgs_pa): {self.imgs_pa.shape[0]}, len(imgs_lat): {self.imgs_lat.shape[0]}, len(labels): {len(self.labels)}, len(report_findings): {len(self.report_findings)}'

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


if __name__ == '__main__':
    unittest.main()
