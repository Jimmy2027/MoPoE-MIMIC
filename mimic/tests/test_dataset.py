import os
import unittest
from itertools import product
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
import torch

from mimic.dataio.utils import filter_labels
from mimic.utils.experiment import MimicExperiment
from mimic.utils.filehandling import expand_paths, get_config_path
from mimic.utils.flags import parser
from mimic.utils.flags import update_flags_with_config, setup_flags

COUNT_LABELS = {
    'train': {'Lung Opacity': [53686, 6775], 'Pleural Effusion': [56852, 3609], 'Support Devices': [59555, 906]},
    'eval': {'Lung Opacity': [423, 52], 'Pleural Effusion': [448, 27], 'Support Devices': [463, 12]},
    'test': {'Lung Opacity': [354, 126], 'Pleural Effusion': [413, 67], 'Support Devices': [466, 14]}
}
STR_LABELS = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']


@pytest.mark.dataset
class TestDataset(TestCase):
    def _load_data(self, flags, img_shape, split, load_images=True):

        dir_dataset = os.path.join(flags.dir_data, f'files_small_{img_shape}')
        fn_img_pa = os.path.join(dir_dataset, split + f'_pa.pt')
        fn_img_lat = os.path.join(dir_dataset, split + f'_lat.pt')
        fn_findings = os.path.join(dir_dataset, split + '_findings.csv')
        fn_labels = os.path.join(dir_dataset, split + '_labels.csv')
        labels = pd.read_csv(fn_labels)[[*STR_LABELS, 'uid']].fillna(0)
        report_findings = pd.read_csv(fn_findings)[['findings', 'uid']]
        if not load_images:
            return labels, report_findings
        imgs_pa = torch.load(fn_img_pa)
        imgs_lat = torch.load(fn_img_lat)
        return labels, imgs_pa, imgs_lat, report_findings

    def _loadflags(self):
        FLAGS = parser.parse_args([])
        FLAGS.config_path = get_config_path()
        FLAGS = update_flags_with_config(FLAGS.config_path, testing=True)
        return expand_paths(FLAGS)

    def _test_datasets(self, split, img_shape, undersample_dataset=False):
        """
        Test to see if different datasets all have the same shape.
        """
        FLAGS = self._loadflags()

        self.labels, self.imgs_pa, self.imgs_lat, self.report_findings = self._load_data(FLAGS, img_shape, split)

        assert self.imgs_pa.shape[0] == self.imgs_lat.shape[0] == len(self.labels) == len(
            self.report_findings), \
            f'before removing -1 labels: all modalities must have the same length. ' \
            f'len(imgs_pa): {self.imgs_pa.shape[0]}, len(imgs_lat): {self.imgs_lat.shape[0]}, ' \
            f'len(labels): {len(self.labels)}, len(report_findings): {len(self.report_findings)}'
        print(
            f'len(imgs_pa): {self.imgs_pa.shape[0]}, len(imgs_lat): {self.imgs_lat.shape[0]}, '
            f'len(labels): {len(self.labels)}, len(report_findings): {len(self.report_findings)}')

        self.labels = filter_labels(self.labels, undersample_dataset, split)

        self.labels = self.labels.values
        self.report_findings = self.report_findings.values

        assert len(np.unique(self.labels)) == 2, \
            'labels should contain 2 classes, might need to remove -1 labels'
        assert self.imgs_pa.shape[0] == self.imgs_lat.shape[0] == len(self.report_findings), \
            f'after removing -1 labels: all modalities must have the same length. ' \
            f'len(imgs_pa): {self.imgs_pa.shape[0]}, len(imgs_lat): {self.imgs_lat.shape[0]}, ' \
            f'len(report_findings): {len(self.report_findings)}'

        if not undersample_dataset:
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
        FLAGS.config_path = get_config_path()
        FLAGS = setup_flags(FLAGS, testing=True)

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

    def test_undersample_dataset(self):
        self._test_datasets('eval', 128, undersample_dataset=True)

    def test_torch_dataset_with_ds(self):
        """
        Compares the torch dataset with the original dataset by comparing labels and findings with a uid.
        """
        for split, img_shape, undersample_dataset in product(['train', 'eval', 'test'], [128, 256], [True, False]):
            FLAGS = self._loadflags()
            labels, report_findings = self._load_data(FLAGS, img_shape, split, load_images=False)
            labels = filter_labels(labels, undersample_dataset, split)
            df_train = pd.read_csv(Path(FLAGS.dir_data) / f'{split}.csv').fillna(0)
            df_train['uid'] = df_train['pa_dicom_id'] + '_' + df_train['lat_dicom_id']

            for _, label_row in labels.iterrows():
                findings_row = report_findings.loc[report_findings['uid'] == label_row.uid]
                df_train_row = df_train.loc[df_train['uid'] == label_row.uid]
                for label in STR_LABELS:
                    assert df_train_row[label].item() == label_row[label], \
                        f'value of train_df {df_train_row[label].item()} does not match value of label_df ' \
                        f'{label_row[label].item()} for label {label}.'
                assert df_train_row['findings'].item() == findings_row['findings'].item(), \
                    f'value of train_df {df_train_row["findings"].item()} does not match value of findings_df ' \
                    f'{findings_row["findings"].item()}.'


if __name__ == '__main__':
    unittest.main()
