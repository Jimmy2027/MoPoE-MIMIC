import argparse
import os
import tempfile
from unittest import TestCase
import unittest
from mimic.utils.filehandling import create_dir_structure
from mimic.dataio.create_tensor_dataset import CreateTensorDataset


class TestFilehandling(TestCase):
    def test_create_tensor_dataset(self):
        """
        Checks if function create_dir_structure is runnnig
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            img_size = (256, 256)
            dir_mimic = '/cluster/work/vogtlab/Projects/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0'
            dir_out = os.path.expanduser('~/scratch/files_small')
            dir_base_resized_compressed = f'/cluster/work/vogtlab/Group/klugh/mimic_images_{img_size[0]}.zip'
            assert os.path.exists(os.path.expandvars('$TMPDIR'))
            dir_base_resize = os.path.join(os.path.expandvars('$TMPDIR'), f'files_small_{img_size[0]}')
            dataset_creator = CreateTensorDataset(dir_base_resize=dir_base_resize, dir_mimic=dir_mimic, dir_out=dir_out,
                                                  img_size=img_size,
                                                  dir_base_resized_compressed=dir_base_resized_compressed)
            dataset_creator()


if __name__ == '__main__':
    unittest.main()
