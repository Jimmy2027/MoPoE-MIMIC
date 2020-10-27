import os
import tempfile
import unittest
from unittest import TestCase

from mimic.dataio.create_tensor_dataset import CreateTensorDataset


class TestCreateTensorDataset(TestCase):
    def _create_tensor_dataset(self, img_size):
        """
        Checks if function create_dir_structure is running.
        This tests only works if dir_mimic exists. The test passes if it doesn't
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            dir_mimic = '/cluster/work/vogtlab/Projects/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0'
            if os.path.exists(dir_mimic):
                dir_out = os.path.expanduser(os.path.join(tmpdirname, 'dir_out'))
                dir_base_resized_compressed = os.path.expanduser(os.path.join(tmpdirname))
                assert os.path.exists(tmpdirname)
                dir_base_resize = os.path.join(tmpdirname, f'files_small_{img_size[0]}')
                dataset_creator = CreateTensorDataset(dir_base_resize=dir_base_resize, dir_mimic=dir_mimic,
                                                      dir_out=dir_out,
                                                      img_size=img_size,
                                                      dir_base_resized_compressed=dir_base_resized_compressed,
                                                      max_it=10)
                dataset_creator()
                assert os.path.exists(dir_out)
                assert os.path.exists(dir_base_resized_compressed)
                assert os.path.exists(os.path.join(dir_base_resized_compressed,
                                                   f'mimic_resized_{img_size[0]}.zip')), \
                    'dir_resized_compressed {} does not exist \n {}'.format(
                        os.path.join(dir_base_resized_compressed, f'mimic_resized_{img_size[0]}.zip'),
                        os.listdir(dir_base_resized_compressed))

    def test_create_tensor_dataset_128(self):
        img_size = (128, 128)
        self._create_tensor_dataset(img_size)

    def test_create_tensor_dataset_256(self):
        img_size = (256, 256)
        self._create_tensor_dataset(img_size)


if __name__ == '__main__':
    unittest.main()
