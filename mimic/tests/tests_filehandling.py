import argparse
import os
import tempfile
from unittest import TestCase

from utils.filehandling import create_dir_structure


class TestFilehandling(TestCase):
    def test_create_dir_structure(self):
        """
        Checks if function create_dir_structure is runnnig
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            parser = argparse.ArgumentParser()
            flags = parser.parse_args()
            flags.dir_experiment = tmpdirname
            flags.dir_fid = None
            flags.dataset = 'mimic'
            flags.dir_clf = os.path.join(tmpdirname, 'clf_dir')
            _ = create_dir_structure(flags)
