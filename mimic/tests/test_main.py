import json
import os
import tempfile

import pytest

from mimic.main_mimic import Main
from mimic.utils.filehandling import get_config_path
from mimic.utils.flags import parser
import numpy as np


@pytest.mark.slow
@pytest.mark.parametrize("img_size,text_encoding,feature_extractor_img",
                         [(128, 'word', 'resnet'), (256, 'char', 'resnet'), (256, 'word', 'densenet')])
def test_main(img_size, text_encoding, feature_extractor_img):
    with tempfile.TemporaryDirectory() as tmpdirname:
        config = {
            "img_size": img_size,
            "text_encoding": text_encoding,
            "feature_extractor_img": feature_extractor_img,
            "reduce_lr_on_plateau": True,
            "steps_per_training_epoch": 5,
            "method": "joint_elbo",
            "end_epoch": 2,
            "calc_nll": False,
            "eval_lr": True,
            "calc_prd": False,
            "use_clf": False,
            "batch_size": 50,
            "eval_freq": 1,
            "eval_freq_fid": 1,
            "dir_experiment": tmpdirname,
            "dir_fid": tmpdirname,
            "distributed": False,
        }

        flags = parser.parse_args([])
        config_path = get_config_path()
        with open(config_path, 'rt') as json_file:
            json_config = json.load(json_file)
        json_config.update(config)
        config_path = tmpdirname + 'config.json'
        with open(config_path, 'w') as outfile:
            json.dump(json_config, outfile)
        assert os.path.exists(config_path)
        main = Main(flags, config_path, testing=True)
        main.main()


# if __name__ == '__main__':
    # test_main_densenet()
