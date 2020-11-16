import json
import os
import tempfile

import pytest

from mimic.main_mimic import Main
from mimic.utils.filehandling import get_config_path
from mimic.utils.flags import parser


@pytest.mark.slow
@pytest.mark.parametrize("img_size,text_encoding", [(128, 'word'), (256, 'char')])
def test_main(img_size, text_encoding):
    with tempfile.TemporaryDirectory() as tmpdirname:
        config = {
            "img_size": img_size,
            "text_encoding": text_encoding,
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
        success = main.run_epochs()
