import json
import os
import tempfile

import pandas as pd
import pytest

from mimic.main_mimic import Main
from mimic.utils.filehandling import get_config_path
from mimic.utils.flags import parser
from pathlib import Path


def clean_experiment_df(str_experiment):
    experiment_df = pd.read_csv(Path(__file__).parent.parent / 'experiments_dataframe.csv')
    experiment_df.drop(experiment_df.index[experiment_df['str_experiment'] == str_experiment])
    experiment_df.to_csv('experiments_dataframe.csv', index=False)


@pytest.mark.slow
@pytest.mark.parametrize("img_size,text_encoding,feature_extractor_img, only_text_modality",
                         [(128, 'word', 'resnet', False), (256, 'char', 'resnet', False),
                          (256, 'char', 'densenet', False),
                          (256, 'word', 'densenet', False)])
def test_main(img_size, text_encoding, feature_extractor_img, only_text_modality: bool):
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
            "only_text_modality": only_text_modality
        }

        flags = parser.parse_args([])
        config_path = get_config_path()
        with open(config_path, 'rt') as json_file:
            json_config = json.load(json_file)
        json_config.update(config)
        flags.config_path = tmpdirname + 'config.json'
        with open(flags.config_path, 'w') as outfile:
            json.dump(json_config, outfile)
        assert os.path.exists(flags.config_path)
        main = Main(flags, testing=True)
        main.main()

        clean_experiment_df(main.flags.str_experiment)


def test_main_densenet(img_size=128, text_encoding='word', feature_extractor_img='resnet', only_text_modality=False):
    with tempfile.TemporaryDirectory() as tmpdirname:
        config = {
            "img_size": img_size,
            "text_encoding": text_encoding,
            "feature_extractor_img": feature_extractor_img,
            "reduce_lr_on_plateau": True,
            "steps_per_training_epoch": 5,
            "method": "joint_elbo",
            "end_epoch": 2,
            "calc_nll": True,
            "eval_lr": True,
            "calc_prd": False,
            "use_clf": True,
            "batch_size": 20,
            "eval_freq": 3,
            "eval_freq_fid": 2,
            "dir_experiment": tmpdirname,
            "dir_fid": tmpdirname,
            "distributed": False,
            "only_text_modality": only_text_modality,
            "rec_weight_m1": 0.25,
            "rec_weight_m2": 0.25,
            "rec_weight_m3": 0.5,
            "weighted_sampler": True,
            "dataloader_workers": 0,
            "binary_labels": True
        }

        flags = parser.parse_args([])
        config_path = get_config_path()
        with open(config_path, 'rt') as json_file:
            json_config = json.load(json_file)
        json_config.update(config)
        flags.config_path = tmpdirname + 'config.json'
        with open(flags.config_path, 'w') as outfile:
            json.dump(json_config, outfile)
        assert os.path.exists(flags.config_path)
        main = Main(flags, testing=True)
        main.main()

        clean_experiment_df(main.flags.str_experiment)


if __name__ == '__main__':
    test_main_densenet()
