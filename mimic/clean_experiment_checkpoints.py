import json
import os
import shutil

import pandas as pd

from mimic.utils.filehandling import get_config_path
from pathlib import Path


def clean_mmvae_exp_df(exp_df_path: Path = Path('experiments_dataframe.csv'), min_epoch_exp_dir: int = 140,
                       min_epoch_exp_df: int = 140):
    """
    Cleans all experiments directories as well as rows in the experiments_dataframe,
    where the model was trained less than ** epochs
    DO NOT run this when a model is training, it will erase its working directory.

    min_epoch_exp_dir: all experiments dirs that have a total epoch below min_epoch_exp_dir will be removed.
    min_epoch_exp_df: all rows in the exp df that have a total epoch below min_epoch_exp_df will be removed.
    """

    df = pd.read_csv(exp_df_path)
    subdf = df.loc[df['total_epochs'] < min_epoch_exp_dir]
    for idx, row in subdf.iterrows():
        if row.total_epochs < min_epoch_exp_df or pd.isna(row.dir_experiment_run):
            df = df.drop(idx)
        if not pd.isna(row.dir_experiment_run) and os.path.exists(row.dir_experiment_run):
            print(f'deleting row of {row.dir_experiment_run}')
            shutil.rmtree(row.dir_experiment_run)
    # removing empty columns
    df = df.dropna(how='all', axis=1)
    df.to_csv('experiments_dataframe.csv', index=False)


def clean_clf_exp_df(exp_df_path: Path = Path('clf_experiments_dataframe.csv'), min_epoch: int = 10):
    """
    Cleans all experiments directories as well as rows in the experiments_dataframe,
    where the model was trained less than ** epochs
    DO NOT run this when a model is training, it will erase its working directory.

    min_epoch: all experiments dirs that have a total epoch below min_epoch will be removed.
    """
    df = pd.read_csv(exp_df_path)
    # subdf = df.loc[df['total_epochs'] < 1]
    for idx, row in df.iterrows():
        if pd.isna(row.dir_clf) or row.total_epochs < min_epoch or row.dir_clf.startswith('/scratch/'):
            print(f'dropping row of {row.dir_logs_clf} in dataframe')
            df = df.drop(idx)
            print(f'deleting {row.dir_logs_clf}')
            if not pd.isna(row.dir_logs_clf) and os.path.exists(row.dir_logs_clf):
                shutil.rmtree(row.dir_logs_clf)
    # removing empty columns
    df = df.dropna(how='all', axis=1)
    df.to_csv('clf_experiments_dataframe.csv', index=False)


def clean_fids(config: dict):
    fid_path = Path(config['dir_fid']).expanduser()
    if fid_path.exists():
        for fid in fid_path.iterdir():
            if not any(fid.iterdir()):
                print(f'removing dir {fid}')
                fid.rmdir()


def clean_exp_dirs(config: dict):
    """
    Removes all experiment dirs that don't have a log dir or where the log dir is empty.
    Theses experiment dir are rests from an unsuccessful "rm -r" command.
    """

    checkpoint_path = Path(config['dir_experiment']).expanduser()

    for experiment_dir in checkpoint_path.iterdir():
        if experiment_dir.name.startswith('Mimic') and experiment_dir.is_dir():
            if not os.path.exists(os.path.join(experiment_dir, 'logs')) or len(
                    os.listdir(os.path.join(experiment_dir, 'logs'))) == 0:
                print(f'removing dir {experiment_dir}')
                shutil.rmtree(experiment_dir)
            elif len(
                    os.listdir(os.path.join(experiment_dir, 'checkpoints'))) == 0:
                print(f'removing dir {experiment_dir}')
                shutil.rmtree(experiment_dir)
            # else:
            #     shutil.make_archive(experiment_dir, 'zip', experiment_dir)
            # todo if this works, rm experiment_dir


def clean_clf_Exp_dirs(config: dict):
    """
    Remove all classifier training experiment dirs
    """
    clf_paths = [config['dir_clf'], config['dir_clf'] + '_gridsearch', '~/klugh/mimic/trained_classifiers',
                 '~/klugh/mimic/trained_classifiers_new', '~/klugh/mimic/trained_classifiers_new_new',
                 '~/klugh/mimic/trained_classifiers_gridsearch']
    for clf_path in clf_paths:
        clf_path = os.path.expanduser(clf_path)
        if os.path.exists(clf_path):
            for d in os.listdir(clf_path):
                if d.startswith('logs'):
                    for experiment in os.listdir(os.path.join(clf_path, d)):
                        if os.path.isdir(os.path.join(clf_path, d, experiment)):
                            experiment_dir = os.path.join(clf_path, d, experiment)
                            modality = experiment.split('_')[1]
                            if f'train_clf_{modality}' not in os.listdir(experiment_dir) \
                                    or f'eval_clf_{modality}' not in os.listdir(experiment_dir):
                                shutil.rmtree(experiment_dir)
                                print(f'removing dir {experiment_dir}')


if __name__ == '__main__':
    clean_clf_exp_df()
    clean_mmvae_exp_df()
    config_path = get_config_path()
    with open(config_path, 'rt') as json_file:
        config = json.load(json_file)
    clean_fids(config)
    clean_exp_dirs(config)
    clean_clf_Exp_dirs(config)
