import json
import os
import shutil

import pandas as pd

from mimic.utils.filehandling import get_config_path

"""
Cleans all experiments directories as well as rows in the experiments_dataframe, 
where the model was trained less than ** epochs
DO NOT run this when a model is training, it will erase its directory
"""

df = pd.read_csv('experiments_dataframe.csv')
subdf = df.loc[df['total_epochs'] < 8]
for idx, row in subdf.iterrows():
    if row.total_epochs < 5 or pd.isna(row.dir_experiment_run):
        df = df.drop(idx)
    if not pd.isna(row.dir_experiment_run) and os.path.exists(row.dir_experiment_run):
        print(f'deleting row of {row.dir_experiment_run}')
        shutil.rmtree(row.dir_experiment_run)
# removing empty columns
df = df.dropna(how='all', axis=1)
df.to_csv('experiments_dataframe.csv', index=False)

""" 
Same for classifier experiments:
"""
df = pd.read_csv('clf_experiments_dataframe.csv')
# subdf = df.loc[df['total_epochs'] < 1]
for idx, row in df.iterrows():
    if pd.isna(row.dir_clf) or row.total_epochs < 3 or row.dir_clf.startswith('/scratch/'):
        print(f'dropping row of {row.dir_logs_clf} in dataframe')
        df = df.drop(idx)
        print(f'deleting {row.dir_logs_clf}')
        if not pd.isna(row.dir_logs_clf) and os.path.exists(row.dir_logs_clf):
            shutil.rmtree(row.dir_logs_clf)
# removing empty columns
df = df.dropna(how='all', axis=1)
df.to_csv('clf_experiments_dataframe.csv', index=False)

"""
Removes all experiment dirs that don't have a log dir or where the log dir is empty. 
Theses experiment dir are rests from an unsuccessful "rm -r" command.
"""
config_path = get_config_path()
with open(config_path, 'rt') as json_file:
    config = json.load(json_file)

checkpoint_path = os.path.expanduser(config['dir_fid'])

for modality_method in ['moe']:
    for factorization in os.listdir(os.path.join(checkpoint_path, modality_method)):
        for experiment in os.listdir(os.path.join(checkpoint_path, modality_method, factorization)):
            experiment_dir = os.path.join(checkpoint_path, modality_method, factorization, experiment)
            if experiment.startswith('Mimic') and os.path.isdir(
                    experiment_dir
            ):
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
