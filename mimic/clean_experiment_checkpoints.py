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
subdf = df.loc[df['total_epochs'] < 10]
for idx, row in subdf.iterrows():
    if row.total_epochs < 5:
        # makes sense to keep experiments in the df for later comparison
        df = df.drop(idx)
    print(f'deleting {row.dir_experiment_run}')
    if os.path.exists(row.dir_experiment_run):
        shutil.rmtree(row.dir_experiment_run)
df.to_csv('experiments_dataframe.csv', index=False)

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
            if os.path.isdir(experiment_dir):
                if not os.path.exists(os.path.join(experiment_dir, 'logs')) or len(
                        os.listdir(os.path.join(experiment_dir, 'logs'))) == 0:
                    print(f'removing dir {experiment_dir}')
                    shutil.rmtree(experiment_dir)
                if len(
                        os.listdir(os.path.join(experiment_dir, 'checkpoints'))) == 0:
                    print(f'removing dir {experiment_dir}')
                    shutil.rmtree(experiment_dir)
                # else:
                #     shutil.make_archive(experiment_dir, 'zip', experiment_dir)
                # todo if this works, rm experiment_dir
