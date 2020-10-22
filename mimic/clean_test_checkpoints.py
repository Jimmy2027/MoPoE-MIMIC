import os
from mimic.utils.filehandling import create_dir_structure, expand_paths, create_dir_structure_testing, get_config_path
import json
import shutil
from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd

# todo can check where total_epochs < 100 to erase test checkpoints
df = pd.read_csv('experiments_dataframe.csv')

config_path = get_config_path()
with open(config_path, 'rt') as json_file:
    config = json.load(json_file)

checkpoint_path = config['dir_fid']

for modality_method in ['moe']:
    for factorization in os.listdir(os.path.join(checkpoint_path, modality_method)):
        for experiment in os.listdir(os.path.join(checkpoint_path, modality_method, factorization)):
            experiment_dir = os.path.join(checkpoint_path, modality_method, factorization, experiment)
            if os.path.isdir(experiment_dir):
                if not os.path.exists(os.path.join(experiment_dir, 'logs')):
                    # this removes trained clfs on bartholin!
                    #     shutil.rmtree(experiment_dir)
                    pass
                else:
                    for file in os.listdir(os.path.join(experiment_dir, 'logs')):
                        print(file)
                        if file.startswith('events.out'):
                            for summary in summary_iterator(
                                    os.path.join(checkpoint_path, modality_method, factorization, experiment, 'logs',
                                                 file)):
                                logs = summary
                                if logs.step > 2:
                                    print(logs, logs.step)
