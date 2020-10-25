import os
import shutil

import pandas as pd

"""
Cleans all experiments directories as well as rows in the experiments_dataframe, 
where the model was trained less than ** epochs
DO NOT run this when a model is training, it will erase its directory
"""

df = pd.read_csv('experiments_dataframe.csv')
subdf = df.loc[df['total_epochs'] < 10]
for idx, row in subdf.iterrows():
    df = df.drop(idx)
    print(f'deleting {row.dir_experiment_run}')
    if os.path.exists(row.dir_experiment_run):
        shutil.rmtree(row.dir_experiment_run)
df.to_csv('experiments_dataframe.csv', index=False)
