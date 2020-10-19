import json
import os

import pandas as pd
import torch

from mimic.utils.filehandling import get_config_path

dset_str = 'train'
config_path = get_config_path()
with open(config_path, 'rt') as json_file:
    json_config = json.load(json_file)

dir_data = json_config['dir_data']
dir_dataset = os.path.join(dir_data, 'files_small')
fn_img_pa = os.path.join(dir_dataset, dset_str + '_pa.pt')
fn_img_lat = os.path.join(dir_dataset, dset_str + '_lat.pt')
fn_findings = pd.read_csv(os.path.join(dir_dataset, dset_str + '_findings.csv'))
fn_labels = pd.read_csv(os.path.join(dir_dataset, dset_str + '_labels.csv'))

imgs_pa = torch.load(fn_img_pa)
imgs_lat = torch.load(fn_img_lat)

for tensor, label in zip([imgs_pa, imgs_lat], ['pa', 'lat']):
    torch.save(tensor.data[:70].clone(), os.path.join(dir_dataset, 'toy_train_{}.pt'.format(label)))
    torch.save(tensor.data[70: 90].clone(), os.path.join(dir_dataset, 'toy_eval_{}.pt'.format(label)))
    torch.save(tensor.data[90:100].clone(), os.path.join(dir_dataset, 'toy_test_{}.pt'.format(label)))

for table, name in zip([fn_labels, fn_findings], ['labels', 'findings']):
    table[:70].to_csv(os.path.join(dir_dataset, 'toy_train_{}.csv'.format(name)), index=False)
    table[70:90].to_csv(os.path.join(dir_dataset, 'toy_eval_{}.csv'.format(name)), index=False)
    table[90:100].to_csv(os.path.join(dir_dataset, 'toy_test_{}.csv'.format(name)), index=False)
