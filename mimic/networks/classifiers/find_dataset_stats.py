# HK, 28.01.21

import json
import os

import pandas as pd
from torch.utils.data import DataLoader

from mimic import log
from mimic.dataio.MimicDataset import Mimic
from mimic.networks.classifiers.utils import LABELS
from mimic.utils.filehandling import get_config_path
from mimic.utils.flags import parser
from mimic.utils.flags import update_flags_with_config


def write_results_to_json(results: dict, path: str = 'dataset_stats.json'):
    log.info(f'Writing to dict: {results} to {path}')
    if os.path.exists(path):
        with open(path, 'r') as jsonfile:
            data = json.load(jsonfile)
        results = {**results, **data}
    with open(path, 'w') as outfile:
        json.dump(results, outfile)


def get_mean_std(d_loader):
    # taken from https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/2

    for mod in ['PA', 'Lateral']:
        mean = 0.
        std = 0.
        for images, _ in d_loader:
            images = images[mod]
            batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)

        mean /= len(d_loader.dataset)
        std /= len(d_loader.dataset)
        stats = {f'{mod}_mean': mean.item(), f'{mod}_std': std.item()}
        log.info(stats)
        write_results_to_json(results=stats)


def get_label_counts(args):
    dir_dataset = os.path.join(args.dir_data, 'files_small_128')
    train_labels_path = os.path.join(dir_dataset, 'train_labels.csv')
    train_labels_df = pd.read_csv(train_labels_path)[LABELS].fillna(0)
    indices = []
    indices += train_labels_df.index[(train_labels_df['Lung Opacity'] == -1)].tolist()
    indices += train_labels_df.index[(train_labels_df['Pleural Effusion'] == -1)].tolist()
    indices += train_labels_df.index[(train_labels_df['Support Devices'] == -1)].tolist()
    indices = list(set(indices))
    train_labels_df = train_labels_df.drop(indices)
    counts = train_labels_df[train_labels_df == 1].count()
    write_results_to_json(results={'counts': counts.to_dict()})


if __name__ == '__main__':
    FLAGS = parser.parse_args()

    config_path = get_config_path(FLAGS)
    FLAGS = update_flags_with_config(config_path)
    FLAGS.modality = 'PA'

    get_label_counts(FLAGS)

    trainset = Mimic(FLAGS, LABELS, split='train', clf_training=False)
    trainset_loader = DataLoader(trainset, batch_size=50, shuffle=False, num_workers=0)

    get_mean_std(trainset_loader)
