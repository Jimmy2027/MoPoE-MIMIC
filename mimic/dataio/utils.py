from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import PIL.Image as Image
import json
import pandas as pd
from typing import List

from mimic import log


class CustomTransforms:
    def to_RGB(self, x: Image) -> Image:
        return x.convert('RGB')

    def crops_to_tensor(self, crops: torch.Tensor) -> torch.Tensor:
        return torch.stack([transforms.ToTensor()(crop) for crop in crops])

    def normalize_crops(self, crops: torch.Tensor, normalize: transforms.Normalize) -> torch.Tensor:
        return torch.stack([normalize(crop) for crop in crops])

    def foo(self, x):
        return x


def get_transform_img(args: any, img_clf_type: str, clf_training=False):
    if clf_training and args.modality != 'text' and img_clf_type == 'densenet':
        return get_densenet_transforms(args)
    tfs = [
        transforms.ToPILImage(),
        transforms.Resize(size=(args.img_size, args.img_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ]
    if clf_training and args.normalization:
        stats = get_ds_stats(args)
        transforms.Normalize(*stats, inplace=True)

    return transforms.Compose(tfs)


def get_densenet_transforms(args):
    """
    densenet needs RGB images and normalization.
    """
    custom_transforms = CustomTransforms()
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    crops_transform = get_crops_transform(args)
    transformation_list = [
        transforms.ToPILImage(),
        transforms.Lambda(custom_transforms.to_RGB),
        # transforms.Resize(args.img_size)
    ]
    if args.n_crops not in [10, 5]:
        transformation_list.extend([transforms.ToTensor(), normalize])
    else:
        # image is split in n_crops number of crops, stacked and every crop is then normalized
        if args.n_crops == 10:
            transformation_list.append(transforms.TenCrop(224))
        elif args.n_crops == 5:
            crops_transform = transforms.FiveCrop(224)

        transformation_list.extend([crops_transform, transforms.Lambda(
            custom_transforms.crops_to_tensor),
                                    transforms.Lambda(
                                        custom_transforms.normalize_crops)])

    return transforms.Compose(transformation_list)


def get_crops_transform(args) -> transforms:
    if args.n_crops == 10:
        return transforms.TenCrop(224)
    elif args.n_crops == 5:
        return transforms.FiveCrop(224)
    else:
        return transforms.Lambda(lambda x: x)


def calculateWeights(label_df, counts, binary_labels):
    labels = counts.keys()
    if binary_labels:
        for idx, row in label_df.iterrows():
            label_df.at[idx, 'weights'] = int(row[labels[0]] == 1) * 1 / counts[labels[0]] + \
                                          int(row[labels[0]] == 0) * 1 / (len(label_df) - counts[labels[0]])
    else:
        for idx, row in label_df.iterrows():
            label_df.at[idx, 'weights'] = (row[labels[0]]) * 1 / counts[labels[0]] + (
                row[labels[1]]) * 1 / counts[labels[1]] + (row[labels[2]]) * 1 / counts[labels[2]] + (
                                                  row[labels[0]] == row[labels[1]] == row[labels[2]] == 0) * 1 / (
                                                  len(label_df) - counts[labels[0]] - counts[labels[1]] - counts[
                                              labels[2]])
    return torch.DoubleTensor(label_df.weights.values), label_df


def get_ds_stats(args):
    path = Path(__file__).parent.parent / 'data/dataset_stats.json'
    if not path.exists():
        from mimic.dataio.find_dataset_stats import get_mean_std
        log.info("Dataset statistics not found, need to create them. This may take a while.")
        get_mean_std(out_path=path, args=args)
    with open(path, 'r') as jsonfile:
        data = json.load(jsonfile)
    return data['PA_mean'], data['PA_std']


def get_label_counts():
    path = 'dataset_stats.json'
    with open(path, 'r') as jsonfile:
        data = json.load(jsonfile)
    return data['counts']


def get_data_loaders(args, dataset, which_set: str, weighted_sampler: bool = False, nbr_samples_4_sampler: int = -1):
    """
    nbr_samples_4_sampler: how many samples will be sampled with the sampler.
        If set to -1 the same number of samples will be sampled as the length of the dataset.
    """
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        num_workers = args.dataloader_workers // args.world_size
    else:
        sampler = None
        num_workers = args.dataloader_workers
    if which_set == 'train' and weighted_sampler:
        # todo shuffle set
        labels_df = dataset.labels
        label_counts = labels_df[labels_df == 1].count()
        weights, label_weights_df = calculateWeights(labels_df, label_counts, args.binary_labels)
        if nbr_samples_4_sampler == -1:
            nbr_samples_4_sampler = len(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, nbr_samples_4_sampler, replacement=True)
    d_loader = DataLoader(dataset, batch_size=args.batch_size,
                          shuffle=(sampler is None),
                          num_workers=num_workers,
                          sampler=sampler, pin_memory=True)

    assert len(d_loader), f'length of the dataloader needs to be at least 1, it is {len(d_loader)}'

    return sampler, d_loader


def samplers_set_epoch(args, train_sampler, test_sampler, epoch: int) -> None:
    if args.distributed:
        for sampler in [train_sampler, test_sampler]:
            sampler.set_epoch(epoch)


def get_undersample_indices(labels_df: pd.DataFrame):
    count_class_1 = labels_df[labels_df == 1].count().sum()
    df_class_0 = labels_df[labels_df == 0]
    df_class_1 = labels_df[labels_df == 1].dropna(how='all').fillna(0)
    df_class_0_under = df_class_0.sample(count_class_1)
    return [*df_class_1.index.to_list(), *df_class_0_under.index.to_list()]


def filter_labels(labels: pd.DataFrame, which_labels: List[str], undersample_dataset: bool, split: str):
    """
    which_labels: for which labels the class (-1) will be removed.
    Need to remove all cases where the labels have 3 classes.
    The 3rd class (-1) represents "uncertain" and can be removed from the dataset.
    """
    indices = []
    for cl in which_labels:
        indices += labels.index[(labels[cl] == -1)].tolist()
    indices = list(set(indices))
    labels = labels.drop(indices)
    if undersample_dataset and split == 'train':
        labels = undersample(labels)
    return labels


def undersample(labels: pd.DataFrame):
    """
    Undersamples the dataset such that there are the same number of datapoints that have no
    label than datapoints that have a label.
    """
    undersample_indices = get_undersample_indices(labels)
    return labels[labels.index.isin(undersample_indices)]


def get_str_labels(binary_labels):
    if binary_labels:
        return ['Finding']
    else:
        return ['Lung Opacity', 'Pleural Effusion', 'Support Devices']
