import os

import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import PIL.Image as Image

import torch

from utils import text as text


class Mimic(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, args, str_labels, alphabet, dataset):
        if dataset == 1:
            filename_csv = os.path.join(args.dir_data, 'train.csv');
            dset_str = 'train';
        elif dataset == 2:
            filename_csv = os.path.join(args.dir_data, 'eval.csv');
            dset_str = 'eval';
        elif dataset == 3:
            dset_str = 'test';
        if args.use_toy_dataset:
            dset_str = 'toy_' + dset_str
        self.args = args;
        dir_dataset = os.path.join(args.dir_data, 'files_small');
        fn_img_pa = os.path.join(dir_dataset, dset_str + '_pa.pt');
        fn_img_lat = os.path.join(dir_dataset, dset_str + '_lat.pt');
        fn_findings = os.path.join(dir_dataset, dset_str + '_findings.csv');
        fn_labels = os.path.join(dir_dataset, dset_str + '_labels.csv');

        self.labels = pd.read_csv(fn_labels)[str_labels].fillna(0).values;
        self.imgs_pa = torch.load(fn_img_pa);
        self.imgs_lat = torch.load(fn_img_lat);
        self.report_findings = pd.read_csv(fn_findings)['findings'].values;
        self.alphabet = alphabet;
        self.transform_img = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.Resize(size=(self.args.img_size, self.args.img_size),
                                                                   interpolation=Image.BICUBIC),
                                                 transforms.ToTensor()])

    def __getitem__(self, index):
        try:
            img_pa = self.imgs_pa[index, :, :]
            img_lat = self.imgs_lat[index, :, :]
            img_pa = self.transform_img(img_pa)
            img_lat = self.transform_img(img_lat)
            text_str = self.report_findings[index]
            if len(text_str) > self.args.len_sequence:
                text_str = text_str[:self.args.len_sequence]
            text_vec = text.one_hot_encode(self.args.len_sequence, self.alphabet, text_str)
            label = torch.from_numpy((self.labels[index, :]).astype(int)).float()
            sample = {'PA': img_pa, 'Lateral': img_lat, 'text': text_vec}
        except (IndexError, OSError):
            return None;
        return sample, label

    def __len__(self):
        return self.labels.shape[0]

    def get_text_str(self, index):
        return self.y[index];
