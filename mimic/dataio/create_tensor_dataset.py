import os
import shutil
import zipfile
from glob import glob
from typing import Tuple

import PIL.Image as Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm

trans1 = transforms.ToTensor()


class CreateTensorDataset:
    def __init__(self, dir_base_resize: str, dir_mimic: str, dir_out: str, img_size: Tuple,
                 dir_base_resized_compressed: str = '', max_it: int = -1):
        """
        dir_out: where the tensor dataset will be saved
        dir_base_resize: where the resized image are saved
        dir_base_resize (optional): where the compressed resized images are. It is recommended to compress the resized
        images after their usage and delete the non-compressed ones to save space.
        max_it: (optional) maximum iterations. Use this for testing only. if -1 (default): do all

        """
        self.dir_base_orig = os.path.join(dir_mimic, 'files')
        self.dir_base_resize = dir_base_resize
        self.dir_base_resized_compressed = dir_base_resized_compressed
        self.dir_resized_compressed = os.path.join(dir_base_resized_compressed, f'mimic_resized_{img_size[0]}.zip')
        self.dir_out = dir_out
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)
        self.fn_train = os.path.join(dir_mimic, 'train.csv')
        self.fn_eval = os.path.join(dir_mimic, 'eval.csv')
        self.fn_test = os.path.join(dir_mimic, 'test.csv')
        self.df_train = pd.read_csv(self.fn_train)
        self.df_eval = pd.read_csv(self.fn_eval)
        self.df_test = pd.read_csv(self.fn_test)
        self.max_it = max_it
        self.img_size = img_size

    def __call__(self):
        splits = ['train', 'eval', 'test']
        dataframes = [self.df_train, self.df_eval, self.df_test]
        for split, df in zip(splits, dataframes):
            self.create_dataset(df, split=split)

    def create_dataset(self, df: pd.DataFrame, split: str):
        """
        dir_out: where the tensor dataset should be created
        """
        dir_src = self.dir_base_resize  # directory of the resized images
        if not os.path.exists(dir_src):
            if os.path.exists(self.dir_resized_compressed):
                print(f'compressed images are found and are decompressed to {dir_src}')
                os.mkdir(dir_src)
                with zipfile.ZipFile(self.dir_resized_compressed, 'r') as zip_ref:
                    zip_ref.extractall(dir_src)
            else:
                print(
                    f'directory of resized images {dir_src} does not exist and needs to be created. This may take a while.')
                _ = self._resize_all()
        dir_out = self.dir_out
        num_samples = df.shape[0] if self.max_it < 0 else self.max_it
        imgs_pa = torch.Tensor(num_samples, self.img_size[0], self.img_size[0])
        imgs_lat = torch.Tensor(num_samples, self.img_size[0], self.img_size[0])
        ind = torch.Tensor(num_samples)  # tensor that indicates the images that were found
        labels = df.filter(
            ['Atelectasis', 'Cardiomegaly', 'Lung Opacity', 'Pleural Effusion', 'Support Devices', 'No Finding'],
            axis=1)
        findings = df.filter(['findings'])
        impressions = df.filter(['impression'])
        for index, row in tqdm(df.iterrows(), postfix=split):
            if self.max_it > 0 and index >= self.max_it:
                break
            # load lat and pa image for each row
            s_id = row['study_id']
            p_id = row['subject_id']
            pa_id = row['pa_dicom_id']
            lat_id = row['lat_dicom_id']
            p_supdir = 'p' + str(p_id)[:2]
            p_dir = 'p' + str(p_id)
            s_dir = 's' + str(s_id)
            dir_img = os.path.join(dir_src, p_supdir, p_dir, s_dir)
            fn_pa_src = os.path.join(dir_img, str(pa_id) + '.jpg')
            fn_lat_src = os.path.join(dir_img, str(lat_id) + '.jpg')
            ind[index] = 0  # 0 indicates that the image was not found

            try:
                img_pa = self._load_image(fn_pa_src)
            except FileNotFoundError as e:
                print(e)
                findings = findings.drop(index, axis=0)
                impressions = impressions.drop(index, axis=0)
                labels = labels.drop(index, axis=0)
                ind[index] = 0
                continue
            try:
                img_lat = self._load_image(fn_lat_src)
            except FileNotFoundError as e:
                print(e)
                findings = findings.drop(index, axis=0)
                impressions = impressions.drop(index, axis=0)
                labels = labels.drop(index, axis=0)
                ind[index] = 0
                continue
            ind[index] = 1  # 1 indicates that the image was found
            imgs_pa[index, :, :] = img_pa
            imgs_lat[index, :, :] = img_lat
        print(imgs_pa.shape)
        mask = ind > 0
        if self.max_it < 0:
            # only do this if not test run
            imgs_pa = imgs_pa[mask, :, :]
            imgs_lat = imgs_lat[mask, :, :]
            print(imgs_pa.shape)

            # need to remove all cases where the labels have 3 classes
            indices = []
            indices += labels.index[(labels['Lung Opacity'] == -1)].tolist()
            indices += labels.index[(labels['Pleural Effusion'] == -1)].tolist()
            indices += labels.index[(labels['Support Devices'] == -1)].tolist()
            indices = list(set(indices))
            labels = labels.drop(indices)
            findings = findings.drop(indices)
            imgs_pa = torch.tensor(np.delete(imgs_pa.numpy(), indices, 0))
            imgs_lat = torch.tensor(np.delete(imgs_lat.numpy(), indices, 0))

        fn_pa_out = os.path.join(dir_out, split + f'_pa{self.img_size[0]}.pt')
        fn_lat_out = os.path.join(dir_out, split + f'_lat{self.img_size[0]}.pt')
        fn_findings_out = os.path.join(dir_out, split + '_findings.csv')
        fn_impressions_out = os.path.join(dir_out, split + '_impressions.csv')
        fn_labels_out = os.path.join(dir_out, split + '_labels.csv')
        torch.save(imgs_pa, fn_pa_out)
        torch.save(imgs_lat, fn_lat_out)
        print(findings.shape)
        findings[:self.max_it].to_csv(fn_findings_out)
        impressions[:self.max_it].to_csv(fn_impressions_out)
        labels[:self.max_it].to_csv(fn_labels_out)

        assert imgs_pa.shape[0] == imgs_lat.shape[0] == len(labels[:self.max_it]) == len(
            findings[
            :self.max_it]), f'all modalities must have the same length. len(imgs_pa): {imgs_pa.shape[0]}, len(imgs_lat): {imgs_lat.shape[0]}, len(labels): {len(labels[:self.max_it])}, len(report_findings): {len(findings[:self.max_it])}'

        if self.dir_base_resized_compressed:
            if not os.path.exists(self.dir_resized_compressed):
                print('zipping resized images folder {} to {} -> {}'.format(dir_src,
                                                                            self.dir_resized_compressed.split('.')[0],
                                                                            self.dir_resized_compressed))

                shutil.make_archive(self.dir_resized_compressed.replace('.zip', ''), 'zip', dir_src, verbose=True)
                assert os.path.exists(
                    self.dir_resized_compressed), 'path does not exist: {}. \n {}'.format(
                    self.dir_resized_compressed, os.listdir(self.dir_base_resized_compressed))
            # this is automatically done if using tmpdir
            # print(f'deleting resized images folder: {dir_src}')
            # shutil.rmtree(dir_src)

    def _fast_scandir(self, dirname):
        subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
        for dirname in list(subfolders):
            subfolders.extend(self._fast_scandir(dirname))
        return subfolders

    def _center_crop_and_resize(self, img):
        width, height = img.size  # Get dimensions
        new_width = min(width, height)
        new_height = min(width, height)
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        # Crop the center of the image
        img_crop = img.crop((left, top, right, bottom))
        img_resized = img_crop.resize(self.img_size, Image.ANTIALIAS)
        return img_resized

    def _compute_total_folders(self, path: str):
        total_folders = len(list(os.walk(path)))
        return total_folders

    def _resize_all(self):
        """
        Walks through the directories of original mimic data and resizes the images to self.img_size
        """
        dir_src_base = self.dir_base_orig
        print('computing total folders')
        total_folders = self._compute_total_folders(dir_src_base) if self.max_it < 0 else None
        count_imgs = 0
        for dirName, subdirList, fileList in tqdm(os.walk(dir_src_base), total=total_folders, postfix='resize'):
            f_imgs = glob(os.path.join(dirName, '*.jpg'))
            if len(f_imgs) > 0:
                d_new = os.path.join(self.dir_base_resize, '/'.join(dirName.split('/')[-3:]))
                if not os.path.exists(d_new):
                    os.makedirs(d_new)
                for f in f_imgs:
                    if self.max_it > 0 and count_imgs >= self.max_it:
                        return True
                    n_img = f.split('/')[-1]
                    f_new = os.path.join(d_new, n_img)
                    if not os.path.exists(f_new):
                        try:
                            img_src = Image.open(f)
                            img_new = self._center_crop_and_resize(img_src)
                            img_new.save(f_new, "JPEG")
                            img_src.close()
                            img_new.close()
                        except OSError:
                            print('file could not be opened...')
                    count_imgs += 1
        return True

    def _load_image(self, fn_img):
        img = Image.open(fn_img)
        img_t = trans1(img)
        return img_t


if __name__ == '__main__':
    img_size = (256, 256)
    dir_mimic = '/cluster/work/vogtlab/Projects/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0'
    dir_out = os.path.expanduser('~/klugh/files_small_new')
    dir_base_resized_compressed = f'/cluster/work/vogtlab/Group/klugh/'
    assert os.path.exists(os.path.expandvars('$TMPDIR'))
    assert os.path.exists(dir_base_resized_compressed)

    dir_base_resize = os.path.join(os.path.expandvars('$TMPDIR'), f'files_small_{img_size[0]}')
    dataset_creator = CreateTensorDataset(dir_base_resize=dir_base_resize, dir_mimic=dir_mimic, dir_out=dir_out,
                                          img_size=img_size, dir_base_resized_compressed=dir_base_resized_compressed)
    dataset_creator()
