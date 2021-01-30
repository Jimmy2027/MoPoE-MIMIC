import io
import json
import os
import pickle
import random
import typing
from collections import Counter, OrderedDict
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from torch import Tensor
from torch.utils.data import Dataset

from mimic.dataio.utils import get_transform_img, filter_labels
from mimic.utils import text as text
from mimic.utils.utils import get_alphabet


class Mimic(Dataset):
    """Custom Dataset for loading mimic images"""

    def __init__(self, args, str_labels, split: str, clf_training=False):
        """
        split: string, either train, eval or test
        clf_training: set to true if the dataset is used to train a classifier. In this case the densenet
        transformations can be applied directly in the dataset transformations.
        """
        self.args = args
        self.split = split
        dir_dataset = os.path.join(args.dir_data, f'files_small_{args.img_size}')
        fn_img_pa = os.path.join(dir_dataset, split + '_pa.pt')
        fn_img_lat = os.path.join(dir_dataset, split + '_lat.pt')
        fn_findings = os.path.join(dir_dataset, split + '_findings.csv')
        fn_labels = os.path.join(dir_dataset, split + '_labels.csv')

        self.labels = pd.read_csv(fn_labels)[str_labels].fillna(0)
        self.imgs_pa = torch.load(fn_img_pa)
        self.imgs_lat = torch.load(fn_img_lat)
        self.report_findings = pd.read_csv(fn_findings)['findings']
        # need to filter out labels that contain the label "-1"
        self.labels = filter_labels(self.labels, args.undersample_dataset)

        self._verify_dataset()

        if self.args.text_encoding == 'char':
            self.args.alphabet = get_alphabet()
            args.num_features = len(self.args.alphabet)
        else:
            # if word_encoding == word, need dataset for report_findings that contains the encodings.
            self.get_report_findings_dataset(dir_dataset)

        self.transform_img = get_transform_img(args, args.feature_extractor_img, clf_training)

        if self.args.text_encoding == 'char':
            self.get_vec = self.get_char_text_vec
        elif self.args.text_encoding == 'word':
            self.get_vec = self.get_word_text_vec
        else:
            raise NotImplementedError(f'{self.args.text_encoding} has to be either char or word')

    def __getitem__(self, label_index) -> typing.Tuple[typing.Mapping[str, Tensor], Tensor]:
        try:
            label = torch.from_numpy((self.labels.iloc[label_index].values).astype(int)).float()
            index = self.labels.iloc[label_index].name
            # get modalities
            img_pa = self.imgs_pa[index, :, :]
            img_lat = self.imgs_lat[index, :, :]
            text_vec = self.get_vec(index)
            # transform images
            img_pa = self.transform_img(img_pa)
            img_lat = self.transform_img(img_lat)

            # if self.args.text_encoding == 'char':
            #     text_str = self.report_findings[index]
            #     if len(text_str) > self.args.len_sequence:
            #         text_str = text_str[:self.args.len_sequence]
            #     text_vec = text.one_hot_encode(self.args.len_sequence, self.args.alphabet, text_str.lower())
            #
            # elif self.args.text_encoding == 'word':
            #     text_vec = self.report_findings_dataset.__getitem__(index)
            # else:
            #     raise NotImplementedError(f'{self.args.text_encoding} has to be either char or word')

            sample = {'PA': img_pa, 'Lateral': img_lat, 'text': text_vec}
        except (IndexError, OSError):
            return None
        return sample, label

    def get_char_text_vec(self, index):
        text_str = self.report_findings[index]
        if len(text_str) > self.args.len_sequence:
            text_str = text_str[:self.args.len_sequence]
        return text.one_hot_encode(self.args.len_sequence, self.args.alphabet, text_str.lower())

    def get_word_text_vec(self, index):
        return self.report_findings_dataset.__getitem__(index)

    def __len__(self):
        return self.labels.shape[0]

    def get_text_str(self, index):
        return self.y[index]

    def get_report_findings_dataset(self, dir_dataset):

        self.report_findings_dataset = MimicSentences(max_squence_len=self.args.len_sequence, data_dir=dir_dataset,
                                                      findings=self.report_findings, split=self.split, transform=True,
                                                      min_occ=self.args.word_min_occ)
        assert len(self.report_findings_dataset) == len(self.report_findings), \
            'report findings dataset must have the same length than the report findings dataframe'
        self.args.vocab_size = self.report_findings_dataset.vocab_size

    def _verify_dataset(self):
        """
        Need to remove all cases where the labels have 3 classes.
        The 3rd class (-1) represents "uncertain" and can be removed from the dataset.
        """
        labels = self.labels.values
        assert len(np.unique(labels)) == 2, \
            'labels should contain 2 classes, might need to remove -1 labels'
        assert self.imgs_pa.shape[0] == self.imgs_lat.shape[0] == len(
            self.report_findings), f'all modalities must have the same length. len(imgs_pa): {self.imgs_pa.shape[0]},' \
                                   f' len(imgs_lat): {self.imgs_lat.shape[0]},' \
                                   f' len(report_findings): {len(self.report_findings)}'


class MimicText(Dataset):
    """
    Custom Dataset for loading the uni-modal mimic text data
    """

    def __init__(self, args, str_labels, split: str):
        """
        split: string, either train, eval or test
        """
        self.args = args
        self.split = split
        dir_dataset = os.path.join(args.dir_data, f'files_small_{args.img_size}')
        fn_findings = os.path.join(dir_dataset, split + '_findings.csv')
        fn_labels = os.path.join(dir_dataset, split + '_labels.csv')

        self.labels = pd.read_csv(fn_labels)[str_labels].fillna(0)

        self.report_findings = pd.read_csv(fn_findings)['findings']
        # need to filter out labels that contain the label "-1"
        self.labels = filter_labels(self.labels, args.undersample_dataset)

        if self.args.text_encoding == 'char':
            self.args.alphabet = get_alphabet()
            args.num_features = len(self.args.alphabet)
        else:
            # if word_encoding == word, need dataset for report_findings that contains the encodings.
            self.get_report_findings_dataset(dir_dataset)

    def __getitem__(self, label_index):
        try:
            label = torch.from_numpy((self.labels.iloc[label_index].values).astype(int)).float()
            index = self.labels.iloc[label_index].name

            if self.args.text_encoding == 'char':
                text_str = self.report_findings[index]
                if len(text_str) > self.args.len_sequence:
                    text_str = text_str[:self.args.len_sequence]
                text_vec = text.one_hot_encode(self.args.len_sequence, self.args.alphabet, text_str)
            elif self.args.text_encoding == 'word':
                text_vec = self.report_findings_dataset.__getitem__(index)
            else:
                raise NotImplementedError(f'{self.args.text_encoding} has to be either char or word')

            sample = {'text': text_vec}
        except (IndexError, OSError):
            return None
        return sample, label

    def __len__(self):
        return self.labels.shape[0]

    def get_text_str(self, index):
        return self.y[index]

    def get_report_findings_dataset(self, dir_dataset):

        self.report_findings_dataset = MimicSentences(max_squence_len=self.args.len_sequence, data_dir=dir_dataset,
                                                      findings=self.report_findings, split=self.split,
                                                      transform=True, min_occ=self.args.word_min_occ)
        assert len(self.report_findings_dataset) == len(self.report_findings), \
            'report findings dataset must have the same length than the report findings dataframe'
        self.args.vocab_size = self.report_findings_dataset.vocab_size


class OrderedCounter(Counter, OrderedDict):
    """
    Counter that remembers the order elements are first encountered.
    """

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def to_tensor(data):
    return torch.Tensor(data)


class MimicSentences(Dataset):
    """
    Modified version of https://github.com/iffsid/mmvae/blob/public/src/datasets.py
    Word encoding for mimic report findings
    """

    def __init__(self, max_squence_len: int, data_dir: str, findings: pd.DataFrame, split: str, transform=False,
                 min_occ: int = 3):
        """split: 'train', 'val' or 'test' """

        super().__init__()
        self.split = split
        self.data_dir = data_dir
        # self.args = args
        self.max_sequence_length = max_squence_len
        self.min_occ = min_occ
        self.transform = to_tensor if transform else None
        self.findings = findings
        self.gen_dir = os.path.join(self.data_dir, "oc:{}_msl:{}".
                                    format(self.min_occ, self.max_sequence_length))

        self.raw_data_path = os.path.join(data_dir, split + '_findings.csv')

        os.makedirs(self.gen_dir, exist_ok=True)
        self.data_file = 'mimic.{}.s{}'.format(split, self.max_sequence_length)
        self.vocab_file = 'mimic.vocab'

        if not os.path.exists(os.path.join(self.gen_dir, self.data_file)):
            print("Data file not found for {} split at {}. Creating new... (this may take a while)".
                  format(split.upper(), os.path.join(self.gen_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Returns tensors/list of length len_sentence
        """
        sent = self.data[str(idx)]['idx']
        if self.transform is not None:
            sent = self.transform(sent)
        return sent

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def _load_data(self, vocab=True):
        with open(os.path.join(self.gen_dir, self.data_file), 'rb') as file:
            self.data = json.load(file)

        if vocab:
            self._load_vocab()

    def _load_vocab(self):
        if not os.path.exists(os.path.join(self.gen_dir, self.vocab_file)):
            self._create_vocab()
        with open(os.path.join(self.gen_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)
        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):
        if self.split == 'train' and not os.path.exists(os.path.join(self.gen_dir, self.vocab_file)):
            self._create_vocab()
        else:
            self._load_vocab()

        sentences = self._tokenize_raw_data()

        data = defaultdict(dict)
        pad_count = 0

        for i, line in enumerate(sentences):
            words = word_tokenize(line.lower())
            # words = word_tokenize(line)
            tok = words[:self.max_sequence_length - 1]
            tok = tok + ['<eos>']
            length = len(tok)
            if self.max_sequence_length > length:
                tok.extend(['<pad>'] * (self.max_sequence_length - length))
                pad_count += 1
            idx = [self.w2i.get(w, self.w2i['<exc>']) for w in tok]

            id = len(data)
            data[id]['tok'] = tok
            data[id]['idx'] = idx
            data[id]['length'] = length

        print("{} out of {} sentences are truncated with max sentence length {}.".
              format(len(sentences) - pad_count, len(sentences), self.max_sequence_length))
        with io.open(os.path.join(self.gen_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _tokenize_raw_data(self) -> List:
        """
        Creates a list of all the findings
        """
        report_findings = self.findings
        return [sentence for sentence in report_findings]

    def _create_vocab(self):

        assert self.split == 'train', "Vocabulary can only be created for training file."

        sentences = self._tokenize_raw_data()

        occ_register = OrderedCounter()
        w2i = {}
        i2w = {}

        special_tokens = ['<exc>', '<pad>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        texts = []
        unq_words = []

        for i, line in enumerate(sentences):
            # words = word_tokenize(line)
            words = word_tokenize(line.lower())
            occ_register.update(words)
            texts.append(words)

        for w, occ in occ_register.items():
            if occ > self.min_occ and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
            else:
                unq_words.append(w)

        assert len(w2i) == len(i2w)

        print("Vocablurary of {} keys created, {} words are excluded (occurrence <= {})."
              .format(len(w2i), len(unq_words), self.min_occ))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.gen_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        with open(os.path.join(self.gen_dir, 'mimic.unique'), 'wb') as unq_file:
            pickle.dump(np.array(unq_words), unq_file)

        with open(os.path.join(self.gen_dir, 'mimic.all'), 'wb') as a_file:
            pickle.dump(occ_register, a_file)

        self._load_vocab()


class Mimic_testing(Dataset):
    """
    Custom Dataset for the testsuite of the training workflow
    """

    def __init__(self, flags, classifier_training=False):
        # used to indicate
        self.classifier_training = classifier_training
        self.vocab_size = 3517
        self.flags = flags
        self.report_findings_dataset = Report_findings_dataset_test(self.vocab_size)
        if self.flags.text_encoding == 'char':
            self.flags.alphabet = get_alphabet()
            self.flags.num_features = len(self.flags.alphabet)

    def __getitem__(self, index):
        sample = self.get_images() if not self.flags.only_text_modality else {}
        if self.flags.text_encoding == 'word':
            sample['text'] = torch.randint(0, 3517, (1, self.flags.len_sequence)).view(self.flags.len_sequence).float()
        elif self.flags.text_encoding == 'char':
            sample['text'] = torch.rand(self.flags.len_sequence, self.flags.num_features).float()

        label = torch.tensor([random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)]).float()

        return sample, label

    def get_images(self) -> dict:
        img_size = (self.flags.img_size, self.flags.img_size)

        return {'PA': torch.rand(1, *img_size).float(),
                'Lateral': torch.rand(1, *img_size).float()}

    def __len__(self) -> int:
        return 2 * self.flags.batch_size

    def get_text_str(self, index):
        return self.y[index]


class Report_findings_dataset_test(Dataset):
    def __init__(self, vocab_size: int):
        self.i2w = {}
        for i in range(vocab_size + 1):
            self.i2w[str(i)] = 'w'  # arbitrary letter
