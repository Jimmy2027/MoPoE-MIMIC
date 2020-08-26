import os

import random
import numpy as np 
from itertools import chain, combinations

import torch
from torchvision import transforms
import torch.optim as optim
import PIL.Image as Image
from sklearn.metrics import accuracy_score

#from utils.BaseExperiment import BaseExperiment
from PIL import ImageFont

from modalities.CMNIST import CMNIST

# from mnistsvhntext.SVHNMNISTDataset import SVHNMNIST
from mmnist.MMNISTDataset import MMNISTDataset

# from mmnist.networks.VAEtrimodalSVHNMNIST import VAEtrimodalSVHNMNIST
from mmnist.networks.VAEMMNIST import VAEMMNIST

# from mmnist.networks.ConvNetworkImgClfSVHN import ClfImgSVHN
# from mmnist.networks.ConvNetworkTextClf import ClfText as ClfText
from mmnist.networks.ConvNetworkImgClfCMNIST import ClfImg as ClfImgCMNIST

from mmnist.networks.ConvNetworksImgCMNIST import EncoderImg, DecoderImg
# from mmnist.networks.ConvNetworksImgSVHN import EncoderSVHN, DecoderSVHN
# from mmnist.networks.ConvNetworksTextMNIST import EncoderText, DecoderText


class MMNISTExperiment():
    def __init__(self, flags, alphabet):
        self.flags = flags
        self.name = flags.name
        self.dataset_name = flags.dataset
        self.num_modalities = flags.num_mods
        self.plot_img_size = torch.Size((3, 28, 28))
        self.font = ImageFont.truetype('FreeSerif.ttf', 38)
        self.alphabet = alphabet
        self.flags.num_features = len(alphabet)

        self.modalities = self.set_modalities()
        self.subsets = self.set_subsets()
        self.dataset_train = None
        self.dataset_test = None
        self.set_dataset()

        self.mm_vae = self.set_model()
        self.clfs = self.set_clfs()
        self.optimizer = None
        self.rec_weights = self.set_rec_weights()
        self.style_weights = self.set_style_weights()

        self.test_samples = self.get_test_samples()
        self.eval_metric = accuracy_score; 
        self.paths_fid = self.set_paths_fid()

    def set_model(self):
        model = VAEMMNIST(self.flags, self.modalities, self.subsets)
        model = model.to(self.flags.device)
        return model

    def set_modalities(self):
        mods = [CMNIST(EncoderImg(self.flags), DecoderImg(self.flags), name="m%d" % m) for m in range(self.num_modalities)]
        mods_dict = {m.name: m for m in mods}
        return mods_dict


    def set_subsets(self):
        num_mods = len(list(self.modalities.keys()))

        """
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)
        (1,2,3)
        """
        xs = list(self.modalities)
        # note we return an iterator rather than a list
        subsets_list = chain.from_iterable(combinations(xs, n) for n in
                                          range(len(xs)+1))
        subsets = dict()
        for k, mod_names in enumerate(subsets_list):
            mods = []
            for l, mod_name in enumerate(mod_names):
                mods.append(self.modalities[mod_name])
            key = '_'.join(mod_names)
            subsets[key] = mods
        return subsets

    def set_dataset(self):
        # transform_mnist = self.get_transform_mnist()
        # transform_svhn = self.get_transform_svhn()
        # transforms = [transform_mnist, transform_svhn]
        # transforms = [self.get_transform_mmnist() for _ in range(self.num_modalities)]
        # train = MMNISTDataset(self.flags,
        #                   self.alphabet,
        #                   train=True,
        #                   transform=transforms)
        # test = MMNISTDataset(self.flags,
        #                  self.alphabet,
        #                  train=False,
        #                  transform=transforms)
        transform = transforms.Compose([transforms.ToTensor()])
        train = MMNISTDataset(self.flags.unimodal_datapaths_train, transform=transform)
        test = MMNISTDataset(self.flags.unimodal_datapaths_test, transform=transform)
        self.dataset_train = train
        self.dataset_test = test

    def set_clfs(self):
        clfs = {"m%d" % m: None for m in range(self.num_modalities)}
        if self.flags.use_clf:
            for m in range(self.num_modalities):
                model_clf = ClfImgCMNIST()
                model_clf.load_state_dict(torch.load(os.path.join(self.flags.dir_clf, "pretrained_img_to_digit_clf_m%d" % m)))
                model_clf = model_clf.to(self.flags.device)
                clfs["m%d" % m] = model_clf
        return clfs

    def set_optimizer(self):
        # optimizer definition
        params = []
        for model in [*self.mm_vae.encoders, *self.mm_vae.decoders]:
            for p in model.parameters():
                params.append(p)
        optimizer = optim.Adam(params, lr=self.flags.initial_learning_rate, betas=(self.flags.beta_1,
                               self.flags.beta_2))
        self.optimizer = optimizer

    def set_rec_weights(self):
        rec_weights = dict()
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key]
            numel_mod = mod.data_size.numel()
            rec_weights[mod.name] = 1.0
        return rec_weights

    def set_style_weights(self):
        weights = {"m%d" % m: self.flags.beta_style for m in range(self.num_modalities)}
        return weights

    def get_transform_mmnist(self):
        # transform_mnist = transforms.Compose([transforms.ToTensor(),
        #                                       transforms.ToPILImage(),
        #                                       transforms.Resize(size=(28, 28), interpolation=Image.BICUBIC),
        #                                       transforms.ToTensor()])
        transform_mnist = transforms.Compose([transforms.ToTensor()])
        return transform_mnist

    def get_test_samples(self, num_images=10):
        n_test = len(self.dataset_test)
        samples = []
        for i in range(num_images):
            while True:
                ix = random.randint(0, n_test-1)
                sample, target = self.dataset_test[ix]
                if target == i:
                    for k, key in enumerate(sample):
                        sample[key] = sample[key].to(self.flags.device)
                    samples.append(sample)
                    break
        return samples

    def mean_eval_metric(self, values):
        return np.mean(np.array(values))

    def set_paths_fid(self):
        dir_real = os.path.join(self.flags.dir_gen_eval_fid, 'real')
        dir_random = os.path.join(self.flags.dir_gen_eval_fid, 'random')
        paths = {'real': dir_real,
                 'random': dir_random}
        dir_cond = self.flags.dir_gen_eval_fid
        for k, name in enumerate(self.subsets):
            paths[name] = os.path.join(dir_cond, name)
        print(paths.keys())
        return paths
