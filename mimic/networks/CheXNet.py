# encoding: utf-8

"""
The main CheXNet model implementation.
"""

import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
from mimic.dataio.utils import get_densenet_transforms


class CheXNet(nn.Module):
    """Taken from https://github.com/arnoweng/CheXNet/blob/master/model.py
    The architecture of this model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, out_size, fixed_extractor=True):
        super(CheXNet, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        if fixed_extractor:
            for param in self.densenet121.parameters():
                param.requires_grad = False
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.densenet121(x)


class PretrainedDenseNet(nn.Module):
    def __init__(self, args):
        super(PretrainedDenseNet, self).__init__()
        original_model = torchvision.models.densenet121(pretrained=True)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.n_crops = args.n_crops
        if args.fixed_image_extractor:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.n_crops in [5, 10]:
            bs, n_crops, c, h, w = x.size()
        else:
            bs, c, h, w = x.size()
        imgs = torch.autograd.Variable(x.view(-1, c, h, w).cuda())
        x = self.features(imgs)
        # x.shape = [bs*n_crop, 1024, 8, 8]
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        return x


class DenseLayers(nn.Module):
    def __init__(self, args, nb_out=320):
        self.n_crops = args.n_crops
        self.batch_size = args.batch_size
        super().__init__()
        if args.n_crops:
            self.dens1 = nn.Linear(in_features=1024 * args.n_crops, out_features=1024)
            self.dens2 = nn.Linear(in_features=1024, out_features=512)
        else:
            self.dens1 = nn.Linear(in_features=1024, out_features=768)
            self.dens2 = nn.Linear(in_features=768, out_features=512)
        self.dens3 = nn.Linear(in_features=512, out_features=nb_out)

    def forward(self, x):
        if self.n_crops in [5, 10]:
            x = x.view(self.batch_size, 1024 * self.n_crops)
        x = self.dens1(x)
        x = nn.functional.selu(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.dens2(x)
        x = nn.functional.selu(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.dens3(x)
        return x


class DenseNetFeatureExtractor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pretrained_dense = PretrainedDenseNet(args)
        self.dense_layers = DenseLayers(args)
        self.transforms = get_densenet_transforms(args)

    def forward(self, x):
        x_tf = self.transform_batch(x)
        out = self.pretrained_dense(x_tf)
        out = self.dense_layers(out)
        out = out.unsqueeze(-1)
        return out

    def transform_batch(self, x):
        x_tf = torch.Tensor(x.shape[0], 3, *x.shape[2:])
        for idx, elem in enumerate(x):
            new = self.transforms(elem.cpu())
            x_tf[idx] = new
        x_tf.to(self.args.device)
        return x_tf


if __name__ == '__main__':
    model = CheXNet(3)
    for param in model.parameters():
        print(param)
        # param.requires_grad = False
