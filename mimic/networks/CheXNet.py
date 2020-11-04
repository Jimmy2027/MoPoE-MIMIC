# encoding: utf-8

"""
The main CheXNet model implementation.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from mimic.dataio.MimicDataset import Mimic
from mimic.utils.experiment import MimicExperiment
from mimic.utils.filehandling import expand_paths, get_config_path
from mimic.utils.flags import parser

CKPT_PATH = 'networks/model.pth.tar'
N_CLASSES = 3
CLASS_NAMES = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']
DATA_DIR = './ChestX-ray14/images'
TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'
BATCH_SIZE = 5


def main():

    cudnn.benchmark = True

    # initialize and load the model
    model = CheXNet(N_CLASSES).cuda()
    # todo pytorch recommends to use DistributedDataParallel instead of DataParallel
    model = torch.nn.DataParallel(model).cuda()

    if False:
    # if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    config_path = get_config_path()
    FLAGS = parser.parse_args()
    with open(config_path, 'rt') as json_file:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(json_file))
        FLAGS = parser.parse_args(namespace=t_args)
    FLAGS = expand_paths(FLAGS)
    print(FLAGS.dir_data)
    FLAGS.str_experiment = 'temp'
    FLAGS.device = 'cuda'
    FLAGS.dir_gen_eval_fid = ''
    FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content, FLAGS.div_weight_m1_content,
                              FLAGS.div_weight_m2_content, FLAGS.div_weight_m3_content]
    FLAGS.text_encoding = 'word'
    FLAGS.img_size = 128
    mimic_experiment = MimicExperiment(flags=FLAGS)
    mimic_test = Mimic(FLAGS, mimic_experiment.labels, split='eval', transform_img=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda
        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda
        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ]))

    test_loader = DataLoader(dataset=mimic_test, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    # switch to evaluate mode
    model.eval()

    for i, (inp, target) in enumerate(test_loader):
        inp = inp['PA']
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
        output = model(input_var)
        output_mean = output.view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output_mean.data), 0)

    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


class CheXNet(nn.Module):
    """Taken from https://github.com/arnoweng/CheXNet/blob/master/model.py
    The architecture of this model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(CheXNet, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

if __name__ == '__main__':
    main()