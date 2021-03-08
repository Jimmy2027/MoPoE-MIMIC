import torch
import torch.nn.functional as F

from sklearn.metrics import roc_curve, auc
import torch.nn as nn


def calc_auc(gt, pred_proba):
    fpr, tpr, thresholds = roc_curve(gt, pred_proba)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc;


def mse_loss(input, target):
    return torch.sum((input - target).pow(2)) / input.data.nelement()


def loss_img_mse(input, target, norm_value=None):
    reconstruct_error_img = F.mse_loss(input, target, reduction='sum');
    if norm_value is not None:
        reconstruct_error_img /= norm_value;
    return reconstruct_error_img;


def log_prob_img(output_dist, target, norm_value):
    log_prob = output_dist.log_prob(target).sum();
    mean_val_logprob = log_prob / norm_value;
    return mean_val_logprob;


def log_prob_text(output_dist, target, norm_value):
    log_prob = output_dist.log_prob(target).sum();
    mean_val_logprob = log_prob / norm_value;
    return mean_val_logprob;


def loss_img_bce(input, target, norm_value=None):
    reconstruct_error_img = F.binary_cross_entropy_with_logits(input, target, reduction='sum');
    if norm_value is not None:
        reconstruct_error_img /= norm_value;
    return reconstruct_error_img;


def loss_text(input, target, norm_value=None):
    reconstruct_error_text = F.binary_cross_entropy_with_logits(input, target, reduction='sum');
    if norm_value is not None:
        reconstruct_error_text /= norm_value;
    return reconstruct_error_text


def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch

    Taken from https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183 then changed.
    """

    smooth = 0.0001

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - 2. * (intersection + smooth) / (A_sum + B_sum + 2 * smooth)


def get_clf_loss(which_loss: str):
    if which_loss == 'binary_crossentropy':
        return nn.BCELoss()
    elif which_loss == 'dice':
        return dice_loss
    else:
        raise NotImplementedError(f'{which_loss} is not implemented yet')


def l1_loss(input, target):
    return torch.sum(torch.abs(input - target)) / input.data.nelement()
