import typing
from typing import Mapping

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from mimic import log
from mimic.networks.VAEtrimodalMimic import VAEtrimodalMimic
from mimic.utils.experiment import MimicExperiment
from mimic.utils.utils import dict_to_device
from mimic.utils.utils import init_twolevel_nested_dict
from mimic.utils.utils import stdout_if_verbose


def train_clf_lr_all_subsets(exp: MimicExperiment):
    args = exp.flags
    mm_vae = exp.mm_vae
    mm_vae.eval()
    mm_vae: VAEtrimodalMimic
    subsets = exp.subsets
    if '' in subsets:
        del subsets['']
    d_loader = DataLoader(exp.dataset_train, batch_size=exp.flags.batch_size,
                          shuffle=True,
                          num_workers=args.dataloader_workers // args.world_size if args.distributed
                          else args.dataloader_workers,
                          drop_last=True)
    if exp.flags.steps_per_training_epoch > 0:
        training_steps = exp.flags.steps_per_training_epoch
    else:
        training_steps = len(d_loader)

    bs = exp.flags.batch_size
    class_dim = exp.flags.class_dim
    n_samples = int(exp.dataset_train.__len__())
    data_train = {
        s_key: np.zeros((n_samples, class_dim))
        for s_key in subsets
    }

    all_labels = np.zeros((n_samples, len(exp.labels)))
    log.info(f"Creating {training_steps} batches of the latent representations for the classifier.")
    for it, (batch_d, batch_l) in tqdm(enumerate(d_loader), total=training_steps, postfix='creating_train_lr'):
        """
        Constructs the training set (labels and inferred subsets) for the classifier training.
        """
        if it > training_steps \
                and any(len(np.unique(all_labels[:, l])) > 1 for l in range(all_labels.shape[-1])) \
                and it > 150:
            # labels need at least 2 classes to train the clf
            break

        batch_d = {k: v.to(exp.flags.device) for k, v in batch_d.items()}
        inferred = mm_vae.module.inference(batch_d) if args.distributed else mm_vae.inference(batch_d)

        lr_subsets = inferred['subsets']
        all_labels[(it * bs):((it + 1) * bs), :] = np.reshape(batch_l, (bs, len(exp.labels)))
        for key in lr_subsets:
            data_train[key][(it * bs):((it + 1) * bs), :] = lr_subsets[key][0].cpu().data.numpy()

    n_train_samples = exp.flags.num_training_samples_lr
    # get random labels such that it contains both classes
    labels, rand_ind_train = get_random_labels(n_samples, n_train_samples, all_labels)
    for s_key in subsets:
        d = data_train[s_key]
        data_train[s_key] = d[rand_ind_train, :]
    return train_clf_lr(exp, data_train, labels)


def get_random_labels(n_samples, n_train_samples, all_labels, max_tries=1000):
    """
    The classifier needs labels from both classes to train. This function resamples "all_labels"
    until it contains examples from both classes
    """
    assert any(len(np.unique(all_labels[:, l])) > 1 for l in range(all_labels.shape[-1])), \
        'The labels must contain at least two classes to train the classifier'
    rand_ind_train = np.random.randint(n_samples, size=n_train_samples)
    labels = all_labels[rand_ind_train, :]
    tries = 1
    while any(len(np.unique(labels[:, l])) <= 1 for l in range(labels.shape[-1])):
        rand_ind_train = np.random.randint(n_samples, size=n_train_samples)
        labels = all_labels[rand_ind_train, :]
        tries += 1
        assert max_tries >= tries, f'Could not get sample containing both classes to train ' \
                                   f'the classifier in {tries} tries. Might need to increase batch_size'
    return labels, rand_ind_train


def test_clf_lr_all_subsets(epoch: int, clf_lr, exp) -> typing.Mapping[str, typing.Mapping[str, float]]:
    """
    Test the classifiers that were trained on latent representations.

    """
    args = exp.flags
    mm_vae = exp.mm_vae
    mm_vae.eval()
    subsets = exp.subsets
    if '' in subsets:
        del subsets['']
    labels = exp.labels

    lr_eval = init_twolevel_nested_dict(exp.labels, subsets, [])

    d_loader = DataLoader(exp.dataset_test, batch_size=exp.flags.batch_size,
                          shuffle=True,
                          num_workers=exp.flags.dataloader_workers, drop_last=True)

    if exp.flags.steps_per_training_epoch > 0:
        training_steps = exp.flags.steps_per_training_epoch
    else:
        training_steps = len(d_loader)
    log.info(f'Creating {training_steps} batches of latent representations for classifier testing '
             f'with a batch_size of {exp.flags.batch_size}.')

    clf_predictions = init_twolevel_nested_dict(exp.labels, subsets, [], copy_init_val=True)
    batch_labels = torch.Tensor()

    for iteration, (batch_d, batch_l) in enumerate(d_loader):
        if iteration > training_steps:
            break
        batch_labels = torch.cat((batch_labels, batch_l), 0)

        batch_d = dict_to_device(batch_d, exp.flags.device)

        inferred = mm_vae.module.inference(batch_d) if args.distributed else mm_vae.inference(batch_d)
        lr_subsets = inferred['subsets']
        data_test = {key: lr_subsets[key][0].cpu().data.numpy() for key in lr_subsets}

        clf_predictions_batch = classify_latent_representations(exp, clf_lr, data_test)
        clf_predictions_batch: Mapping[str, Mapping[str, np.array]]

        for label in labels:
            for subset in subsets:
                clf_predictions[label][subset].append(clf_predictions_batch[label][subset])

    for l_idx, l_key in enumerate(labels):
        for s_key in subsets:
            lr_eval[l_key][s_key]: float = exp.eval_metric(batch_labels[:, l_idx],
                                                           np.array(clf_predictions[l_key][s_key]).ravel())
    return lr_eval


def classify_latent_representations(exp, clf_lr: Mapping[str, Mapping[str, LogisticRegression]], data) \
        -> Mapping[str, Mapping[str, np.array]]:
    """
    Returns the classification of each subset of the powerset for each label.
    """
    clf_predictions = {}
    for label_str in exp.labels:
        stdout_if_verbose(verbose=exp.flags.verbose,
                          message=f'classifying the latent representations of label {label_str}', min_level=10)

        clf_pred_subset = {}

        for s_key, data_rep in data.items():
            # get the classifier for the subset
            clf_lr_rep = clf_lr[label_str][s_key]

            clf_pred_subset[s_key] = clf_lr_rep.predict(data_rep)

        clf_predictions[label_str] = clf_pred_subset
    return clf_predictions


def train_clf_lr(exp, data, labels):
    labels = np.reshape(labels, (labels.shape[0], len(exp.labels)))
    clf_lr_labels = {}
    for l, label_str in enumerate(exp.labels):
        stdout_if_verbose(message=f"Training lr classifier on label {label_str}", min_level=1,
                          verbose=exp.flags.verbose)
        gt = labels[:, l]
        clf_lr_reps = {}
        for s_key in data.keys():
            data_rep = data[s_key]
            clf_lr_s = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000)
            if exp.flags.dataset == 'testing':
                # when using the testing dataset, the vae data_rep might contain nans. Replace them for testing purposes
                clf_lr_s.fit(np.nan_to_num(data_rep), gt.ravel())
            else:
                clf_lr_s.fit(data_rep, gt.ravel())
            clf_lr_reps[s_key] = clf_lr_s
        clf_lr_labels[label_str] = clf_lr_reps
    return clf_lr_labels
