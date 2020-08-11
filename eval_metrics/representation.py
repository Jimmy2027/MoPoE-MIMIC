import sys
import os

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def train_clf_lr_all_subsets(exp):
    mm_vae = exp.mm_vae;
    mm_vae.eval();

    d_loader = DataLoader(exp.dataset_train, batch_size=exp.flags.batch_size,
                        shuffle=True,
                        num_workers=8, drop_last=True);

    num_batches_epoch = int(exp.dataset_train.__len__() /float(exp.flags.batch_size));
    data_train = dict();
    for iteration, batch in enumerate(d_loader):
        if iteration == num_batches_epoch-1:
            batch_d = batch[0];
            batch_l = batch[1];
            for k, m_key in enumerate(batch_d.keys()):
                batch_d[m_key] = batch_d[m_key].to(exp.flags.device);
            inferred = mm_vae.inference(batch_d);
            lr_subsets = inferred['subsets'];
            for k, key in enumerate(lr_subsets.keys()):
                data_train[key] = lr_subsets[key][0].cpu().data.numpy();
    clf_lr = train_clf_lr(exp, data_train, batch_l);
    return clf_lr;
    

def test_clf_lr_all_subsets(epoch, clf_lr, exp):
    mm_vae = exp.mm_vae;
    mm_vae.eval();

    lr_eval = dict();
    for k, key in enumerate(clf_lr.keys()):
        lr_eval[key] = [];

    d_loader = DataLoader(exp.dataset_test, batch_size=exp.flags.batch_size,
                        shuffle=True,
                        num_workers=8, drop_last=True);

    num_batches_epoch = int(exp.dataset_test.__len__() /float(exp.flags.batch_size));
    for iteration, batch in enumerate(d_loader):
        batch_d = batch[0];
        batch_l = batch[1];
        for k, m_key in enumerate(batch_d.keys()):
            batch_d[m_key] = batch_d[m_key].to(exp.flags.device);
        inferred = mm_vae.inference(batch_d);
        lr_subsets = inferred['subsets'];
        data_test = dict();
        for k, key in enumerate(lr_subsets.keys()):
            data_test[key] = lr_subsets[key][0].cpu().data.numpy();
        evals = classify_latent_representations(exp,
                                                epoch,
                                                clf_lr,
                                                data_test,
                                                batch_l);
        for k, key in enumerate(lr_subsets.keys()):
            lr_eval[key].append(evals[key]);
    for k, key in enumerate(lr_eval.keys()):
        lr_eval[key] = exp.mean_eval_metric(lr_eval[key]);
    return lr_eval;


def classify_latent_representations(exp, epoch, clf_lr, data, labels):
    evals = dict()
    for key in clf_lr:
        data_rep = data[key];
        clf_lr_rep = clf_lr[key];
        y_pred_rep = clf_lr_rep.predict(data_rep);
        eval_rep = exp.eval_metric(labels.cpu().data.numpy().ravel(),
                                   y_pred_rep.ravel());
        evals[key] = eval_rep;
    return evals;


def train_clf_lr(exp, data, labels):
    clf_lr = dict();
    for k, key in enumerate(data.keys()):
        data_rep = data[key];
        clf_lr_rep = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000);
        clf_lr_rep.fit(data_rep, labels.cpu().data.numpy().ravel());
        clf_lr[key] = clf_lr_rep;
    return clf_lr;


def classify_cond_gen_samples(exp, epoch, labels, cond_samples):
    clfs = exp.clfs;
    evals = dict();
    for key in clfs:
        if key in cond_samples:
            mod_cond_gen = cond_samples[key];
            mod_clf = clfs[key];
            attr_hat = mod_clf(mod_cond_gen);
            pred = np.argmax(attr_hat.cpu().data.numpy(), axis=1).astype(int);
            evals[key] = exp.eval_metric(labels, pred);
        else:
            print(str(key) + 'not existing in cond_gen_samples');
    return evals;


def calculate_coherence(exp, samples):
    clfs = exp.clfs;
    mods = exp.modalities;
    pred_mods = np.zeros((len(mods.keys()), exp.flags.batch_size))
    for k, m_key in enumerate(mods.keys()):
        mod = mods[m_key];
        clf_mod = clfs[mod.name];
        samples_mod = samples[mod.name];
        attr_mod = clf_mod(samples_mod);
        output_prob_mod = attr_mod.cpu().data.numpy();
        pred_mod = np.argmax(output_prob_mod, axis=1).astype(int);
        pred_mods[k,:] = pred_mod;

    for k, m_key in enumerate(mods.keys()):
        if k > 0:
            coh = (coh == pred_mods[k-1,:]);
        else:
            coh = (pred_mods[k,:] == pred_mods[k,:])

    coherence = np.sum(coh) / np.sum(np.ones(coh.shape));
    return coherence;

