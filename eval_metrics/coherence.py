
import sys
import os

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.save_samples import save_generated_samples_singlegroup


def classify_cond_gen_samples(exp, epoch, labels, cond_samples):
    clfs = exp.clfs;
    evals = dict();
    for key in clfs.keys():
        if key in cond_samples.keys():
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


def test_generation(epoch, exp):
    mods = exp.modalities;
    mm_vae = exp.mm_vae;

    dict_mods = dict();
    for m, m_key in enumerate(mods.keys()):
        dict_mods[m_key] = [];

    d_loader = DataLoader(exp.dataset_test,
                          batch_size=exp.flags.batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True);

    num_batches_epoch = int(exp.dataset_test.__len__() /float(exp.flags.batch_size));
    print('num batches test:' + str(num_batches_epoch))
    gen_perf = dict();
    for iteration, batch in enumerate(d_loader):
        batch_d = batch[0];
        batch_l = batch[1];
        rand_gen = mm_vae.generate();
        coherence_random = calculate_coherence(exp, rand_gen);
        if 'random' not in gen_perf.keys():
            gen_perf['random'] = [];
        gen_perf['random'].append(coherence_random);

        if (exp.flags.batch_size*iteration) < exp.flags.num_samples_fid:
            save_generated_samples_singlegroup(exp, iteration,
                                               'random',
                                               rand_gen);
            save_generated_samples_singlegroup(exp, iteration,
                                               'real',
                                               batch_d);

        for k, m_key in enumerate(batch_d.keys()):
            batch_d[m_key] = batch_d[m_key].to(exp.flags.device);
        inferred = mm_vae.inference(batch_d);
        lr_subsets = inferred['subsets'];
        cg = mm_vae.cond_generation(lr_subsets)
        for k, s_key in enumerate(cg.keys()):
            if s_key not in gen_perf.keys():
                gen_perf[s_key] = dict_mods.copy();

            clf_cg = classify_cond_gen_samples(exp, epoch,
                                               batch_l,
                                               cg[s_key]);
            for m, modname in enumerate(clf_cg.keys()):
                gen_perf[s_key][modname].append(clf_cg[modname]);
            if (exp.flags.batch_size*iteration) < exp.flags.num_samples_fid:
                save_generated_samples_singlegroup(exp, iteration,
                                                   s_key,
                                                   cg[s_key]);
    for k, key in enumerate(gen_perf.keys()):
        if key != 'random':
            for m, mod in enumerate(gen_perf[key].keys()):
                gen_perf[key][mod] = exp.mean_eval_metric(gen_perf[key][mod]);
        else:
            gen_perf['random'] = exp.mean_eval_metric(gen_perf['random']);
    return gen_perf;



