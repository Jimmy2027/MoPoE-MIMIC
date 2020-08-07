import sys
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from utils import utils
from utils import plot

from mnistsvhntext.constants import indices


def generate_random_samples_plots(exp, epoch):
    model = exp.mm_vae;
    mods = exp.modalities;
    num_samples = 100;
    random_samples = model.generate(num_samples)
    random_plots = dict();
    for k, m_key_in in enumerate(mods.keys()):
        mod = mods[m_key_in];
        samples_mod = random_samples[m_key_in];
        rec = torch.zeros(exp.plot_img_size,
                          dtype=torch.float32).repeat(num_samples,1,1,1);
        for l in range(0, num_samples):
            rand_plot = mod.plot_data(samples_mod[l]);
            rec[l, :, :, :] = rand_plot;
        random_plots[m_key_in] = rec;

    for k, m_key in enumerate(mods.keys()):
        fn = os.path.join(exp.flags.dir_random_samples, 'random_epoch_' +
                             str(epoch).zfill(4) + '_' + m_key + '.png');
        mod_plot = random_plots[m_key];
        p = plot.create_fig(fn, mod_plot, 10);
        random_plots[m_key] = p;
    return random_plots;


def generate_swapping_plot(exp, epoch):
    model = exp.mm_vae;
    mods = exp.modalities;
    samples = exp.test_samples;
    swap_plots = dict();
    for k, m_key_in in enumerate(mods.keys()):
        mod_in = mods[m_key_in];
        for l, m_key_out in enumerate(mods.keys()):
            mod_out = mods[m_key_out];
            rec = Variable(torch.zeros([121, exp.plot_img_size], dtype=torch.float32));
            rec = rec.to(exp.flags.device);
            for i in range(len(samples)):
                c_sample_in = mod_in.plot_data(samples[i][mod_in.name]);
                s_sample_out = mod_out.plot_data(samples[i][mod_out.name]);
                rec[i+1, :, :, :] = c_sample_in;
                rec[(i + 1) * 11, :, :, :] = s_sample_out;
            # style transfer
            for i in range(len(samples)):
                for j in range(len(samples)):
                    l_style = model.inference(samples[i][mod_out.name])
                    l_content = model.inference(samples[j][mod_in.name])

                    s_emb = utils.reparameterize(l_style[0], l_style[1]);
                    c_emb = utils.reparameterize(l_content[0], l_content[1]);
                    style_emb = {mod_out.name: s_emb}
                    emb_swap = {'content': c_emb, 'style': style_emb};
                    swap_sample = model.generate_from_latents(emb_swap);
                    swap_out = mod_out.plot_data(swap_sample[mod_out.name]);
                    rec[(i+1) * 11 + (j+1), :, :, :] = swap_out;
                    fn_comb = (mod_in.name + '_to_' + mod_out.name + '_epoch_'
                               + str(epoch).zfill(4) + '.png');
                    fn = os.path.join(exp.flags.dir_swapping, fn_comb);
                    swap_plot = plot.create_fig(fn, rec, 11);
                    swap_plots[mod_in.name + '_' + mod_out.name] = swap_plot;
    return swap_plots;


def generate_conditional_fig_M(exp, epoch, M):
    model = exp.mm_vae;
    mods = exp.modalities;
    samples = exp.test_samples;
    subsets = exp.subsets;

    # get style from random sampling
    random_styles = model.get_random_styles(10);

    cond_plots = dict();
    for k, s_key in enumerate(subsets.keys()):
        subset = subsets[s_key];
        num_mod_s = len(subset);

        if num_mod_s == M:
            s_in = subset;
            for l, m_key_out in enumerate(mods.keys()):
                mod_out = mods[m_key_out];
                rec = torch.zeros(exp.plot_img_size,
                                  dtype=torch.float32).repeat(100 + M*10,1,1,1);
                for m, sample in enumerate(samples):
                    for n, mod_in in enumerate(s_in):
                        c_in = mod_in.plot_data(sample[mod_in.name]);
                        rec[m + n*10, :, :, :] = c_in;
                cond_plots[s_key + '__' + mod_out.name] = rec;

            # style transfer
            for i in range(len(samples)):
                for j in range(len(samples)):
                    i_batch = dict();
                    for o, mod in enumerate(s_in):
                        i_batch[mod.name] = samples[j][mod.name].unsqueeze(0);
                    latents = model.inference(i_batch)
                    c_in = latents['subsets'][s_key];
                    c_rep = utils.reparameterize(mu=c_in[0], logvar=c_in[1]);

                    style = dict();
                    for l, m_key_out in enumerate(mods.keys()):
                        mod_out = mods[m_key_out];
                        if exp.flags.factorized_representation:
                            style[mod_out.name] = random_styles[mod_out.name][i];
                        else:
                            style[mod_out.name] = None;
                    cond_mod_in = {'content': c_rep, 'style': style};
                    cond_gen_samples = model.generate_from_latents(cond_mod_in);

                    for l, m_key_out in enumerate(mods.keys()):
                        mod_out = mods[m_key_out];
                        rec = cond_plots[s_key + '__' + mod_out.name];
                        squeezed = cond_gen_samples[mod_out.name].squeeze(0);
                        p_out = mod_out.plot_data(squeezed);
                        rec[(i+M) * 10 + j, :, :, :] = p_out;
                        cond_plots[s_key + '__' + mod_out.name] = rec;

    for k, s_key_in in enumerate(subsets.keys()):
        subset = subsets[s_key_in];
        if len(subset) == M:
            s_in = subset;
            for l, m_key_out in enumerate(mods.keys()):
                mod_out = mods[m_key_out];
                rec = cond_plots[s_key_in + '__' + mod_out.name];
                fn_comb = (s_key_in + '_to_' + mod_out.name + '_epoch_' +
                           str(epoch).zfill(4) + '.png')
                fn_out = os.path.join(exp.flags.dir_cond_gen, fn_comb);
                plot_out = plot.create_fig(fn_out, rec, 10);
                cond_plots[s_key_in + '__' + mod_out.name] = plot_out;
    return cond_plots;


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

