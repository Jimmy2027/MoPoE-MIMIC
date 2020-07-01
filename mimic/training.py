
import sys, os
import numpy as np
from itertools import cycle
import json
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from mimic.networks.VAEtrimodalMimic import VAEtrimodalMimic
from mimic.networks.ConvNetworkImgClf import ClfImg as ClfImg
from mimic.networks.ConvNetworkTextClf import ClfText as ClfText

from utils.loss import log_prob_img, log_prob_text
from divergence_measures.kl_div import calc_kl_divergence
from divergence_measures.mm_div import poe

from mimic.testing import generate_swapping_plot
from mimic.testing import generate_conditional_fig_1a
from mimic.testing import generate_conditional_fig_2a
from mimic.testing import generate_random_samples_plots
from mimic.testing import classify_rand_gen_samples
from mimic.testing import classify_cond_gen_samples
from mimic.testing import classify_latent_representations
from mimic.testing import train_clfs_latent_representation
from utils.test_functions import calculate_inception_features_for_gen_evaluation
from utils.test_functions import calculate_fid, calculate_fid_dict
from utils.test_functions import calculate_prd, calculate_prd_dict
from utils.test_functions import get_clf_activations
from utils.test_functions import load_inception_activations
from mimic.likelihood import calc_log_likelihood_batch
from mimic.constants import LABELS

from mimic.MimicDataset import Mimic
from utils.save_samples import save_generated_samples_singlegroup
from utils import utils

torch.multiprocessing.set_sharing_strategy('file_system')

# global variables
SEED = None 
SAMPLE1 = None
if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)


def get_10_mimic_samples(flags, dataset):
    samples = []
    while len(samples) < 10:
        ix = np.random.randint(0, dataset.__len__())
        data = dataset.__getitem__(ix)
        if data is not None:
            img_pa, img_lat, text, target = data;
            img_pa = img_pa.to(flags.device);
            img_lat = img_lat.to(flags.device);
            text = text.to(flags.device);
            samples.append((img_pa, img_lat, text, target))
    return samples


def run_epoch(epoch, vae_trimodal, optimizer, data, writer, alphabet, train=False, flags={},
              model_clf_m1=None, model_clf_m2=None, model_clf_m3 = None, clf_lr=None, step_logs=0):

    loader = cycle(DataLoader(data, batch_size=flags.batch_size, shuffle=True, num_workers=8, drop_last=True))

    # set up weights
    beta_style = flags.beta_style;
    beta_content = flags.beta_content;

    beta_m1_style = flags.beta_m1_style;
    beta_m2_style = flags.beta_m2_style;
    beta_m3_style = flags.beta_m3_style;

    #rec_weight_m1 = flags.rec_weight_m1;
    #rec_weight_m2 = flags.rec_weight_m2;
    #rec_weight_m3 = flags.rec_weight_m3;
    rec_weight_m1 = vae_trimodal.rec_w1;
    rec_weight_m2 = vae_trimodal.rec_w2;
    rec_weight_m3 = vae_trimodal.rec_w3;

    beta = flags.beta;
    rec_weight = 1.0;

    if not train:
        vae_trimodal.eval();
        ll_pa_pa = []; ll_pa_lateral = []; ll_pa_text = []; ll_pa_joint = [];
        ll_lateral_pa = []; ll_lateral_lateral = []; ll_lateral_text = []; ll_lateral_joint = [];
        ll_text_pa = []; ll_text_lateral = []; ll_text_text = []; ll_text_joint = [];
        ll_joint_pa = []; ll_joint_lateral = []; ll_joint_text = []; ll_joint_joint = [];
        ll_pl_text = []; ll_pl_joint = []; ll_pt_lateral = [];
        ll_pt_joint = []; ll_lt_pa = []; ll_lt_joint = [];
        cg_m1 = {'pa': [], 'lateral': [], 'text': []};
        cg_m2 = {'pa': [], 'lateral': [], 'text': []};
        cg_m3 = {'pa': [], 'lateral': [], 'text': []};
        cg_m1m2 = {'pa': [], 'lateral': [], 'text': []};
        cg_m1m3 = {'pa': [], 'lateral': [], 'text': []};
        cg_m2m3 = {'pa': [], 'lateral': [], 'text': []};
        cg_dp_m1m2 = {'pa': [], 'lateral': [], 'text': []};
        cg_dp_m1m3 = {'pa': [], 'lateral': [], 'text': []};
        cg_dp_m2m3 = {'pa': [], 'lateral': [], 'text': []};
        random_gen_acc = [];
        lr_m1_c = []; lr_m2_c = []; lr_m3_c = [];
        lr_m1_s = []; lr_m2_s = []; lr_m3_s = [];
        lr_m1m2 = []; lr_m1m3 = []; lr_m2m3 = [];
        lr_joint = []; lr_m1m2m3 = []; lr_dyn_prior = [];
    else:
        vae_trimodal.train();

    mod_weights = utils.reweight_weights(torch.Tensor(flags.alpha_modalities));
    mod_weights = mod_weights.to(flags.device);

    num_batches_epoch = int(data.__len__() /float(flags.batch_size));

    step_print_progress = 0;
    for iteration in range(num_batches_epoch):
        # load a mini-batch
        try:
            batch = next(loader)
        except TypeError:
            batch = None;
        while batch is None:
            try:
                batch = next(loader)
            except TypeError:
                batch = None;
            iteration += 1;
        m1_batch, m2_batch, m3_batch, labels_batch = batch;
        m1_batch = m1_batch.to(flags.device);
        m2_batch = m2_batch.to(flags.device);
        m3_batch = m3_batch.to(flags.device);
        labels_batch = labels_batch.to(flags.device);

        results_joint = vae_trimodal(input_pa=Variable(m1_batch),
                                     input_lat=Variable(m2_batch),
                                     input_text=Variable(m3_batch));
        m1_reconstruction = results_joint['rec']['pa'];
        m2_reconstruction = results_joint['rec']['lateral'];
        m3_reconstruction = results_joint['rec']['text'];
        latents = results_joint['latents'];
        [m1_c_mu, m1_c_logvar] = latents['pa'];
        [m1_s_mu, m1_s_logvar] = latents['pa_style'];
        [m2_c_mu, m2_c_logvar] = latents['lateral'];
        [m2_s_mu, m2_s_logvar] = latents['lateral_style'];
        [m3_c_mu, m3_c_logvar] = latents['text'];
        [m3_s_mu, m3_s_logvar] = latents['text_style'];
        [m1m2_c_mu, m1m2_c_logvar] = latents['pa_lateral'];
        [m1m3_c_mu, m1m3_c_logvar] = latents['pa_text'];
        [m2m3_c_mu, m2m3_c_logvar] = latents['lateral_text'];
        [m1m2m3_c_mu, m1m2m3_c_logvar] = latents['pa_lateral_text'];
        [group_mu, group_logvar] = results_joint['group_distr'];
        group_divergence = results_joint['joint_divergence'];
        if flags.modality_jsd:
            [dyn_prior_mu, dyn_prior_logvar] = results_joint['dyn_prior'];
            kld_dyn_prior = calc_kl_divergence(dyn_prior_mu, dyn_prior_logvar, norm_value=flags.batch_size)

        if flags.factorized_representation:
            kld_m1_style = calc_kl_divergence(m1_s_mu, m1_s_logvar, norm_value=flags.batch_size)
            kld_m2_style = calc_kl_divergence(m2_s_mu, m2_s_logvar, norm_value=flags.batch_size)
            kld_m3_style = calc_kl_divergence(m3_s_mu, m3_s_logvar, norm_value=flags.batch_size)
        else:
            m1_style_mu = torch.zeros(1).to(flags.device);
            m1_style_logvar = torch.zeros(1).to(flags.device);
            m2_style_mu = torch.zeros(1).to(flags.device);
            m2_style_logvar = torch.zeros(1).to(flags.device);
            m3_style_mu = torch.zeros(1).to(flags.device);
            m3_style_logvar = torch.zeros(1).to(flags.device);
            kld_m1_style = torch.zeros(1).to(flags.device);
            kld_m2_style = torch.zeros(1).to(flags.device);
            kld_m3_style = torch.zeros(1).to(flags.device);

        kld_m1_class = calc_kl_divergence(m1_c_mu, m1_c_logvar, norm_value=flags.batch_size);
        kld_m2_class = calc_kl_divergence(m2_c_mu, m2_c_logvar, norm_value=flags.batch_size);
        kld_m3_class = calc_kl_divergence(m3_c_mu, m3_c_logvar, norm_value=flags.batch_size);
        kld_group = calc_kl_divergence(group_mu, group_logvar, norm_value=flags.batch_size);
        rec_error_m1 = -log_prob_img(m1_reconstruction, Variable(m1_batch), flags.batch_size);
        rec_error_m2 = -log_prob_img(m2_reconstruction, Variable(m2_batch), flags.batch_size);
        rec_error_m3 = -log_prob_text(m3_reconstruction, Variable(m3_batch), flags.batch_size);

        rec_error_weighted = rec_weight_m1*rec_error_m1 + rec_weight_m2*rec_error_m2 + rec_weight_m3*rec_error_m3;
        if flags.modality_jsd or flags.modality_moe:
            kld_style = beta_m1_style * kld_m1_style + beta_m2_style * kld_m2_style + beta_m3_style*kld_m3_style;
            kld_content = group_divergence;
            kld_weighted_all = beta_style * kld_style + beta_content * kld_content;
            total_loss = rec_weight * rec_error_weighted + beta * kld_weighted_all
        elif flags.modality_poe:
            klds_joint = {'content': group_divergence,
                          'style': {'pa': kld_m1_style,
                                    'lateral': kld_m2_style,
                                    'text': kld_m3_style}}
            recs_joint = {'pa': rec_error_m1,
                          'lateral': rec_error_m2,
                          'text': rec_error_m3}
            elbo_joint = utils.calc_elbo_mimic(flags, 'joint', recs_joint, klds_joint);
            results_pa = vae_trimodal(input_pa=m1_batch,
                                         input_lat=None,
                                         input_text=None);
            pa_m1_rec = results_pa['rec']['pa'];
            pa_m1_rec_error = -log_prob_img(pa_m1_rec, m1_batch, flags.batch_size);
            recs_pa = {'pa': pa_m1_rec_error}
            klds_pa = {'content': kld_m1_class,
                       'style': {'pa': kld_m1_style}};
            elbo_pa = utils.calc_elbo_mimic(flags, 'pa', recs_pa, klds_pa);

            results_lateral = vae_trimodal(input_pa=None,
                                         input_lat=m2_batch,
                                         input_text=None);
            lateral_m2_rec = results_lateral['rec']['lateral']
            lateral_m2_rec_error = -log_prob_img(lateral_m2_rec, m2_batch, flags.batch_size);
            recs_lateral = {'lateral': lateral_m2_rec_error};
            klds_lateral = {'content': kld_m2_class,
                         'style': {'lateral': kld_m2_style}}
            elbo_lateral = utils.calc_elbo_mimic(flags, 'lateral', recs_lateral,
                                           klds_lateral);

            results_text = vae_trimodal(input_pa=None,
                                        input_lat=None,
                                        input_text=m3_batch);
            text_m3_rec = results_text['rec']['text'];
            text_m3_rec_error = -log_prob_text(text_m3_rec, m3_batch, flags.batch_size);
            recs_text = {'text': text_m3_rec_error};
            klds_text = {'content': kld_m3_class,
                         'style': {'text': kld_m3_style}};
            elbo_text = utils.calc_elbo_mimic(flags, 'text', recs_text, klds_text);
            total_loss = elbo_joint + elbo_pa + elbo_lateral + elbo_text;

        data_class_m1 = m1_class_mu.cpu().data.numpy();
        data_class_m2 = m2_class_mu.cpu().data.numpy();
        data_class_m3 = m3_class_mu.cpu().data.numpy();
        data_class_m1m2 = m1m2_c_mu.cpu().data.numpy();
        data_class_m1m3 = m1m3_c_mu.cpu().data.numpy();
        data_class_m2m3 = m2m3_c_mu.cpu().data.numpy();
        data_class_m1m2m3 = m1m2m3_c_mu.cpu().data.numpy();
        data_class_joint = group_mu.cpu().data.numpy();
        data = {'pa': data_class_m1,
                'lateral': data_class_m2,
                'text': data_class_m3,
                'pl': data_class_m1m2,
                'pt': data_class_m1m3,
                'lt': data_class_m2m3,
                'plt': data_class_m1m2m3,
                'joint': data_class_joint,
                }
        if flags.factorized_representation:
            data_style_m1 = m1_style_mu.cpu().data.numpy();
            data_style_m2 = m2_style_mu.cpu().data.numpy();
            data_style_m3 = m3_style_mu.cpu().data.numpy();
            data['pa_style'] = data_style_m1;
            data['lateral_style'] = data_style_m2;
            data['text_style'] = data_style_m3;
        labels = labels_batch.cpu().data.numpy().reshape(flags.batch_size, len(LABELS));
        if (epoch + 1) % flags.eval_freq == 0 or (epoch + 1) == flags.end_epoch:
            if train == False:
                # log-likelihood
                if flags.calc_nll:
                    # 12 imp samples because dividible by 3 (needed for joint)
                    ll_pa_batch = calc_log_likelihood_batch(flags, 'pa', batch, vae_trimodal, mod_weights, num_imp_samples=12)
                    ll_lateral_batch = calc_log_likelihood_batch(flags, 'lateral', batch, vae_trimodal, mod_weights, num_imp_samples=12)
                    ll_text_batch = calc_log_likelihood_batch(flags, 'text', batch, vae_trimodal, mod_weights, num_imp_samples=12)
                    ll_pl_batch = calc_log_likelihood_batch(flags, 'pa_lateral', batch, vae_trimodal, mod_weights, num_imp_samples=12);
                    ll_pt_batch = calc_log_likelihood_batch(flags, 'pa_text', batch, vae_trimodal, mod_weights, num_imp_samples=12);
                    ll_lt_batch = calc_log_likelihood_batch(flags, 'lateral_text', batch, vae_trimodal, mod_weights, num_imp_samples=12);
                    ll_joint = calc_log_likelihood_batch(flags, 'joint', batch, vae_trimodal, mod_weights, num_imp_samples=12);
                    ll_pa_pa.append(ll_pa_batch['pa'].item())
                    ll_pa_lateral.append(ll_pa_batch['lateral'].item())
                    ll_pa_text.append(ll_pa_batch['text'].item())
                    ll_pa_joint.append(ll_pa_batch['joint'].item())
                    ll_lateral_pa.append(ll_lateral_batch['pa'].item())
                    ll_lateral_lateral.append(ll_lateral_batch['lateral'].item())
                    ll_lateral_text.append(ll_lateral_batch['text'].item())
                    ll_lateral_joint.append(ll_lateral_batch['joint'].item())
                    ll_text_pa.append(ll_text_batch['pa'].item())
                    ll_text_lateral.append(ll_text_batch['lateral'].item())
                    ll_text_text.append(ll_text_batch['text'].item())
                    ll_text_joint.append(ll_text_batch['joint'].item())
                    ll_joint_pa.append(ll_joint['pa'].item())
                    ll_joint_lateral.append(ll_joint['lateral'].item())
                    ll_joint_text.append(ll_joint['text'].item())
                    ll_joint_joint.append(ll_joint['joint'].item());
                    ll_pl_text.append(ll_pl_batch['text'].item());
                    ll_pl_joint.append(ll_pl_batch['joint'].item());
                    ll_pt_lateral.append(ll_pt_batch['lateral'].item());
                    ll_pt_joint.append(ll_pt_batch['joint'].item());
                    ll_lt_pa.append(ll_lt_batch['pa'].item());
                    ll_lt_joint.append(ll_lt_batch['joint'].item());

                # conditional generation 1 modalitiy available
                latent_distr = dict();
                latent_distr['pa'] = [m1_c_mu, m1_c_logvar];
                latent_distr['lateral'] = [m2_c_mu, m2_c_logvar];
                latent_distr['text'] = [m3_c_mu, m3_c_logvar];
                if flags.modality_jsd:
                    latent_distr['dynamic_prior'] = [dyn_prior_mu, dyn_prior_logvar];
                rand_gen_samples = vae_trimodal.generate();
                cond_gen_samples = vae_trimodal.cond_generation_1a(latent_distr);
                m1_cond = cond_gen_samples['pa']  # samples conditioned on pa                
                m2_cond = cond_gen_samples['lateral']  # samples conditioned on lateral;
                m3_cond = cond_gen_samples['text']  # samples conditioned on lateral;
                real_samples = {'pa': m1_batch, 'lateral': m2_batch, 'text': m3_batch}
                if (flags.batch_size*iteration) < flags.num_samples_fid:
                    save_generated_samples_singlegroup(flags, iteration, alphabet, 'real', real_samples)
                    save_generated_samples_singlegroup(flags, iteration, alphabet, 'random_sampling', rand_gen_samples)
                    save_generated_samples_singlegroup(flags, iteration,
                                                       alphabet,
                                                       'cond_gen_1a2m_pa',
                                                       m1_cond)
                    save_generated_samples_singlegroup(flags, iteration,
                                                       alphabet,
                                                       'cond_gen_1a2m_lateral', m2_cond)
                    save_generated_samples_singlegroup(flags, iteration, alphabet, 'cond_gen_1a2m_text', m3_cond)

                #conditional generation: 2 available modalities
                latent_distr_pairs = dict();
                latent_distr_pairs['pa_lateral'] = {'latents': {'pa': [m1_c_mu, m1_c_logvar],
                                                                'lateral': [m2_c_mu, m2_c_logvar]},
                                                    'weights': [flags.alpha_modalities[1],
                                                                flags.alpha_modalities[2]]};
                latent_distr_pairs['pa_text'] = {'latents': {'pa': [m1_c_mu, m1_c_logvar],
                                                             'text': [m3_c_mu, m3_c_logvar]},
                                                 'weights': [flags.alpha_modalities[1],
                                                             flags.alpha_modalities[3]]};
                latent_distr_pairs['lateral_text'] = {'latents': {'lateral': [m2_c_mu, m2_c_logvar],
                                                                  'text': [m3_c_mu, m3_c_logvar]},
                                                      'weights': [flags.alpha_modalities[2],
                                                                  flags.alpha_modalities[3]]};
                cond_gen_2a = vae_trimodal.cond_generation_2a(latent_distr_pairs)
                if (flags.batch_size*iteration) < flags.num_samples_fid:
                    save_generated_samples_singlegroup(flags, iteration,
                                                       alphabet,
                                                       'cond_gen_2a1m_pa_lateral',
                                                       cond_gen_2a['pa_lateral']);
                    save_generated_samples_singlegroup(flags, iteration, alphabet, 
                                                       'cond_gen_2a1m_pa_text',
                                                       cond_gen_2a['pa_text']);
                    save_generated_samples_singlegroup(flags, iteration, alphabet,
                                                       'cond_gen_2a1m_lateral_text',
                                                       cond_gen_2a['lateral_text']);

                if flags.modality_jsd:
                    # conditional generation 2 modalities available -> dyn
                    # prior generation
                    cond_gen_dp = vae_trimodal.cond_generation_2a(latent_distr_pairs,
                                                                  dp_gen=True)
                    if (flags.batch_size*iteration) < flags.num_samples_fid:
                        save_generated_samples_singlegroup(flags, iteration,
                                                           alphabet,
                                                           'dynamic_prior_pa_lateral',
                                                           cond_gen_dp['pa_lateral']);
                        save_generated_samples_singlegroup(flags, iteration, alphabet,
                                                           'dynamic_prior_pa_text',
                                                           cond_gen_dp['pa_text']);
                        save_generated_samples_singlegroup(flags, iteration,
                                                           alphabet,
                                                           'dynamic_prior_2a1m_lateral_text',
                                                           cond_gen_dp['lateral_text']);

                if (model_clf_m1 is not None and model_clf_m2 is not None and
                    model_clf_m3 is not None):
                    clfs_gen = {'pa': model_clf_m1,
                                'lateral': model_clf_m2,
                                'text': model_clf_m3};
                    coherence_random_triples = classify_rand_gen_samples(flags, epoch, clfs_gen, rand_gen_samples);
                    random_gen_acc,append(coherence_random_triples);

                    eval_cond_m1 = classify_cond_gen_samples(flags, epoch, clfs_gen, labels, m1_cond);
                    cg_m1['pa'].append(eval_cond_m1['pa']);
                    cg_m1['lateral'].append(eval_cond_m1['lateral']);
                    cg_m1['text'].append(eval_cond_m1['text']);
                    eval_cond_m2 = classify_cond_gen_samples(flags, epoch, clfs_gen, labels, m2_cond);
                    cg_m2['pa'].append(eval_cond_m2['pa']);
                    cg_m2['lateral'].append(eval_cond_m2['lateral']);
                    cg_m2['text'].append(eval_cond_m2['text']);
                    eval_cond_m3 = classify_cond_gen_samples(flags, epoch, clfs_gen, labels, m3_cond);
                    cg_m3['pa'].append(eval_cond_m3['pa']);
                    cg_m3['lateral'].append(eval_cond_m3['lateral']);
                    cg_m3['text'].append(eval_cond_m3['text']);

                    eval_cond_pl = classify_cond_gen_samples(flags, epoch, clfs_gen,
                                                      labels,
                                                      cond_gen_2a['pa_lateral']);
                    cg_m1m2['pa'].append(eval_cond_pl['pa']);
                    cg_m1m2['lateral'].append(eval_cond_pl['lateral']);
                    cg_m1m2['text'].append(eval_cond_pl['text']);

                    eval_cond_pt = classify_cond_gen_samples(flags, epoch, clfs_gen,
                                                      labels,
                                                      cond_gen_2a['pa_text']);
                    cg_m1m3['pa'].append(eval_cond_pt['pa']);
                    cg_m1m3['lateral'].append(eval_cond_pt['lateral']);
                    cg_m1m3['text'].append(eval_cond_pt['text']);

                    eval_cond_lt = classify_cond_gen_samples(flags, epoch, clfs_gen,
                                                      labels,
                                                      cond_gen_2a['lateral_text']);
                    cg_m2m3['pa'].append(eval_cond_lt['pa']);
                    cg_m2m3['lateral'].append(eval_cond_lt['lateral']);
                    cg_m2m3['text'].append(eval_cond_lt['text']);

                    if flags.modality_jsd:
                        cond_dp_pl = classify_cond_gen_samples(flags,
                                                               epoch,
                                                               clfs_gen,
                                                               labels,
                                                               cond_gen_dp['pa_lateral']);
                        cg_dp_m1m2['pa'].append(cond_dp_ms['pa']);
                        cg_dp_m1m2['lateral'].append(cond_dp_ms['lateral']);
                        cg_dp_m1m2['text'].append(cond_dp_ms['text']);
                        cond_dp_mt = classify_cond_gen_samples(flags,
                                                               epoch,
                                                               clfs_gen,
                                                               labels,
                                                               cond_gen_dp['pa_text']);
                        cg_dp_m1m3['pa'].append(cond_dp_mt['pa']);
                        cg_dp_m1m3['lateral'].append(cond_dp_mt['lateral']);
                        cg_dp_m1m3['text'].append(cond_dp_mt['text']);
                        cond_dp_st = classify_cond_gen_samples(flags,
                                                               epoch,
                                                               clfs_gen,
                                                               labels,
                                                               cond_gen_dp['lateral_text']);
                        cg_dp_m2m3['pa'].append(cond_dp_st['pa']);
                        cg_dp_m2m3['lateral'].append(cond_dp_st['lateral']);
                        cg_dp_m2m3['text'].append(cond_dp_st['text']);
            if train:
                if iteration == (num_batches_epoch - 1):
                    gt = np.argmax(labels, axis=1).astype(int)
                    clf_lr = train_clfs_latent_representation(data, labels);
            else:
                if clf_lr is not None:
                    ap_all = classify_latent_representations(flags, epoch, clf_lr, data, labels);
                    lr_m1_c.append(np.mean(accuracies['pa']))
                    lr_m2_c.append(np.mean(accuracies['lateral']))
                    lr_m3_c.append(np.mean(accuracies['text']))
                    lr_m1m2.append(np.mean(accuracies['pl']))
                    lr_m1m3.append(np.mean(accuracies['pt']))
                    lr_m2m3.append(np.mean(accuracies['lt']))
                    lr_m1m2m3.append(np.mean(accuracies['plt']))
                    lr_joint.append(np.mean(accuracies['joint']))
                    if flags.modality_jsd:
                        lr_dyn_prior.append(np.mean(accuracies['dyn_prior']));
                    if flags.factorized_representation:
                        lr_m1_s.append(np.mean(accuracies['pa_style']))
                        lr_m2_s.append(np.mean(accuracies['lateral_style']))
                        lr_m3_s.append(np.mean(accuracies['text_style']))

        # backprop
        if train == True:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            utils.printProgressBar(step_print_progress, num_batches_epoch)

        # write scalars to tensorboard
        name = "train" if train else "test"
        writer.add_scalars('%s/Loss' % name, {'loss': total_loss.data.item()}, step_logs)
        writer.add_scalars('%s/RecLoss' % name, {
            'M1': rec_error_m1.item(),
            'M2': rec_error_m2.item(),
            'M3': rec_error_m3.item(),
        }, step_logs)
        writer.add_scalars('%s/KLD' % name, {
            'Content_M1': kld_m1_class.item(),
            'Style_M1': kld_m1_style.item(),
            'Content_M2': kld_m2_class.item(),
            'Style_M2': kld_m2_style.item(),
            'Content_M3': kld_m3_class.item(),
            'Style_M3': kld_m3_style.item(),
            }, step_logs)
        writer.add_scalars('%s/group_divergence' % name, {
            'group_div': group_divergence.item(),
            'KLDgroup': kld_group.item(),
        }, step_logs)
        if flags.modality_jsd:
            writer.add_scalars('%s/group_divergence' % name, {
                'KLDdyn_prior': kld_dyn_prior.item(),
            }, step_logs)
            writer.add_scalars('%s/mu' % name, {
                'content_alpha': group_mu.mean().item(),
            }, step_logs)
            writer.add_scalars('%s/logvar' % name, {
                'content_alpha': group_logvar.mean().item(),
            }, step_logs)
        writer.add_scalars('%s/mu' % name, {
            'content_m1': m1_class_mu.mean().item(),
            'style_m1': m1_style_mu.mean().item(),
            'content_m2': m2_class_mu.mean().item(),
            'style_m2': m2_style_mu.mean().item(),
            'content_m3': m3_class_mu.mean().item(),
            'style_m3': m3_style_mu.mean().item(),
        }, step_logs)
        writer.add_scalars('%s/logvar' % name, {
            'style_m1': m1_style_logvar.mean().item(),
            'content_m1': m1_class_logvar.mean().item(),
            'style_m2': m2_style_logvar.mean().item(),
            'content_m2': m2_class_logvar.mean().item(),
            'style_m3': m3_style_logvar.mean().item(),
            'content_m3': m3_class_logvar.mean().item(),
        }, step_logs)
        step_logs += 1
        step_print_progress += 1;

    # write style-transfer ("swapping") figure to tensorboard
    if train == False:
        if flags.factorized_representation:
            # pa to pa: swapping content and style intra modal
            swapping_figs = generate_swapping_plot(flags, epoch, vae_trimodal,
                                                   SAMPLE1, alphabet)
            swaps_pa_content = swapping_figs['pa'];
            swaps_lateral_content = swapping_figs['lateral'];
            swaps_text_content = swapping_figs['text'];
            swap_pa_pa = swaps_pa_content['pa'];
            swap_pa_lateral = swaps_pa_content['lateral'];
            swap_pa_text = swaps_pa_content['text'];
            swap_lateral_pa = swaps_lateral_content['pa'];
            swap_lateral_lateral = swaps_lateral_content['lateral'];
            swap_lateral_text = swaps_lateral_content['text'];
            swap_text_pa = swaps_text_content['pa'];
            swap_text_lateral = swaps_text_content['lateral'];
            swap_text_text = swaps_text_content['text'];
            writer.add_image('Swapping pa to pa', swap_pa_pa, epoch, dataformats="HWC")
            writer.add_image('Swapping pa to lateral', swap_pa_lateral, epoch, dataformats="HWC")
            writer.add_image('Swapping pa to text', swap_pa_text, epoch, dataformats="HWC")
            writer.add_image('Swapping lateral to pa', swap_lateral_pa, epoch, dataformats="HWC")
            writer.add_image('Swapping lateral to lateral',
                             swap_lateral_lateral, epoch, dataformats="HWC")
            writer.add_image('Swapping lateral to text', swap_lateral_text, epoch, dataformats="HWC")
            writer.add_image('Swapping text to pa', swap_text_pa, epoch, dataformats="HWC")
            writer.add_image('Swapping text to lateral', swap_text_lateral, epoch, dataformats="HWC")
            writer.add_image('Swapping text to text', swap_text_text, epoch, dataformats="HWC")

        conditional_figs = generate_conditional_fig_1a(flags, epoch, vae_trimodal, SAMPLE1, alphabet)
        figs_cond_pa = conditional_figs['pa'];
        figs_cond_lateral = conditional_figs['lateral'];
        figs_cond_text = conditional_figs['text'];
        cond_pa_pa = figs_cond_pa['pa'];
        cond_pa_lateral = figs_cond_pa['lateral'];
        cond_pa_text = figs_cond_pa['text'];
        cond_lateral_pa = figs_cond_lateral['pa'];
        cond_lateral_lateral = figs_cond_lateral['lateral'];
        cond_lateral_text = figs_cond_lateral['text'];
        cond_text_pa = figs_cond_text['pa'];
        cond_text_lateral = figs_cond_text['lateral'];
        cond_text_text = figs_cond_text['text'];
        writer.add_image('Cond_pa_to_pa', cond_pa_pa, epoch, dataformats="HWC")
        writer.add_image('Cond_pa_to_lateral', cond_pa_lateral, epoch, dataformats="HWC")
        writer.add_image('Cond_pa_to_text', cond_pa_text, epoch, dataformats="HWC")
        writer.add_image('Cond_lateral_to_pa', cond_lateral_pa, epoch, dataformats="HWC")
        writer.add_image('Cond_lateral_to_lateral', cond_lateral_lateral, epoch, dataformats="HWC")
        writer.add_image('Cond_lateral_to_text', cond_lateral_text, epoch, dataformats="HWC")
        writer.add_image('Cond_text_to_pa', cond_text_pa, epoch, dataformats="HWC")
        writer.add_image('Cond_text_to_lateral', cond_text_lateral, epoch, dataformats="HWC")
        writer.add_image('Cond_text_to_text', cond_text_text, epoch, dataformats="HWC")

        conditional_figs_2a = generate_conditional_fig_2a(flags, epoch,
                                                          vae_trimodal,
                                                          SAMPLE1, alphabet);
        figs_cond_pl = conditional_figs_2a['pa_lateral'];
        figs_cond_pt = conditional_figs_2a['pa_text'];
        figs_cond_lt = conditional_figs_2a['lateral_text'];
        cond_pl_p = figs_cond_pl['pa'];
        cond_pl_l = figs_cond_pl['lateral'];
        cond_pl_t = figs_cond_pl['text'];
        cond_pt_p = figs_cond_pt['pa'];
        cond_pt_l = figs_cond_pt['lateral'];
        cond_pt_t = figs_cond_pt['text'];
        cond_lt_p = figs_cond_lt['pa'];
        cond_lt_l = figs_cond_lt['lateral'];
        cond_lt_t = figs_cond_lt['text'];
        writer.add_image('Cond_pl_to_p', cond_pl_p, epoch, dataformats="HWC")
        writer.add_image('Cond_pl_to_l', cond_pl_l, epoch, dataformats="HWC")
        writer.add_image('Cond_pl_to_t', cond_pl_t, epoch, dataformats="HWC")
        writer.add_image('Cond_pt_to_p', cond_pt_p, epoch, dataformats="HWC")
        writer.add_image('Cond_pt_to_l', cond_pt_l, epoch, dataformats="HWC")
        writer.add_image('Cond_pt_to_t', cond_pt_t, epoch, dataformats="HWC")
        writer.add_image('Cond_lt_to_p', cond_lt_p, epoch, dataformats="HWC")
        writer.add_image('Cond_lt_to_l', cond_lt_l, epoch, dataformats="HWC")
        writer.add_image('Cond_lt_to_t', cond_lt_t, epoch, dataformats="HWC")

        random_figs = generate_random_samples_plots(flags, epoch,
                                                    vae_trimodal, alphabet);
        random_pa = random_figs['pa'];
        random_lateral = random_figs['lateral'];
        random_text = random_figs['text'];
        writer.add_image('Random PA', random_pa, epoch, dataformats="HWC");
        writer.add_image('Random SVHN', random_lateral, epoch, dataformats="HWC");
        writer.add_image('Random Text', random_text, epoch, dataformats="HWC");

        if (epoch + 1) % flags.eval_freq == 0 or (epoch + 1) == flags.end_epoch:
            cg_m1['pa'] = np.mean(np.array(cg_m1['pa']))
            cg_m1['lateral'] = np.mean(np.array(cg_m1['lateral']))
            cg_m1['text'] = np.mean(np.array(cg_m1['text']))
            cg_m2['pa'] = np.mean(np.array(cg_m2['pa']))
            cg_m2['lateral'] = np.mean(np.array(cg_m2['lateral']))
            cg_m2['text'] = np.mean(np.array(cg_m2['text']))
            cg_m3['pa'] = np.mean(np.array(cg_m3['pa']))
            cg_m3['lateral'] = np.mean(np.array(cg_m3['lateral']))
            cg_m3['text'] = np.mean(np.array(cg_m3['text']))
            writer.add_scalars('%s/cond_pa_clf' % name,
                               cg_m1, step_logs)
            writer.add_scalars('%s/cond_lateral_clf' % name,
                               cg_m2, step_logs)
            writer.add_scalars('%s/cond_text_clf' % name,
                               cg_m3, step_logs)
            writer.add_scalars('%s/random_coherence' % name, {
                'random': np.mean(np.array(random_gen_acc)),
            }, step_logs)
            cg_m1m2['pa'] = np.mean(np.array(cg_m1m2['pa']))
            cg_m1m2['lateral'] = np.mean(np.array(cg_m1m2['lateral']))
            cg_m1m2['text'] = np.mean(np.array(cg_m1m2['text']))
            cg_m1m3['pa'] = np.mean(np.array(cg_m1m3['pa']))
            cg_m1m3['lateral'] = np.mean(np.array(cg_m1m3['lateral']))
            cg_m1m3['text'] = np.mean(np.array(cg_m1m3['text']))
            cg_m2m3['pa'] = np.mean(np.array(cg_m2m3['pa']))
            cg_m2m3['lateral'] = np.mean(np.array(cg_m2m3['lateral']))
            cg_m2m3['text'] = np.mean(np.array(cg_m2m3['text']))
            writer.add_scalars('%s/cond_pl_clf' % name,
                               cg_m1m2, step_logs)
            writer.add_scalars('%s/cond_pt_clf' % name,
                               cg_m1m3, step_logs)
            writer.add_scalars('%s/cond_lt_clf' % name,
                               cg_m2m3, step_logs)
            if flags.modality_jsd:
                cg_dp_m1m2['pa'] = np.mean(np.array(cg_dp_m1m2['pa']))
                cg_dp_m1m2['lateral'] = np.mean(np.array(cg_dp_m1m2['lateral']))
                cg_dp_m1m2['text'] = np.mean(np.array(cg_dp_m1m2['text']))
                cg_dp_m1m3['pa'] = np.mean(np.array(cg_dp_m1m3['pa']))
                cg_dp_m1m3['lateral'] = np.mean(np.array(cg_dp_m1m3['lateral']))
                cg_dp_m1m3['text'] = np.mean(np.array(cg_dp_m1m3['text']))
                cg_dp_m2m3['pa'] = np.mean(np.array(cg_dp_m2m3['pa']))
                cg_dp_m2m3['lateral'] = np.mean(np.array(cg_dp_m2m3['lateral']))
                cg_dp_m2m3['text'] = np.mean(np.array(cg_dp_m2m3['text']))
                writer.add_scalars('%s/cond_pl_dp_clf' % name,
                                   cg_dp_m1m2, step_logs)
                writer.add_scalars('%s/cond_pt_dp_clf' % name,
                                   cg_dp_m1m3, step_logs)
                writer.add_scalars('%s/cond_lt_dp_clf' % name,
                                   cg_dp_m2m3, step_logs)

            writer.add_scalars('%s/representation' % name, {
                'm1': np.mean(np.array(lr_m1_c)),
                'm2': np.mean(np.array(lr_m2_c)),
                'm3': np.mean(np.array(lr_m3_c)),
                'm1m2': np.mean(np.array(lr_m1m2)),
                'm1m3': np.mean(np.array(lr_m1m3)),
                'm2m3': np.mean(np.array(lr_m2m3)),
                'm1m2m3': np.mean(np.array(lr_m1m2m3)),
                'joint': np.mean(np.array(lr_joint)),
            }, step_logs)
            if flags.factorized_representation:
                writer.add_scalars('%s/representation' % name, {
                    'style_m1': np.mean(np.array(lr_m1_s)),
                    'style_m2': np.mean(np.array(lr_m2_s)),
                    'style_m3': np.mean(np.array(lr_m3_s)),
                }, step_logs)
            if flags.calc_nll:
                writer.add_scalars('%s/marginal_loglikelihood' % name, {
                    'pa_pa': np.mean(ll_pa_pa),
                    'pa_lateral': np.mean(ll_pa_lateral),
                    'pa_text': np.mean(ll_pa_text),
                    'pa_joint': np.mean(ll_pa_joint),
                    'lateral_pa': np.mean(ll_lateral_pa),
                    'lateral_lateral': np.mean(ll_lateral_lateral),
                    'lateral_text': np.mean(ll_lateral_text),
                    'lateral_joint': np.mean(ll_lateral_joint),
                    'text_pa': np.mean(ll_text_pa),
                    'text_lateral': np.mean(ll_text_lateral),
                    'text_text': np.mean(ll_text_lateral),
                    'text_joint': np.mean(ll_text_joint),
                    'synergy_pa': np.mean(ll_joint_pa),
                    'synergy_lateral': np.mean(ll_joint_lateral),
                    'synergy_text': np.mean(ll_joint_text),
                    'joint': np.mean(ll_joint_joint),
                    'pl_text': np.mean(ll_pl_text),
                    'pl_joint': np.mean(ll_pl_joint),
                    'pt_lateral': np.mean(ll_pt_lateral),
                    'pt_joint': np.mean(ll_pt_joint),
                    'lt_pa': np.mean(ll_lt_pa),
                    'lt_joint': np.mean(ll_lt_joint),
                }, step_logs)

        if ((epoch + 1) % flags.eval_freq_fid == 0 or (epoch + 1) == flags.end_epoch):
            cond_1a2m = {'pa': os.path.join(flags.dir_gen_eval_fid_cond_gen_1a2m, 'pa'),
                         'lateral': os.path.join(flags.dir_gen_eval_fid_cond_gen_1a2m, 'lateral'),
                         'text': os.path.join(flags.dir_gen_eval_fid_cond_gen_1a2m, 'text')}
            cond_2a1m = {'pa_lateral': os.path.join(flags.dir_gen_eval_fid_cond_gen_2a1m, 'pa_lateral'),
                         'pa_text': os.path.join(flags.dir_gen_eval_fid_cond_gen_2a1m, 'pa_text'),
                         'lateral_text': os.path.join(flags.dir_gen_eval_fid_cond_gen_2a1m, 'lateral_text')}
            dyn_prior_2a = {'pa_lateral': os.path.join(flags.dir_gen_eval_fid_dynamicprior, 'pa_lateral'),
                            'pa_text': os.path.join(flags.dir_gen_eval_fid_dynamicprior, 'pa_text'),
                            'lateral_text': os.path.join(flags.dir_gen_eval_fid_dynamicprior, 'lateral_text')}
            if (epoch+1) == flags.eval_freq_fid:
                paths = {'real': flags.dir_gen_eval_fid_real,
                         'conditional_1a2m': cond_1a2m,
                         'conditional_2a1m': cond_2a1m,
                         'random': flags.dir_gen_eval_fid_random}
            else:
                paths = {'conditional_1a2m': cond_1a2m,
                         'conditional_2a1m': cond_2a1m,
                         'random': flags.dir_gen_eval_fid_random}
            if flags.modality_jsd:
                paths['dynamic_prior'] = dyn_prior_2a;
            calculate_inception_features_for_gen_evaluation(flags, paths,
                                                            modality='pa',
                                                            batch_size=64);
            calculate_inception_features_for_gen_evaluation(flags, paths,
                                                            modality='lateral',
                                                            batch_size=64);
            if flags.modality_poe or flags.modality_moe:
                conds = [cond_1a2m, cond_2a1m];
            else:
                conds = [cond_1a2m, cond_2a1m, dyn_prior_2a];
            act_lateral = load_inception_activations(flags, 'lateral', num_modalities=3, conditionals=conds);
            [act_inc_real_lateral, act_inc_rand_lateral, cond_1a2m_lateral,
             cond_2a1m_lateral, act_inc_dynprior_lateral] = act_lateral;
            act_pa = load_inception_activations(flags, 'pa', num_modalities=3, conditionals=conds)
            [act_inc_real_pa, act_inc_rand_pa, cond_1a2m_pa, cond_2a1m_pa, act_inc_dynprior_pa] = act_pa;
            fid_random_lateral = calculate_fid(act_inc_real_lateral,
                                               act_inc_rand_lateral);
            fid_cond_2a1m_lateral = calculate_fid_dict(act_inc_real_lateral,
                                                       cond_2a1m_lateral);
            fid_cond_1a2m_lateral = calculate_fid_dict(act_inc_real_lateral,
                                                       cond_1a2m_lateral);
            fid_random_pa = calculate_fid(act_inc_real_pa, act_inc_rand_pa);
            fid_cond_2a1m_pa = calculate_fid_dict(act_inc_real_pa, cond_2a1m_pa);
            fid_cond_1a2m_pa = calculate_fid_dict(act_inc_real_pa, cond_1a2m_pa);
            ap_prd_random_lateral = calculate_prd(act_inc_real_lateral,
                                                  act_inc_rand_lateral);
            ap_prd_cond_2a1m_lateral = calculate_prd_dict(act_inc_real_lateral,
                                                          cond_2a1m_lateral);
            ap_prd_cond_1a2m_lateral = calculate_prd_dict(act_inc_real_lateral,
                                                          cond_1a2m_lateral);
            ap_prd_random_pa = calculate_prd(act_inc_real_pa, act_inc_rand_pa);
            ap_prd_cond_1a2m_pa = calculate_prd_dict(act_inc_real_pa,
                                                     cond_1a2m_pa);
            ap_prd_cond_2a1m_pa = calculate_prd_dict(act_inc_real_pa,
                                                     cond_2a1m_pa);
            if flags.modality_jsd:
                fid_dp_2a1m_pa = calculate_fid_dict(act_inc_real_pa,
                                                    act_inc_dynprior_pa);
                ap_prd_dp_2a1m_pa = calculate_prd_dict(act_inc_real_pa,
                                                       act_inc_dynprior_pa);
                fid_dp_2a1m_lateral = calculate_fid_dict(act_inc_real_lateral,
                                                         act_inc_dynprior_lateral);
                ap_prd_dp_2a1m_lateral = calculate_prd_dict(act_inc_real_lateral,
                                                            act_inc_dynprior_lateral);

            writer.add_scalars('%s/fid' % name, {
                'pa_random': fid_random_pa,
                'lateral_random': fid_random_lateral,
                'lateral_cond_1a2m_lateral': fid_cond_1a2m_lateral['lateral'],
                'lateral_cond_1a2m_pa': fid_cond_1a2m_lateral['pa'],
                'lateral_cond_1a2m_text': fid_cond_1a2m_lateral['text'],
                'pa_cond_1a2m_lateral': fid_cond_1a2m_pa['lateral'],
                'pa_cond_1a2m_pa': fid_cond_1a2m_pa['pa'],
                'pa_cond_1a2m_text': fid_cond_1a2m_pa['text'],
                'lateral_2a1m_pa_text': fid_cond_2a1m_lateral['pa_text'],
                'pa_2a1m_lateral_text': fid_cond_2a1m_pa['lateral_text'],
            }, step_logs)
            writer.add_scalars('%s/prd' % name, {
                'pa_random': ap_prd_random_pa,
                'lateral_random': ap_prd_random_lateral,
                'lateral_cond_1a2m_lateral': ap_prd_cond_1a2m_lateral['lateral'],
                'lateral_cond_1a2m_pa': ap_prd_cond_1a2m_lateral['pa'],
                'lateral_cond_1a2m_text': ap_prd_cond_1a2m_lateral['text'],
                'pa_cond_1a2m_lateral': ap_prd_cond_1a2m_pa['lateral'],
                'pa_cond_1a2m_pa': ap_prd_cond_1a2m_pa['pa'],
                'pa_cond_1a2m_text': ap_prd_cond_1a2m_pa['text'],
                'lateral_2a1m_pa_text': ap_prd_cond_2a1m_lateral['pa_text'],
                'pa_2a1m_lateral_text': ap_prd_cond_2a1m_pa['lateral_text'],
            }, step_logs)
            if flags.modality_jsd:
                writer.add_scalars('%s/fid' % name, {
                    'pa_dp_2a1m_st': fid_dp_2a1m_pa['lateral_text'],
                    'lateral_dp_2a1m_mt': fid_dp_2a1m_lateral['pa_text'],
                }, step_logs)
                writer.add_scalars('%s/prd' % name, {
                    'pa_dp_2a1m_st': ap_prd_dp_2a1m_pa['lateral_text'],
                    'lateral_dp_2a1m_mt': ap_prd_dp_2a1m_lateral['pa_text'],
                }, step_logs)
    return step_logs, clf_lr;


def training_mimic(FLAGS):
    global SAMPLE1, SAMPLE2, SEED

    # load data set and create data loader instance
    print('Loading Mimic (multimodal) dataset...')
    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json');
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    FLAGS.num_features = len(alphabet)

    mimic_train = Mimic(FLAGS, alphabet, dataset=1)
    mimic_eval = Mimic(FLAGS, alphabet, dataset=2)
    print(mimic_eval.__len__())
    use_cuda = torch.cuda.is_available();
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu');
    # load global samples
    SAMPLE1 = get_10_mimic_samples(FLAGS, mimic_eval)

    # model definition
    vae_trimodal = VAEtrimodalMimic(FLAGS);

    # load saved models if load_saved flag is true
    if FLAGS.load_saved:
        vae_trimodal.load_state_dict(torch.load(os.path.join(FLAGS.dir_checkpoints, FLAGS.vae_trimodal_save)));

    model_clf_pa = None;
    model_clf_lat = None;
    model_clf_text = None;
    if FLAGS.use_clf:
        model_clf_pa = ClfImg(FLAGS);
        model_clf_pa.load_state_dict(torch.load(os.path.join(FLAGS.dir_clf, FLAGS.clf_save_m1)))
        model_clf_lat = ClfImg(FLAGS);
        model_clf_lat.load_state_dict(torch.load(os.path.join(FLAGS.dir_clf, FLAGS.clf_save_m2)))
        model_clf_text = ClfText(FLAGS);
        model_clf_text.load_state_dict(torch.load(os.path.join(FLAGS.dir_clf, FLAGS.clf_save_m3)))

    vae_trimodal = vae_trimodal.to(FLAGS.device);
    if model_clf_text is not None:
        model_clf_text = model_clf_text.to(FLAGS.device);
    if model_clf_pa is not None:
        model_clf_pa = model_clf_pa.to(FLAGS.device);
    if model_clf_lat is not None:
        model_clf_lat = model_clf_lat.to(FLAGS.device);

    # optimizer definition
    auto_encoder_optimizer = optim.Adam(
        list(vae_trimodal.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2))

    # initialize summary writer
    writer = SummaryWriter(FLAGS.dir_logs)

    str_flags = utils.save_and_log_flags(FLAGS);
    writer.add_text('FLAGS', str_flags, 0)

    it_num_batches = 0;
    for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch):
        print('epoch: ' + str(epoch))
        #utils.printProgressBar(epoch, FLAGS.end_epoch)
        # one epoch of training and testing
        it_num_batches, clf_lr = run_epoch(epoch, vae_trimodal,
                                           auto_encoder_optimizer, mimic_train,
                                           writer, alphabet,
                                           train=True, flags=FLAGS,
                                           model_clf_m1=model_clf_pa,
                                           model_clf_m2=model_clf_lat,
                                           model_clf_m3=model_clf_text,
                                           clf_lr=None,
                                           step_logs=it_num_batches)

        with torch.no_grad():
            it_num_batches, clf_lr = run_epoch(epoch, vae_trimodal,
                                               auto_encoder_optimizer,
                                               mimic_eval, writer, alphabet,
                                               train=False, flags=FLAGS,
                                               model_clf_m1=model_clf_pa,
                                               model_clf_m2=model_clf_lat,
                                               model_clf_m3=model_clf_text,
                                               clf_lr=clf_lr,
                                               step_logs=it_num_batches)
        print('')
        # save checkpoints after every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == FLAGS.end_epoch:
            dir_network_epoch = os.path.join(FLAGS.dir_checkpoints, str(epoch).zfill(4));
            if not os.path.exists(dir_network_epoch):
                os.makedirs(dir_network_epoch);
            vae_trimodal.save_networks()
            torch.save(vae_trimodal.state_dict(), os.path.join(dir_network_epoch, FLAGS.vae_trimodal_save))
