import sys, os
import numpy as np
from itertools import cycle
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from divergence_measures.kl_div import calc_kl_divergence
from divergence_measures.mm_div import poe

from eval_metrics.coherence import test_generation
from eval_metrics.representation import train_clf_lr_all_subsets
from eval_metrics.representation import test_clf_lr_all_subsets
from eval_metrics.sample_quality import calc_prd_score
from eval_metrics.likelihood import estimate_likelihoods

from plotting import generate_plots

from utils import utils
from utils.TBLogger import TBLogger


# global variables
SEED = None 
SAMPLE1 = None
if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED) 


def calc_log_probs(exp, result, batch):
    mods = exp.modalities;
    log_probs = dict()
    weighted_log_prob = 0.0;
    for m, m_key in enumerate(mods.keys()):
        mod = mods[m_key]
        log_probs[mod.name] = -mod.calc_log_prob(result['rec'][mod.name],
                                                 batch[0][mod.name],
                                                 exp.flags.batch_size);
        weighted_log_prob += exp.rec_weights[mod.name]*log_probs[mod.name];
    return log_probs, weighted_log_prob;


def calc_klds(exp, result):
    latents = result['latents']['subsets'];
    klds = dict();
    for m, key in enumerate(latents.keys()):
        mu, logvar = latents[key];
        klds[key] = calc_kl_divergence(mu, logvar,
                                       norm_value=exp.flags.batch_size)
    return klds;


def calc_style_kld(exp, klds):
    mods = exp.modalities;
    style_weights = exp.style_weights;
    weighted_klds = 0.0;
    for m, mod in enumerate(mods.keys()):
        weighted_klds += style_weights[mod.name]*klds[mod.name+'_style'];
    return weighted_klds;



def basic_routine_epoch(exp, batch):
    # set up weights
    beta_style = exp.flags.beta_style;
    beta_content = exp.flags.beta_content;
    beta = exp.flags.beta;
    rec_weight = 1.0;

    mm_vae = exp.mm_vae;
    batch_d = batch[0];
    batch_l = batch[1];
    mods = exp.modalities;
    for k, m_key in enumerate(batch_d.keys()):
        batch_d[m_key] = Variable(batch_d[m_key]).to(exp.flags.device);
    results = mm_vae(batch_d);

    log_probs, weighted_log_prob = calc_log_probs(exp, results, batch);
    group_divergence = results['joint_divergence'];

    klds = calc_klds(exp, results);

    if exp.flags.modality_jsd or exp.flags.modality_moe:
        if exp.flags.factorized_representation:
            kld_style = calc_style_kld(exp, klds);
        else:
            kld_style = 0.0;
        kld_content = group_divergence;
        kld_weighted = beta_style * kld_style + beta_content * kld_content;
        total_loss = rec_weight * weighted_log_prob + beta * kld_weighted;
    elif exp.flags.modality_poe:
        klds_joint = {'content': group_divergence,
                      'style': {'m1': kld_m1_style,
                                'm2': kld_m2_style,
                                'm3': kld_m3_style}}
        recs_joint = {'m1': rec_error_m1,
                      'm2': rec_error_m2,
                      'm3': rec_error_m3}
        elbo_joint = utils.calc_elbo(flags, 'joint', recs_joint, klds_joint);
        results_mnist = vae_trimodal(input_mnist=m1_batch,
                                     input_svhn=None,
                                     input_m3=None);
        mnist_m1_rec = results_mnist['rec']['m1'];
        mnist_m1_rec_error = -log_prob_img(mnist_m1_rec, m1_batch, flags.batch_size);
        recs_mnist = {'m1': mnist_m1_rec_error}
        klds_mnist = {'content': kld_m1_class,
                      'style': {'m1': kld_m1_style}};
        elbo_mnist = utils.calc_elbo(flags, 'm1', recs_mnist, klds_mnist);

        results_svhn = vae_trimodal(input_mnist=None,
                                     input_svhn=m2_batch,
                                     input_m3=None);
        svhn_m2_rec = results_svhn['rec']['m2']
        svhn_m2_rec_error = -log_prob_img(svhn_m2_rec, m2_batch, flags.batch_size);
        recs_svhn = {'m2': svhn_m2_rec_error};
        klds_svhn = {'content': kld_m2_class,
                     'style': {'m2': kld_m2_style}}
        elbo_svhn = utils.calc_elbo(flags, 'm2', recs_svhn, klds_svhn);

        results_m3 = vae_trimodal(input_mnist=None,
                                     input_svhn=None,
                                     input_m3=m3_batch);
        m3_m3_rec = results_m3['rec']['m3'];
        m3_m3_rec_error = -log_prob_m3(m3_m3_rec, m3_batch, flags.batch_size);
        recs_m3 = {'m3': m3_m3_rec_error};
        klds_m3 = {'content': kld_m3_class,
                     'style': {'m3': kld_m3_style}};
        elbo_m3 = utils.calc_elbo(flags, 'm3', recs_m3, klds_m3);
        total_loss = elbo_joint + elbo_mnist + elbo_svhn + elbo_m3;

    out_basic_routine = dict();
    out_basic_routine['results'] = results;
    out_basic_routine['log_probs'] = log_probs;
    out_basic_routine['total_loss'] = total_loss;
    out_basic_routine['klds'] = klds;
    return out_basic_routine;


def train(epoch, exp, tb_logger):
    mm_vae = exp.mm_vae;
    mm_vae.train();
    exp.mm_vae = mm_vae;

    d_loader = DataLoader(exp.dataset_train, batch_size=exp.flags.batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True);

    for iteration, batch in enumerate(d_loader):
        basic_routine = basic_routine_epoch(exp, batch);
        results = basic_routine['results'];
        total_loss = basic_routine['total_loss'];
        klds = basic_routine['klds'];
        log_probs = basic_routine['log_probs'];
        # backprop
        exp.optimizer.zero_grad()
        total_loss.backward()
        exp.optimizer.step()
        tb_logger.write_training_logs(results, total_loss, log_probs, klds);


def test(epoch, exp, tb_logger):
    with torch.no_grad():
        mm_vae = exp.mm_vae;
        mm_vae.eval();
        exp.mm_vae = mm_vae;

        # set up weights
        beta_style = exp.flags.beta_style;
        beta_content = exp.flags.beta_content;
        beta = exp.flags.beta;
        rec_weight = 1.0;

        d_loader = DataLoader(exp.dataset_test, batch_size=exp.flags.batch_size,
                            shuffle=True,
                            num_workers=8, drop_last=True);

        for iteration, batch in enumerate(d_loader):
            basic_routine = basic_routine_epoch(exp, batch);
            results = basic_routine['results'];
            total_loss = basic_routine['total_loss'];
            klds = basic_routine['klds'];
            log_probs = basic_routine['log_probs'];
            tb_logger.write_testing_logs(results, total_loss, log_probs, klds);

        plots = generate_plots(exp, epoch);
        tb_logger.write_plots(plots, epoch);

        if (epoch + 1) % exp.flags.eval_freq == 0 or (epoch + 1) == exp.flags.end_epoch:
            if exp.flags.eval_lr:
                clf_lr = train_clf_lr_all_subsets(exp);
                lr_eval = test_clf_lr_all_subsets(epoch, clf_lr, exp);
                tb_logger.write_lr_eval(lr_eval);

            if exp.flags.use_clf:
                gen_eval = test_generation(epoch, exp);
                tb_logger.write_coherence_logs(gen_eval);

            if exp.flags.calc_nll:
                lhoods = estimate_likelihoods(exp);
                tb_logger.write_lhood_logs(lhoods);

            if exp.flags.calc_prd:
                prd_scores = calc_prd_score(exp);
                tb_logger.write_prd_scores(prd_scores)


def run_epochs(exp):

    # initialize summary writer
    writer = SummaryWriter(exp.flags.dir_logs)
    tb_logger = TBLogger(exp.flags.str_experiment, writer)
    str_flags = utils.save_and_log_flags(exp.flags);
    tb_logger.writer.add_text('FLAGS', str_flags, 0)

    print('training epochs progress:')
    for epoch in range(exp.flags.start_epoch, exp.flags.end_epoch):
        utils.printProgressBar(epoch, exp.flags.end_epoch)
        # one epoch of training and testing
        train(epoch, exp, tb_logger);
        test(epoch, exp, tb_logger);
        # save checkpoints after every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == exp.flags.end_epoch:
            dir_network_epoch = os.path.join(exp.flags.dir_checkpoints, str(epoch).zfill(4));
            if not os.path.exists(dir_network_epoch):
                os.makedirs(dir_network_epoch);
            exp.mm_vae.save_networks()
            torch.save(exp.mm_vae.state_dict(),
                       os.path.join(dir_network_epoch, exp.flags.mm_vae_save))
