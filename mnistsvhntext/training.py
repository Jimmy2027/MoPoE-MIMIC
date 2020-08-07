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

from mnistsvhntext.networks.VAEtrimodalSVHNMNIST import VAEtrimodalSVHNMNIST
from mnistsvhntext.networks.ConvNetworkImgClfMNIST import ClfImg as ClfImgMNIST
from mnistsvhntext.networks.ConvNetworkImgClfSVHN import ClfImgSVHN
from mnistsvhntext.networks.ConvNetworkTextClf import ClfText as ClfText

from divergence_measures.kl_div import calc_kl_divergence
from divergence_measures.mm_div import poe

from mnistsvhntext.testing import generate_swapping_plot
from mnistsvhntext.testing import generate_conditional_fig_M
from mnistsvhntext.testing import generate_random_samples_plots
from mnistsvhntext.testing import calculate_coherence
from mnistsvhntext.testing import classify_cond_gen_samples
from mnistsvhntext.testing import classify_latent_representations
from mnistsvhntext.testing import train_clf_lr
from utils.test_functions import calc_inception_features
from utils.test_functions import calculate_fid, calculate_fid_dict
from utils.test_functions import calculate_prd, calculate_prd_dict
from utils.test_functions import get_clf_activations
from utils.test_functions import load_inception_activations
from utils.save_samples import save_generated_samples_singlegroup
from mnistsvhntext.likelihood import calc_log_likelihood_batch


from utils import utils
from utils.TBLogger import TBLogger

torch.multiprocessing.set_sharing_strategy('file_system')

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


def train_lr_clf(exp):
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
    

def test_clf_lr(epoch, clf_lr, exp):
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


def test_generation(epoch, exp):
    mods = exp.modalities;
    mm_vae = exp.mm_vae;

    dict_mods = dict();
    for m, m_key in enumerate(mods.keys()):
        mod = mods[m_key]
        dict_mods[mod.name] = [];

    d_loader = DataLoader(exp.dataset_test,
                          batch_size=exp.flags.batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True);

    num_batches_epoch = int(exp.dataset_test.__len__() /float(exp.flags.batch_size));
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
        data_test = dict();
        cg = mm_vae.cond_generation_1a(lr_subsets)
        cg_perf = dict()
        for k, s_key in enumerate(lr_subsets.keys()):
            clf_cg = classify_cond_gen_samples(exp, epoch,
                                                batch_l,
                                                cg[s_key]);
            if not s_key in cg_perf.keys():
                cg_perf[s_key] = dict_mods.copy();
            for m, modname in enumerate(clf_cg):
                cg_perf[s_key][modname].append(clf_cg[modname]);
            if (exp.flags.batch_size*iteration) < exp.flags.num_samples_fid:
                save_generated_samples_singlegroup(exp, iteration,
                                                   s_key,
                                                   cg[s_key]);
    for k, key in enumerate(gen_perf.keys()):
        if not key == 'random':
            for m, mod in enumerate(gen_perf[key]):
                gen_perf[key][mod] = exp.mean_eval_metric(gen_perf[key][mod]);
        else:
            gen_perf['random'] = exp.mean_eval_metric(gen_perf['random']);
    return gen_perf;


def estimate_likelihoods(exp):
    model = exp.mm_vae;
    mods = exp.modalities;
    bs_normal = exp.flags.batch_size;
    exp.flags.batch_size = 64;
    d_loader = DataLoader(exp.dataset_test,
                          batch_size=exp.flags.batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True);

    subsets = exp.subsets;
    lhoods = dict()
    for iteration, batch in enumerate(d_loader):
        print(iteration)
        batch_d = batch[0];
        for m, m_key in enumerate(mods.keys()):
            batch_d[m_key] = batch_d[m_key].to(exp.flags.device);

        latents = model.inference(batch_d);
        for k, s_key in enumerate(subsets.keys()): 
            if s_key != '':
                if not s_key in lhoods.keys():
                    lhoods[s_key] = dict();
                subset = subsets[s_key];
                ll_batch = calc_log_likelihood_batch(exp, latents,
                                                     s_key, subset,
                                                     batch_d,
                                                     num_imp_samples=12)
                for l, m_key in enumerate(ll_batch.keys()):
                    if m_key not in lhoods[s_key].keys():
                        lhoods[s_key][m_key] = [];
                    lhoods[s_key][m_key].append(ll_batch[m_key]);
        #del batch_d;
        #torch.cuda.empty_cache()

    for k, s_key in enumerate(lhoods.keys()):
        lh_subset = lhoods[s_key];
        for l, m_key in enumerate(lh_subset.keys()):
            mean_val = np.mean(np.array(lh_subset[m_key]))
            lhoods[s_key][m_key] = mean_val;
    exp.flags.batch_size = bs_normal;
    return lhoods;


def calc_prd_score(exp):
    calc_inception_features(exp);
    acts = load_inception_activations(exp);
    ap_prds = dict();
    for m, m_key in enumerate(exp.modalities.keys()):
        mod = exp.modalities[m_key];
        if mod.gen_quality_eval:
            for k, key in enumerate(exp.subsets):
                if key == '':
                    continue;
                ap_prd = calculate_prd(acts[mod.name]['real'],
                                       acts[mod.name][key]);
                ap_prds[key + '_' + mod.name] = ap_prd;
    return ap_prds;


def generate_plots(exp, epoch):
    plots = dict();
    if exp.flags.factorized_representation:
        # mnist to mnist: swapping content and style intra modal
        swapping_figs = generate_swapping_plot(exp, epoch)
        plots['swapping'] = swapping_figs;

    for k in range(len(exp.modalities.keys())):
        cond_k = generate_conditional_fig_M(exp, epoch, k+1)
        plots['cond_gen_' + str(k+1).zfill(2)] = cond_k;

    plots['random'] = generate_random_samples_plots(exp, epoch);

    #conditional_figs_2a = generate_conditional_fig_2a(exp, epoch);
    #figs_cond_ms = conditional_figs_2a['mnist_svhn'];
    #figs_cond_mt = conditional_figs_2a['mnist_m3'];
    #figs_cond_st = conditional_figs_2a['svhn_m3'];
    #cond_ms_m = figs_cond_ms['m1'];
    #cond_ms_s = figs_cond_ms['m2'];
    #cond_ms_t = figs_cond_ms['m3'];
    #cond_mt_m = figs_cond_mt['m1'];
    #cond_mt_s = figs_cond_mt['m2'];
    #cond_mt_t = figs_cond_mt['m3'];
    #cond_st_m = figs_cond_st['m1'];
    #cond_st_s = figs_cond_st['m2'];
    #cond_st_t = figs_cond_st['m3'];
    #writer.add_image('Cond_ms_to_m', cond_ms_m, epoch, dataformats="HWC")
    #writer.add_image('Cond_ms_to_s', cond_ms_s, epoch, dataformats="HWC")
    #writer.add_image('Cond_ms_to_t', cond_ms_t, epoch, dataformats="HWC")
    #writer.add_image('Cond_mt_to_m', cond_mt_m, epoch, dataformats="HWC")
    #writer.add_image('Cond_mt_to_s', cond_mt_s, epoch, dataformats="HWC")
    #writer.add_image('Cond_mt_to_t', cond_mt_t, epoch, dataformats="HWC")
    #writer.add_image('Cond_st_to_m', cond_st_m, epoch, dataformats="HWC")
    #writer.add_image('Cond_st_to_s', cond_st_s, epoch, dataformats="HWC")
    #writer.add_image('Cond_st_to_t', cond_st_t, epoch, dataformats="HWC")

    #random_figs = generate_random_samples_plots(flags, epoch,
    #                                            vae_trimodal, alphabet);
    #random_mnist = random_figs['m1'];
    #random_svhn = random_figs['m2'];
    #random_m3 = random_figs['m3'];
    #writer.add_image('Random MNIST', random_mnist, epoch, dataformats="HWC");
    #writer.add_image('Random SVHN', random_svhn, epoch, dataformats="HWC");
    #writer.add_image('Random Text', random_m3, epoch, dataformats="HWC");
    return plots;


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
            clf_lr = train_lr_clf(exp);
            lr_eval = test_clf_lr(epoch, clf_lr, exp);
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
