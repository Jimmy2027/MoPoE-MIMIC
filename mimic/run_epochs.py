import gc
import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from mimic.evaluation.divergence_measures.kl_div import calc_kl_divergence
from mimic.evaluation.eval_metrics.coherence import test_generation
from mimic.evaluation.eval_metrics.likelihood import estimate_likelihoods
from mimic.evaluation.eval_metrics.representation import test_clf_lr_all_subsets
from mimic.evaluation.eval_metrics.representation import train_clf_lr_all_subsets
from mimic.evaluation.eval_metrics.sample_quality import calc_prd_score
from mimic.utils import utils
from mimic.utils.TBLogger import TBLogger

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
        weighted_log_prob += exp.rec_weights[mod.name] * log_probs[mod.name];
    return log_probs, weighted_log_prob;


def calc_klds(exp, result):
    latents = result['latents']['subsets'];
    klds = dict();
    for m, key in enumerate(latents.keys()):
        mu, logvar = latents[key];
        klds[key] = calc_kl_divergence(mu, logvar,
                                       norm_value=exp.flags.batch_size)
    return klds;


def calc_klds_style(exp, result):
    latents = result['latents']['modalities']
    klds = dict()
    for m, key in enumerate(latents.keys()):
        if key.endswith('style'):
            mu, logvar = latents[key]
            klds[key] = calc_kl_divergence(mu, logvar,
                                           norm_value=exp.flags.batch_size)
    return klds


def calc_style_kld(exp, klds):
    mods = exp.modalities
    style_weights = exp.style_weights
    weighted_klds = 0.0
    for m, m_key in enumerate(mods.keys()):
        weighted_klds += style_weights[m_key] * klds[m_key + '_style']
    return weighted_klds


def basic_routine_epoch(exp, batch) -> dict:
    # set up weights
    beta_style = exp.flags.beta_style
    beta_content = exp.flags.beta_content
    beta = exp.flags.beta
    rec_weight = 1.0

    mm_vae = exp.mm_vae
    batch_d = batch[0]
    batch_l = batch[1]
    mods = exp.modalities
    for k, m_key in enumerate(batch_d.keys()):
        batch_d[m_key] = Variable(batch_d[m_key]).to(exp.flags.device)
    results = mm_vae(batch_d)
    # checking if the latents contain NaNs. If they do the experiment is started again
    for key in results['latents']['modalities']:
        if np.isnan(results['latents']['modalities'][key][0].mean().item()) or \
                np.isnan(results['latents']['modalities'][key][1].mean().item()):
            print(key, results['latents']['modalities'][key][0].mean().item())
            print(key, results['latents']['modalities'][key][1].mean().item())
            torch.cuda.empty_cache()
            gc.collect()
            del exp
            raise ValueError
            # exp.restart_experiment = True
        results['latents']['modalities'][key][1].mean().item()
    """
    temporary section to find out which values ar nan and give the warning of the logger
    """
    # temp
    import pandas as pd
    bugs_dir = os.path.join(exp.flags.dir_data, 'bugs')
    run = exp.flags.dir_experiment_run.split('/')[-1]
    if not os.path.exists(bugs_dir):
        os.makedirs(bugs_dir)
    if not os.path.exists(os.path.join(bugs_dir, 'basic_routine_epoch.csv')):
        table = pd.DataFrame()
    else:
        table = pd.read_csv(bugs_dir + '/basic_routine_epoch.csv')
    row = {'run': run, 'number_restarts': exp.number_restarts}
    for key in results['latents']['modalities']:
        row[key + '0_mean'] = results['latents']['modalities'][key][0].mean().item()
        row[key + '1_mean'] = results['latents']['modalities'][key][1].mean().item()
        row[key + '0_batch'] = batch_d[key][0].mean().item()
        row[key + '1_batch'] = batch_d[key][1].mean().item()
    table = table.append(row, ignore_index=True)
    table.to_csv(bugs_dir + '/basic_routine_epoch.csv', index=False)
    """
    end_temp
    """
    # getting the log probabilities
    log_probs, weighted_log_prob = calc_log_probs(exp, results, batch);
    group_divergence = results['joint_divergence'];

    klds = calc_klds(exp, results);
    if exp.flags.factorized_representation:
        klds_style = calc_klds_style(exp, results);

    # Calculation of the loss
    if (exp.flags.modality_jsd or exp.flags.modality_moe
            or exp.flags.joint_elbo):
        if exp.flags.factorized_representation:
            kld_style = calc_style_kld(exp, klds_style);
        else:
            kld_style = 0.0;
        kld_content = group_divergence;
        kld_weighted = beta_style * kld_style + beta_content * kld_content;
        total_loss = rec_weight * weighted_log_prob + beta * kld_weighted;
    elif exp.flags.modality_poe:
        klds_joint = {'content': group_divergence,
                      'style': dict()};
        recs_joint = dict();
        elbos = dict();
        for m, m_key in enumerate(mods.keys()):
            mod = mods[m_key];
            if exp.flags.factorized_representation:
                kld_style_m = klds_style[m_key + '_style'];
            else:
                kld_style_m = 0.0;
            klds_joint['style'][m_key] = kld_style_m;
            i_batch_mod = {m_key: batch_d[m_key]};
            r_mod = mm_vae(i_batch_mod);
            log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                              batch_d[m_key],
                                              exp.flags.batch_size);
            log_prob = {m_key: log_prob_mod};
            klds_mod = {'content': klds[m_key],
                        'style': {m_key: kld_style_m}};
            elbo_mod = utils.calc_elbo(exp, m_key, log_prob, klds_mod);
            elbos[m_key] = elbo_mod;
        elbo_joint = utils.calc_elbo(exp, 'joint', log_probs, klds_joint);
        elbos['joint'] = elbo_joint;
        total_loss = sum(elbos.values())

    out_basic_routine = dict();
    out_basic_routine['results'] = results;
    out_basic_routine['log_probs'] = log_probs;
    out_basic_routine['total_loss'] = total_loss;
    out_basic_routine['klds'] = klds;
    return out_basic_routine;


def train(epoch, exp, tb_logger):
    mm_vae = exp.mm_vae
    mm_vae.train()
    exp.mm_vae = mm_vae

    d_loader = DataLoader(exp.dataset_train, batch_size=exp.flags.batch_size,
                          shuffle=True,
                          num_workers=exp.flags.dataloader_workers, drop_last=True);
    if 0 < exp.flags.steps_per_training_epoch < len(d_loader):
        training_steps = exp.flags.steps_per_training_epoch
    else:
        training_steps = len(d_loader)

    for iteration, batch in tqdm(enumerate(d_loader), total=training_steps, postfix='train'):
        if iteration > training_steps:
            break
        basic_routine = basic_routine_epoch(exp, batch)
        results = basic_routine['results']
        total_loss = basic_routine['total_loss']
        klds = basic_routine['klds']
        log_probs = basic_routine['log_probs']
        # backprop
        exp.optimizer.zero_grad()
        total_loss.backward()
        exp.optimizer.step()
        tb_logger.write_training_logs(results, total_loss, log_probs, klds)


def test(epoch, exp, tb_logger):
    with torch.no_grad():
        mm_vae = exp.mm_vae;
        mm_vae.eval();
        exp.mm_vae = mm_vae;
        # set up weights
        beta_style = exp.flags.beta_style
        beta_content = exp.flags.beta_content
        beta = exp.flags.beta
        rec_weight = 1.0

        d_loader = DataLoader(exp.dataset_test, batch_size=exp.flags.batch_size,
                              shuffle=True,
                              num_workers=exp.flags.dataloader_workers, drop_last=True)
        total_losses = []
        for iteration, batch in tqdm(enumerate(d_loader), total=len(d_loader), postfix='test'):
            basic_routine = basic_routine_epoch(exp, batch)
            results = basic_routine['results']
            total_loss = basic_routine['total_loss']
            klds = basic_routine['klds']
            log_probs = basic_routine['log_probs']
            tb_logger.write_testing_logs(results, total_loss, log_probs, klds)
            total_losses.append(total_loss.item())

        print('generating plots')
        # plots = generate_plots(exp, epoch)
        # tb_logger.write_plots(plots, epoch)

        if (epoch + 1) % exp.flags.eval_freq == 0 or (epoch + 1) == exp.flags.end_epoch:
            if exp.flags.eval_lr:
                print('evaluation of latent representation')
                clf_lr = train_clf_lr_all_subsets(exp)
                lr_eval = test_clf_lr_all_subsets(epoch, clf_lr, exp)
                tb_logger.write_lr_eval(lr_eval)

            if exp.flags.use_clf:
                print('test generation')
                gen_eval = test_generation(epoch, exp)
                tb_logger.write_coherence_logs(gen_eval)

            if exp.flags.calc_nll:
                print('estimating likelihoods')
                lhoods = estimate_likelihoods(exp)
                tb_logger.write_lhood_logs(lhoods)

            if exp.flags.calc_prd and ((epoch + 1) % exp.flags.eval_freq_fid == 0):
                print('calculating prediction score')
                prd_scores = calc_prd_score(exp)
                tb_logger.write_prd_scores(prd_scores)
        exp.update_experiments_dataframe({'total_loss': np.mean(total_losses)})


def run_epochs(exp):
    # initialize summary writer
    writer = SummaryWriter(exp.flags.dir_logs)
    tb_logger = TBLogger(exp.flags.str_experiment, writer)
    str_flags = utils.save_and_log_flags(exp.flags);
    tb_logger.writer.add_text('FLAGS', str_flags, 0)

    print('training epochs progress:')
    for epoch in tqdm(range(exp.flags.start_epoch, exp.flags.end_epoch), postfix='epochs'):
        # utils.printProgressBar(epoch, exp.flags.end_epoch)
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

    #     if exp.restart_experiment:
    #         return exp.restart_experiment
    # return True
