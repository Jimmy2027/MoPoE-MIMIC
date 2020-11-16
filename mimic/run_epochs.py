import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.utils import get_data_loaders, samplers_set_epoch
from mimic.evaluation.divergence_measures.kl_div import calc_kl_divergence
from mimic.evaluation.eval_metrics.coherence import test_generation
from mimic.evaluation.eval_metrics.likelihood import estimate_likelihoods
from mimic.evaluation.eval_metrics.representation import test_clf_lr_all_subsets
from mimic.evaluation.eval_metrics.representation import train_clf_lr_all_subsets
from mimic.evaluation.eval_metrics.sample_quality import calc_prd_score
from mimic.utils import text
from mimic.utils import utils
from mimic.utils.experiment import Callbacks, NaNInLatent, MimicExperiment
from mimic.utils.plotting import generate_plots

# global variables
SEED = None
SAMPLE1 = None
if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)


def calc_log_probs(exp, result, batch):
    """
    Calculates log_probs of batch
    """
    mods = exp.modalities
    log_probs = dict()
    weighted_log_prob = 0.0
    for m, m_key in enumerate(mods.keys()):
        mod = mods[m_key]
        ba = batch[0][mod.name]
        if m_key == 'text' and exp.flags.text_encoding == 'word':
            ba = text.one_hot_encode_word(exp.flags, ba)
        log_probs[mod.name] = -mod.calc_log_prob(out_dist=result['rec'][mod.name], target=ba,
                                                 norm_value=exp.flags.batch_size)
        weighted_log_prob += exp.rec_weights[mod.name] * log_probs[mod.name]
    return log_probs, weighted_log_prob


def calc_klds(exp, result):
    latents = result['latents']['subsets']
    klds = dict()
    for m, key in enumerate(latents.keys()):
        mu, logvar = latents[key]
        klds[key] = calc_kl_divergence(mu, logvar,
                                       norm_value=exp.flags.batch_size)
    return klds


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

    mods = exp.modalities
    for k, m_key in enumerate(batch_d.keys()):
        batch_d[m_key] = Variable(batch_d[m_key]).to(exp.flags.device)
    results = mm_vae(batch_d)
    # checking if the latents contain NaNs. If they do raise Value error and the experiment is started again
    for key in results['latents']['modalities']:
        if not exp.flags.dataset == 'testing' and (np.isnan(results['latents']['modalities'][key][0].mean().item())
                                                   or
                                                   np.isnan(results['latents']['modalities'][key][1].mean().item())):
            print(key, results['latents']['modalities'][key][0].mean().item())
            print(key, results['latents']['modalities'][key][1].mean().item())
            raise NaNInLatent(f'The latent representations contain NaNs: {results["latents"]}')

        results['latents']['modalities'][key][1].mean().item()

    # getting the log probabilities
    log_probs, weighted_log_prob = calc_log_probs(exp, results, batch)
    group_divergence = results['joint_divergence']

    klds = calc_klds(exp, results)
    if exp.flags.factorized_representation:
        klds_style = calc_klds_style(exp, results)

    # Calculation of the loss
    if (exp.flags.modality_jsd or exp.flags.modality_moe
            or exp.flags.joint_elbo):
        if exp.flags.factorized_representation:
            kld_style = calc_style_kld(exp, klds_style)
        else:
            kld_style = 0.0
        kld_content = group_divergence
        kld_weighted = beta_style * kld_style + beta_content * kld_content
        total_loss = rec_weight * weighted_log_prob + beta * kld_weighted
    elif exp.flags.modality_poe:
        klds_joint = {'content': group_divergence,
                      'style': dict()}
        recs_joint = dict()
        elbos = dict()
        for m, m_key in enumerate(mods.keys()):
            mod = mods[m_key]
            if exp.flags.factorized_representation:
                kld_style_m = klds_style[m_key + '_style']
            else:
                kld_style_m = 0.0
            klds_joint['style'][m_key] = kld_style_m
            i_batch_mod = {m_key: batch_d[m_key]}
            r_mod = mm_vae(i_batch_mod)
            log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                              batch_d[m_key],
                                              exp.flags.batch_size)
            log_prob = {m_key: log_prob_mod}
            klds_mod = {'content': klds[m_key],
                        'style': {m_key: kld_style_m}}
            elbo_mod = utils.calc_elbo(exp, m_key, log_prob, klds_mod)
            elbos[m_key] = elbo_mod
        elbo_joint = utils.calc_elbo(exp, 'joint', log_probs, klds_joint)
        elbos['joint'] = elbo_joint
        total_loss = sum(elbos.values())

    out_basic_routine = dict()
    out_basic_routine['results'] = results
    out_basic_routine['log_probs'] = log_probs
    out_basic_routine['total_loss'] = total_loss
    out_basic_routine['klds'] = klds
    return out_basic_routine


def train(exp: MimicExperiment, train_loader: DataLoader):
    tb_logger = exp.tb_logger
    mm_vae = exp.mm_vae
    mm_vae.train()
    exp.mm_vae = mm_vae

    if 0 < exp.flags.steps_per_training_epoch < len(train_loader):
        training_steps = exp.flags.steps_per_training_epoch
    else:
        training_steps = len(train_loader)

    for iteration, batch in tqdm(enumerate(train_loader), total=training_steps, postfix='train'):
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
        if tb_logger:
            tb_logger.write_training_logs(results, total_loss, log_probs, klds)


def test(epoch, exp, test_loader: DataLoader):
    tb_logger = exp.tb_logger
    print(tb_logger)
    args = exp.flags
    with torch.no_grad():
        mm_vae = exp.mm_vae
        mm_vae.eval()
        exp.mm_vae = mm_vae

        total_losses = []
        for iteration, batch in tqdm(enumerate(test_loader), total=len(test_loader), postfix='test'):
            basic_routine = basic_routine_epoch(exp, batch)
            results = basic_routine['results']
            total_loss = basic_routine['total_loss']
            klds = basic_routine['klds']
            log_probs = basic_routine['log_probs']
            if tb_logger:
                tb_logger.write_testing_logs(results, total_loss, log_probs, klds)
            total_losses.append(total_loss.item())

        if epoch >= np.ceil(exp.flags.end_epoch * 0.8):
            if tb_logger:
                print('generating plots')
                plots = generate_plots(exp, epoch)
                tb_logger.write_plots(plots, epoch)

        if (epoch + 1) % exp.flags.eval_freq == 0 or (epoch + 1) == exp.flags.end_epoch:
            if exp.flags.eval_lr:
                if tb_logger:
                    print('evaluation of latent representation')
                    clf_lr = train_clf_lr_all_subsets(exp)
                    lr_eval = test_clf_lr_all_subsets(epoch, clf_lr, exp)
                    tb_logger.write_lr_eval(lr_eval)

            if exp.flags.use_clf:
                if tb_logger:
                    print('test generation')
                    gen_eval = test_generation(epoch, exp)
                    tb_logger.write_coherence_logs(gen_eval)

            if exp.flags.calc_nll:
                if tb_logger:
                    print('estimating likelihoods')
                    lhoods = estimate_likelihoods(exp)
                    tb_logger.write_lhood_logs(lhoods)

            if exp.flags.calc_prd and ((epoch + 1) % exp.flags.eval_freq_fid == 0):
                if tb_logger:
                    print('calculating prediction score')
                    prd_scores = calc_prd_score(exp)
                    tb_logger.write_prd_scores(prd_scores)
        mean_loss = np.mean(total_losses)
        if tb_logger:
            exp.update_experiments_dataframe({'total_test_loss': np.mean(total_losses), 'total_epochs': epoch})
        return mean_loss


def run_epochs(rank: any, exp: MimicExperiment) -> None:
    """
    rank: is int if multiprocessing and torch.device otherwise
    """
    print('running epochs')
    exp.mm_vae = exp.mm_vae.to(rank)
    args = exp.flags
    args.device = rank

    if not exp.flags.distributed or rank % exp.flags.world_size == 0:
        # set up logger only for one process
        exp.tb_logger = exp.init_summary_writer()
    if args.distributed:
        utils.set_up_process_group(args.world_size, rank)
        exp.mm_vae = DDP(exp.mm_vae, device_ids=[exp.flags.device])

    train_sampler, train_loader = get_data_loaders(args, exp.dataset_train)
    test_sampler, test_loader = get_data_loaders(args, exp.dataset_test)

    callbacks = Callbacks(exp)

    end = time.time()
    for epoch in tqdm(range(exp.flags.start_epoch, exp.flags.end_epoch), postfix='epochs'):
        end = time.time()
        samplers_set_epoch(args, train_sampler, test_sampler, epoch)
        # one epoch of training and testing
        train(exp, train_loader)
        mean_eval_loss = test(epoch, exp, test_loader)

        if callbacks.update_epoch(epoch, mean_eval_loss, time.time() - end):
            break

    if exp.tb_logger:
        exp.tb_logger.writer.close()
    if args.distributed:
        dist.destroy_process_group()
