import random
import time
import typing
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
from termcolor import colored
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from mimic import log
from mimic.dataio.utils import get_data_loaders, samplers_set_epoch
from mimic.evaluation.eval_metrics.coherence import test_generation, flatten_cond_gen_values
from mimic.evaluation.eval_metrics.likelihood import estimate_likelihoods
from mimic.evaluation.eval_metrics.representation import test_clf_lr_all_subsets, train_clf_lr_all_subsets
from mimic.evaluation.eval_metrics.sample_quality import calc_prd_score
from mimic.evaluation.losses import calc_log_probs, calc_klds, calc_klds_style, calc_poe_loss, calc_joint_elbo_loss
from mimic.utils import utils
from mimic.utils.average_meters import AverageMeter, AverageMeterDict, AverageMeterLatents
from mimic.utils.exceptions import CudaOutOfMemory
from mimic.utils.experiment import Callbacks, MimicExperiment
from mimic.utils.plotting import generate_plots
from mimic.utils.utils import check_latents, at_most_n, get_items_from_dict


# set the seed for reproducibility
def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


@contextmanager
def catching_cuda_out_of_memory(batch_size):
    """
    Context that throws CudaOutOfMemory error if GPU is out of memory.
    """
    try:
        yield
    # if the GPU runs out of memory, start the experiment again with a smaller batch size
    except RuntimeError as e:
        if str(e).startswith('CUDA out of memory.') and batch_size > 10:
            raise CudaOutOfMemory(e)
        else:
            raise e


def basic_routine_epoch(exp, batch) -> typing.Mapping[str, any]:
    # set up weights
    beta_style = exp.flags.beta_style
    beta_content = exp.flags.beta_content
    beta = exp.flags.beta
    mm_vae = exp.mm_vae
    batch_d = batch[0]

    mods = exp.modalities
    for k, m_key in enumerate(batch_d.keys()):
        batch_d[m_key] = Variable(batch_d[m_key]).to(exp.flags.device)

    with catching_cuda_out_of_memory(batch_size=exp.flags.batch_size):
        results = mm_vae(batch_d)

    for key in results['latents']['modalities']:
        results['latents']['modalities'][key][1].mean().item()
        # checking if the latents contain NaNs. If they do raise NaNInLatent error and the experiment is started again
        check_latents(exp.flags, results['latents']['modalities'][key])

    # getting the log probabilities
    with catching_cuda_out_of_memory(batch_size=exp.flags.batch_size):
        log_probs, weighted_log_prob = calc_log_probs(exp, results, batch)
    group_divergence = results['joint_divergence']

    klds = calc_klds(exp, results)
    if exp.flags.factorized_representation:
        klds_style = calc_klds_style(exp, results)
    else:
        klds_style = None

    # Calculation of the loss
    if (exp.flags.modality_jsd or exp.flags.modality_moe
            or exp.flags.joint_elbo):
        total_loss = calc_joint_elbo_loss(exp, klds_style, group_divergence, beta_style, beta_content,
                                          weighted_log_prob, beta)
    elif exp.flags.modality_poe:
        total_loss = calc_poe_loss(exp, mods, group_divergence, klds, klds_style, batch_d, mm_vae, log_probs)

    return {
        'results': results,
        'log_probs': log_probs,
        'total_loss': total_loss,
        'klds': klds,
    }


def train(exp: MimicExperiment, train_loader: DataLoader) -> None:
    tb_logger = exp.tb_logger
    mm_vae = exp.mm_vae
    mm_vae.train()
    exp.mm_vae = mm_vae

    average_meters = {
        'total_loss': AverageMeter('total_test_loss'),
        'klds': AverageMeterDict('klds'),
        'log_probs': AverageMeterDict('log_probs'),
        'joint_divergence': AverageMeter('joint_divergence'),
        'latents': AverageMeterLatents('latents'),
    }

    if 0 < exp.flags.steps_per_training_epoch < len(train_loader):
        training_steps = exp.flags.steps_per_training_epoch
    else:
        training_steps = None

    for iteration, batch in tqdm(enumerate(at_most_n(train_loader, training_steps or None)),
                                 total=training_steps or len(train_loader), postfix='train'):
        # log text input once evey epoch:
        if iteration == 0:
            tb_logger.write_tensor_to_text(batch[0]['text'][0], exp, log_tag='train_input')
        basic_routine = basic_routine_epoch(exp, batch)
        results = basic_routine['results']
        total_loss = basic_routine['total_loss']

        # backprop
        exp.optimizer.zero_grad()
        with catching_cuda_out_of_memory(exp.flags.batch_size):
            total_loss.backward()
        exp.optimizer.step()

        batch_results = {
            'total_loss': total_loss.item(),
            'klds': get_items_from_dict(basic_routine['klds']),
            'log_probs': get_items_from_dict(basic_routine['log_probs']),
            'joint_divergence': results['joint_divergence'].item(),
            'latents': results['latents']['modalities'],
        }

        for key, value in batch_results.items():
            average_meters[key].update(value)

    epoch_averages = {k: v.get_average() for k, v in average_meters.items()}
    tb_logger.write_training_logs(**epoch_averages)


def test(epoch, exp, test_loader: DataLoader):
    with torch.no_grad():
        mm_vae = exp.mm_vae
        mm_vae.eval()
        exp.mm_vae = mm_vae

        average_meters = {
            'total_loss': AverageMeter('total_test_loss'),
            'klds': AverageMeterDict('klds'),
            'log_probs': AverageMeterDict('log_probs'),
            'joint_divergence': AverageMeter('joint_divergence'),
            'latents': AverageMeterLatents('latents'),
        }
        tb_logger = exp.tb_logger

        for iteration, batch in tqdm(enumerate(test_loader), total=len(test_loader), postfix='test'):
            basic_routine = basic_routine_epoch(exp, batch)
            results = basic_routine['results']
            batch_results = {
                'total_loss': basic_routine['total_loss'].item(),
                'klds': get_items_from_dict(basic_routine['klds']),
                'log_probs': get_items_from_dict(basic_routine['log_probs']),
                'joint_divergence': results['joint_divergence'].item(),
                'latents': results['latents']['modalities'],
            }

            for key in batch_results:
                average_meters[key].update(batch_results[key])

        klds: typing.Mapping[str, float]
        log_probs: typing.Mapping[str, float]
        joint_divergence: float
        latents: typing.Mapping[str, tuple]

        test_results = {k: v.get_average() for k, v in average_meters.items()}
        tb_logger.write_testing_logs(**test_results)

        # set a lower batch_size for testing to spare GPU memory
        log.info(f'setting batch size to {exp.flags.batch_size}')
        training_batch_size = exp.flags.batch_size
        exp.flags.batch_size = 30

        test_results['lr_eval'] = None
        if (epoch + 1) % exp.flags.eval_freq == 0 or (epoch + 1) == exp.flags.end_epoch:
            log.info('generating plots')
            plots = generate_plots(exp, epoch)
            tb_logger.write_plots(plots, epoch)

            if exp.flags.eval_lr:
                log.info('evaluation of latent representation')
                clf_lr = train_clf_lr_all_subsets(exp)
                lr_eval = test_clf_lr_all_subsets(clf_lr, exp)
                tb_logger.write_lr_eval(lr_eval)
                test_results['lr_eval'] = lr_eval

            if exp.flags.use_clf:
                log.info('test generation')
                gen_eval = test_generation(exp)
                tb_logger.write_coherence_logs(gen_eval)
                test_results['gen_eval'] = flatten_cond_gen_values(gen_eval)

            if exp.flags.calc_nll:
                log.info('estimating likelihoods')
                lhoods = estimate_likelihoods(exp)
                tb_logger.write_lhood_logs(lhoods)
                test_results['lhoods'] = lhoods

            if exp.flags.calc_prd and ((epoch + 1) % exp.flags.eval_freq_fid == 0):
                log.info('calculating prediction score')
                prd_scores = calc_prd_score(exp)
                tb_logger.write_prd_scores(prd_scores)
                test_results['prd_scores'] = prd_scores

        test_results['latents'] = {mod: {'mu': test_results['latents'][mod][0],
                                         'logvar': test_results['latents'][mod][1]} for mod in test_results['latents']}

        exp.update_experiments_dataframe({'total_epochs': epoch, **utils.flatten(test_results)})

        # setting batch size back to training batch size
        exp.flags.batch_size = training_batch_size
        return test_results['total_loss'], test_results['lr_eval']


def run_epochs(rank: any, exp: MimicExperiment) -> None:
    """
    rank: is int if multiprocessing and torch.device otherwise
    """
    log.info('running epochs')

    set_random_seed(exp.flags.seed)

    exp.set_optimizer()
    exp.mm_vae = exp.mm_vae.to(rank)
    args = exp.flags
    args.device = rank
    exp.tb_logger = exp.init_summary_writer()

    if args.distributed:
        utils.set_up_process_group(args.world_size, rank)
        exp.mm_vae = DDP(exp.mm_vae, device_ids=[exp.flags.device])

    train_sampler, train_loader = get_data_loaders(args, exp.dataset_train, which_set='train',
                                                   weighted_sampler=args.weighted_sampler)
    test_sampler, test_loader = get_data_loaders(args, exp.dataset_test, which_set='eval')

    callbacks = Callbacks(exp)

    end = time.time()
    for epoch in tqdm(range(exp.flags.start_epoch, exp.flags.end_epoch), postfix='epochs'):
        print(colored(f'\nEpoch {epoch} {"-" * 140}\n', 'green'))
        end = time.time()
        samplers_set_epoch(args, train_sampler, test_sampler, epoch)
        exp.tb_logger.set_epoch(epoch)
        # one epoch of training and testing
        train(exp, train_loader)
        mean_eval_loss, results_lr = test(epoch, exp, test_loader)

        if callbacks.update_epoch(epoch, mean_eval_loss, time.time() - end, results_lr):
            break

    if exp.tb_logger:
        exp.tb_logger.writer.close()
    if args.distributed:
        dist.destroy_process_group()
