import typing
from contextlib import contextmanager

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from mimic import log
from mimic.utils.exceptions import CudaOutOfMemory
from mimic.utils.save_samples import save_generated_samples_singlegroup
from mimic.utils.utils import at_most_n, dict_to_device
from itertools import product
from typing import Mapping, Union


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


def classify_cond_gen_samples_(exp, labels: Tensor, cond_samples: typing.Mapping[str, Tensor]) \
        -> typing.Mapping[str, typing.Mapping[str, float]]:
    labels = np.reshape(labels, (labels.shape[0], len(exp.labels)))
    clfs = exp.clfs
    eval_labels = {l_key: {} for l_key in exp.labels}
    for key in clfs.keys():
        if key in cond_samples.keys():
            mod_cond_gen = cond_samples[key]
            mod_clf = clfs[key]
            mod_cond_gen = transform_gen_samples(mod_cond_gen, exp.clf_transforms[key]).to(exp.flags.device)
            # classify generated sample to evaluate coherence
            attr_hat = mod_clf(mod_cond_gen)
            for l, label_str in enumerate(exp.labels):
                if exp.flags.dataset == 'testing':
                    # when using the testing dataset, the vae attr_hat might contain nans.
                    # Replace them for testing purposes
                    score = exp.eval_label(np.nan_to_num(attr_hat.cpu().data.numpy()), labels,
                                           index=l)
                else:
                    # score is nan if it only contains one class.
                    score = exp.eval_label(attr_hat.cpu().data.numpy(), labels, index=l)
                eval_labels[label_str][key] = score
        else:
            print(str(key) + 'not existing in cond_gen_samples')
    return eval_labels


def classify_cond_gen_samples(exp, labels: Tensor, cond_samples: typing.Mapping[str, Tensor]) \
        -> typing.Mapping[str, Tensor]:
    """
    Classifies for each modality all the conditionally generated samples.
    Returns a dict like the following:
    {'PA': tensor,
    'Lateral': tensor,
    'text': tensor}

    """
    clfs = exp.clfs
    clf_predictions = {mod: {} for mod in exp.modalities}
    for mod in exp.modalities:
        if mod in cond_samples:
            mod_cond_gen: Tensor = cond_samples[mod]
            mod_clf = clfs[mod]
            mod_cond_gen = transform_gen_samples(mod_cond_gen, exp.clf_transforms[mod]).to(exp.flags.device)
            # classify generated sample to evaluate coherence
            clf_predictions[mod] = mod_clf(mod_cond_gen).cpu()
        else:
            log.info(str(mod) + 'not existing in cond_gen_samples')
    return clf_predictions


def calculate_coherence(exp, samples) -> dict:
    """
    Classifies generated modalities. The generated samples are coherent if all modalities
    are classified as belonging to the same class.
    """
    clfs = exp.clfs
    mods = exp.modalities
    # TODO: make work for num samples NOT EQUAL to batch_size
    c_labels = {}
    for j, l_key in enumerate(exp.labels):
        pred_mods = np.zeros((len(mods.keys()), exp.flags.batch_size))
        for k, m_key in enumerate(mods.keys()):
            mod = mods[m_key]
            clf_mod = clfs[mod.name].to(exp.flags.device)
            samples_mod = samples[mod.name]
            samples_mod = transform_gen_samples(samples_mod, exp.clf_transforms[m_key])
            samples_mod = samples_mod.to(exp.flags.device)

            attr_mod = clf_mod(samples_mod)
            output_prob_mod = attr_mod.cpu().data.numpy()
            pred_mod = np.argmax(output_prob_mod, axis=1).astype(int)
            pred_mods[k, :] = pred_mod
        coh_mods = np.all(pred_mods == pred_mods[0, :], axis=0)
        coherence = np.sum(coh_mods.astype(int)) / float(exp.flags.batch_size)
        c_labels[l_key] = coherence
    return c_labels


def transform_gen_samples(gen_samples, transform):
    """
    transforms the generated samples as needed for the classifier
    """

    transformed_samples = [
        transform(gen_samples[idx].cpu())
        for idx in range(gen_samples.shape[0])
    ]

    return torch.stack(transformed_samples)


def save_generated_samples(exp, rand_gen: dict, iteration: int, batch_d: dict) -> None:
    """
    Saves generated samples to dir_fid
    """
    save_generated_samples_singlegroup(exp, iteration, 'random', rand_gen)
    if exp.flags.text_encoding == 'word':
        batch_d_temp = batch_d.copy()
        batch_d_temp['text'] = torch.nn.functional.one_hot(batch_d_temp['text'].to(torch.int64),
                                                           num_classes=exp.flags.vocab_size)

        save_generated_samples_singlegroup(exp, iteration,
                                           'real',
                                           batch_d_temp)
    else:
        save_generated_samples_singlegroup(exp, iteration,
                                           'real',
                                           batch_d)


def init_gen_perf(labels, subsets, mods) -> typing.Mapping[str, dict]:
    """
    Initialises gen_perf dict with empty iterables.
    The result will look like this:

    {'cond':
        {'Lung Opacity':
            {'PA': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral_PA': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'PA_text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral_text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral_PA_text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()}},
      'Pleural Effusion':
          {'PA': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral_PA': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'PA_text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral_text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral_PA_text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()}},
      'Support Devices':
          {'PA': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral_PA': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'PA_text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral_text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral_PA_text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()}}},

    'random':
        {'Lung Opacity': Tensor(), 'Pleural Effusion': Tensor(), 'Support Devices': Tensor()}}
    """
    return {'cond': init_twolevel_nested_dict(labels, subsets, init_val={mod: torch.Tensor() for mod in mods}),
            'random': {k: [] for k in labels}}


def calc_coherence_random_gen(exp, mm_vae, iteration: int, gen_perf: typing.Mapping[str, dict], batch_d: dict) -> \
        typing.Mapping[str, dict]:
    args = exp.flags
    # generating random samples
    with catching_cuda_out_of_memory(batch_size=args.batch_size):
        rand_gen = mm_vae.module.generate() if args.distributed else mm_vae.generate()
    rand_gen = {k: v.to(args.device) for k, v in rand_gen.items()}
    # classifying generated examples
    coherence_random = calculate_coherence(exp, rand_gen)
    for j, l_key in enumerate(exp.labels):
        gen_perf['random'][l_key].append(coherence_random[l_key])

    if (exp.flags.batch_size * iteration) < exp.flags.num_samples_fid:
        # saving generated samples to dir_fid
        save_generated_samples(exp, rand_gen, iteration, batch_d)

    return gen_perf


def test_generation_(epoch, exp):
    args = exp.flags

    mods = exp.modalities
    mm_vae = exp.mm_vae
    subsets = exp.subsets

    gen_perf = init_gen_perf(exp)
    old_batch_size = args.batch_size
    args.batch_size = 5
    log.info(f'setting batch size to {args.batch_size}')
    d_loader = DataLoader(exp.dataset_test,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=exp.flags.dataloader_workers, drop_last=True)

    total_steps = None if args.steps_per_training_epoch < 0 else args.steps_per_training_epoch
    # temp
    total_steps = 1
    for iteration, batch in tqdm(enumerate(at_most_n(d_loader, total_steps)),
                                 total=len(d_loader), postfix='test_generation'):
        batch_d = batch[0]
        batch_l = batch[1]
        # generating random samples
        with catching_cuda_out_of_memory(batch_size=args.batch_size):
            rand_gen = mm_vae.module.generate() if args.distributed else mm_vae.generate()
        rand_gen = {k: v.to(args.device) for k, v in rand_gen.items()}
        # classifying generated examples
        coherence_random = calculate_coherence(exp, rand_gen)
        for j, l_key in enumerate(exp.labels):
            gen_perf['random'][l_key].append(coherence_random[l_key])

        if (exp.flags.batch_size * iteration) < exp.flags.num_samples_fid:
            # saving generated samples to dir_fid
            save_generated_samples(exp, rand_gen, iteration, batch_d)

        for k, m_key in enumerate(batch_d.keys()):
            batch_d[m_key] = batch_d[m_key].to(exp.flags.device)

        inferred = mm_vae.module.inference(batch_d) if args.distributed else mm_vae.inference(batch_d)
        lr_subsets = inferred['subsets']
        cg = mm_vae.module.cond_generation(lr_subsets) if args.distributed else mm_vae.cond_generation(lr_subsets)

        for k, s_key in enumerate(cg.keys()):
            clf_cg = classify_cond_gen_samples(exp, batch_l, cg[s_key])
            for j, l_key in enumerate(exp.labels):
                for m, m_key in enumerate(mods.keys()):
                    gen_perf['cond'][l_key][s_key][m_key].append(clf_cg[l_key][m_key])
            if (exp.flags.batch_size * iteration) < exp.flags.num_samples_fid:
                save_generated_samples_singlegroup(exp, iteration, s_key, cg[s_key])

    for j, l_key in enumerate(exp.labels):
        for k, s_key in enumerate(subsets.keys()):
            if s_key != '':
                for l, m_key in enumerate(mods.keys()):
                    perf = exp.mean_eval_metric(gen_perf['cond'][l_key][s_key][m_key])
                    gen_perf['cond'][l_key][s_key][m_key] = perf
                    if np.isnan(perf):
                        log.warning(f'mean coherence eval metric "cond" is nan for cond {l_key},{s_key},{m_key}')
                        log.debug(
                            f'mean coherence eval metric "cond" is nan for cond {l_key},{s_key},{m_key}. '
                            f'With coherence gen perf:\n {gen_perf["cond"][l_key][s_key][m_key]}')
        eval_score = exp.mean_eval_metric(gen_perf['random'][l_key])
        if np.isnan(eval_score):
            log.warning(f'mean coherence eval metric is nan for the label {l_key}')
            log.debug(
                f'mean coherence eval metric for the label {l_key} is nan. '
                f'With coherence eval metric:\n {gen_perf["random"][l_key]}')
        gen_perf['random'][l_key] = eval_score

    args.batch_size = old_batch_size
    return gen_perf


def init_twolevel_nested_dict(level1_keys, level2_keys, init_val: any) -> dict:
    """
    Initialises a 2 level nested dict with value: init_val
    HK, 15.12.20
    """
    return {l1: {l2: init_val for l2 in level2_keys if l2} for l1 in level1_keys if l1}


def eval_classified_gen_samples(exp, subsets, mods, cond_gen_classified, gen_perf, batch_labels):
    """
    HK, 15.12.20
    """
    # compare the classification on the generated samples with the ground truth
    for l_idx, l_key in enumerate(exp.labels):
        for s_key, m_key in product(subsets, mods):
            perf = exp.mean_eval_metric(
                exp.eval_label(cond_gen_classified[s_key][m_key].cpu().data.numpy(), batch_labels, l_idx))
            gen_perf['cond'][l_key][s_key][m_key] = perf
            if np.isnan(perf):
                log.warning(f'mean coherence eval metric "cond" is nan for cond {l_key},{s_key},{m_key}')
                log.debug(
                    f'mean coherence eval metric "cond" is nan for cond {l_key},{s_key},{m_key}. '
                    f'With coherence gen perf:\n {gen_perf["cond"][l_key][s_key][m_key]}')
        eval_score = exp.mean_eval_metric(gen_perf['random'][l_key])
        if np.isnan(eval_score):
            log.warning(f'mean coherence eval metric is nan for the label {l_key}')
            log.debug(
                f'mean coherence eval metric for the label {l_key} is nan. '
                f'With coherence eval metric:\n {gen_perf["random"][l_key]}')
        gen_perf['random'][l_key] = eval_score
    return gen_perf


def test_generation(epoch, exp):
    """
    Generates random and conditioned samples and evaluates coherence.
    """
    args = exp.flags

    mods = exp.modalities
    mm_vae = exp.mm_vae
    subsets = exp.subsets
    del subsets['']
    labels = exp.labels

    gen_perf = init_gen_perf(labels, subsets, mods)
    d_loader = DataLoader(exp.dataset_test,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=exp.flags.dataloader_workers, drop_last=True)

    total_steps = None if args.steps_per_training_epoch < 0 else args.steps_per_training_epoch
    # all labels accumulated over batches:
    batch_labels = torch.Tensor()
    cond_gen_classified = init_twolevel_nested_dict(subsets, mods, init_val=torch.Tensor())
    cond_gen_classified: Mapping[subsets, Mapping[mods, Tensor]]

    for iteration, (batch_d, batch_l) in tqdm(enumerate(at_most_n(d_loader, total_steps)),
                                              total=len(d_loader), postfix='test_generation'):

        batch_labels = torch.cat((batch_labels, batch_l), 0)
        batch_d = dict_to_device(batch_d, exp.flags.device)
        # evaluating random generation
        gen_perf = calc_coherence_random_gen(exp, mm_vae, iteration, gen_perf, batch_d)

        # evaluating conditional generation
        # first generates the conditional gen_samples
        # classifies them and stores the classifier predictions
        inferred = mm_vae.module.inference(batch_d) if args.distributed else mm_vae.inference(batch_d)
        lr_subsets = inferred['subsets']
        cg = mm_vae.module.cond_generation(lr_subsets) if args.distributed else mm_vae.cond_generation(lr_subsets)
        cg: typing.Mapping[subsets, typing.Mapping[mods, Tensor]]

        # classify the cond. generated samples
        for cond_key, cond_val in cg.items():
            clf_cg: Mapping[mods, Tensor] = classify_cond_gen_samples(exp, batch_l, cond_val)
            for mod in mods:
                cond_gen_classified[cond_key][mod] = torch.cat((cond_gen_classified[cond_key][mod], clf_cg[mod]), 0)
            if (exp.flags.batch_size * iteration) < exp.flags.num_samples_fid:
                save_generated_samples_singlegroup(exp, iteration, cond_key, cond_val)

    gen_perf['cond']: typing.Mapping[str, typing.Mapping[str, typing.Mapping[str, float]]]
    return eval_classified_gen_samples(exp, subsets, mods, cond_gen_classified, gen_perf, batch_labels)
