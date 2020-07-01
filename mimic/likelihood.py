
import numpy as np
from scipy.special import logsumexp
from itertools import cycle

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import utils
from utils.likelihood import log_mean_exp
from utils.likelihood import gaussian_log_pdf
from utils.likelihood import unit_gaussian_log_pdf
from utils.likelihood import get_latent_samples
from utils.likelihood import get_dyn_prior
from utils.likelihood import log_marginal_estimate
from utils.likelihood import log_joint_estimate

from divergence_measures.mm_div import alpha_poe

from mimic.constants_mimic import MODS


def calc_log_likelihood_batch(flags, mod, batch, model, mod_weights, num_imp_samples=10):
    pa_batch, lateral_batch, text_batch, labels_batch = batch;
    num_samples_batch, p1, p2, p3 = pa_batch.shape;
    num_samples_batch, l1, l2, l3 = lateral_batch.shape;
    num_samples_batch, t1, t2 = text_batch.shape;
    #TODO: add permutation of samples in batch
    num_samples = num_samples_batch*num_imp_samples;
    pa_batch = pa_batch.unsqueeze(0).repeat(num_imp_samples, 1, 1, 1, 1);
    lateral_batch = lateral_batch.unsqueeze(0).repeat(num_imp_samples, 1, 1, 1, 1);
    text_batch = text_batch.unsqueeze(0).repeat(num_imp_samples, 1, 1, 1);
    pa_batch = pa_batch.view(num_samples, p1, p2, p3);
    lateral_batch = lateral_batch.view(num_samples, l1, l2, l3);
    text_batch = text_batch.view(num_samples, t1, t2);
    pa_batch = pa_batch.to(flags.device);
    lateral_batch = lateral_batch.to(flags.device);
    text_batch = text_batch.to(flags.device);
    batch_joint = {'pa': pa_batch, 'lateral': lateral_batch, 'text': text_batch}
    if mod == 'pa':
        i_pa = pa_batch;
        i_lateral = None;
        i_text = None;
    elif mod == 'lateral':
        i_pa = None;
        i_lateral = lateral_batch;
        i_text = None;
    elif mod == 'text':
        i_pa = None;
        i_lateral = None;
        i_text = text_batch;
    elif mod == 'pa_lateral':
        i_pa = pa_batch;
        i_lateral = lateral_batch;
        i_text = None;
    elif mod == 'lateral_text':
        i_pa = None;
        i_lateral = lateral_batch;
        i_text = text_batch;
    elif mod == 'pa_text':
        i_pa = pa_batch;
        i_lateral = None;
        i_text = text_batch;
    elif mod == 'joint':
        i_pa = pa_batch;
        i_lateral = lateral_batch;
        i_text = text_batch;

    mod_names = MODS.keys();
    latents = model.inference(input_pa=i_pa,
                              input_lateral=i_lateral,
                              input_text=i_text);

    c_mu, c_logvar = latents['joint'];
    style = dict();
    random_styles = model.get_random_style_dists(flags.batch_size*num_imp_samples);
    if flags.factorized_representation:
        for k, key in enumerate(mod_names):
            if latents[key][0] is None and latents[key][1] is None:
                style[key] = random_styles[key];
            else:
                style[key] = latents[key][:2];
    else:
        style = None;

    l_mod = {'content': [c_mu, c_logvar], 'style': style};
    l = get_latent_samples(flags, l_mod, mod_names);
    dyn_prior = None;
    use_pa_style = False;
    use_lateral_style = False;
    use_text_style = False;
    if mod == 'pa':
        use_pa_style = True;
    elif mod == 'lateral':
        use_lateral_style = True;
    elif mod == 'text':
        use_text_style = True;
    else:
        if flags.modality_jsd:
            dyn_prior = get_dyn_prior(latents['weights'],
                                      latents['mus'],
                                      latents['logvars'])
        if mod == 'pa_lateral':
            use_pa_style = True;
            use_lateral_style = True;
        elif mod == 'pa_text':
            use_pa_style = True;
            use_text_style = True;
        elif mod == 'lateral_text':
            use_lateral_style = True;
            use_text_style = True;
        elif mod == 'joint':
            use_pa_style = True;
            use_lateral_style = True;
            use_text_style = True;

    p = l['style']['pa'];
    l = l['style']['lateral'];
    c = l['content'];
    c_z_k = c['z'];
    if flags.factorized_representation:
        p_z_k = m['z'];
        l_z_k = s['z'];
        style_z_p = p_z_k.view(flags.batch_size * num_imp_samples, -1);
        style_z_l = l_z_k.view(flags.batch_size * num_imp_samples, -1);
    else:
        style_z_p = None;
        style_z_l = None;

    style_marg = {'pa': style_z_p, 'lateral': style_z_p};
    if len(mod_weights) > 3:
        t = l['style']['text'];
        if flags.factorized_representation:
            style_z_t = t['z'].view(flags.batch_size*num_imp_samples, -1);
        else:
            style_z_t = None;
        style_marg = {'pa': style_z_p, 'lateral': style_z_p, 'text': style_z_t};

    z_content = c_z_k.view(num_samples, -1);
    latents_dec = {'content': z_content, 'style': style_marg};
    gen = model.generate_sufficient_statistics_from_latents(latents_dec);
    suff_stats_pa = gen['pa'];
    suff_stats_lateral = gen['lateral'];

    # compute marginal log-likelihood
    if use_pa_style:
        ll_pa = log_marginal_estimate(flags, num_imp_samples, gen['pa'], pa_batch, p, c)
    else:
        ll_pa = log_marginal_estimate(flags, num_imp_samples, gen['pa'], pa_batch, None, c)

    if use_lateral_style:
        ll_lateral = log_marginal_estimate(flags, num_imp_samples, gen['lateral'], lateral_batch, l, c)
    else:
        ll_lateral = log_marginal_estimate(flags, num_imp_samples, gen['lateral'], lateral_batch, None, c)
    ll = {'pa': ll_pa, 'lateral': ll_lateral};
    if len(mod_weights) > 3:
        if use_text_style:
            ll['text'] = log_marginal_estimate(flags, num_imp_samples, gen['text'], text_batch, t, c)
        else:
            ll['text'] = log_marginal_estimate(flags, num_imp_samples, gen['text'], text_batch, None, c)
    ll_joint = log_joint_estimate(flags, num_imp_samples, gen, batch_joint, l['style'], c);
    ll['joint'] = ll_joint;
    return ll;


