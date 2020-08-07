
import numpy as np
from scipy.special import logsumexp
from itertools import cycle
import math

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
from mnistsvhntext.constants import indices as IND3
#from mnistsvhntext.constants_svhnmnist_img import indices as IND2
from divergence_measures.mm_div import alpha_poe
from divergence_measures.mm_div import poe

LOG2PI = float(np.log(2.0 * math.pi))



#at the moment: only marginals and joint
def calc_log_likelihood_batch(exp, latents, subset_key, subset, batch, num_imp_samples=10):
    flags = exp.flags;
    model = exp.mm_vae;
    mod_weights = exp.style_weights;
    mods = exp.modalities;
     
    s_dist = latents['subsets'][subset_key]
    n_total_samples = s_dist[0].shape[0]*num_imp_samples;

    style = dict();
    if flags.factorized_representation:
        enc_mods = latents['modalities'];
        style = model.get_random_style_dists(flags.batch_size);
        for m, mod in enumerate(subset):
            if (enc_mods[mod.name + '_style'][0] is not None
                and enc_mods[mod.name + '_style'][1] is not None):
                style[mod.name] = enc_mods[mod.name + '_style'];
    else:
        style = None;

    l_subset = {'content': s_dist, 'style': style};
    mod_names = mods.keys()
    l = get_latent_samples(flags, l_subset, num_imp_samples, mod_names);

    l_style_rep = l['style'];
    l_content_rep = l['content'];

    c = {'mu': l_content_rep['mu'].view(n_total_samples, -1),
         'logvar': l_content_rep['logvar'].view(n_total_samples, -1),
         'z': l_content_rep['z'].view(n_total_samples, -1)}
    l_lin_rep = {'content': c,
                 'style': dict()};
    for m, m_key in enumerate(l_style_rep.keys()):
        if flags.factorized_representation:
            s = {'mu': l_style_rep[mod.name]['mu'].view(n_total_samples, -1),
                 'logvar': l_style_rep[mod.name]['logvar'].view(n_total_samples, -1),
                 'z': l_style_rep[mod.name]['z'].view(n_total_samples, -1)}
            l_lin_rep['style'][m_key] = s;
        else:
            l_lin_rep['style'][m_key] = None;

    l_dec = {'content': l_lin_rep['content']['z'],
             'style': dict()};
    for m, m_key in enumerate(l_style_rep.keys()):
        if flags.factorized_representation:
            l_dec['style'][m_key] = l_lin_rep['style'][m_key]['z'];
        else:
            l_dec['style'][m_key] = None;

    gen = model.generate_sufficient_statistics_from_latents(l_dec);
    l_lin_rep_style = l_lin_rep['style'];
    l_lin_rep_content = l_lin_rep['content'];
    ll = dict();
    for k, m_key in enumerate(mods.keys()):
        mod = mods[m_key];
        # compute marginal log-likelihood
        if mod in subset:
            style_mod = l_lin_rep_style[mod.name];
        else:
            style_mod = None;
        ll_mod = log_marginal_estimate(flags,
                                       num_imp_samples,
                                       gen[mod.name],
                                       batch[mod.name],
                                       style_mod,
                                       l_lin_rep_content)
        ll[mod.name] = ll_mod;

    ll_joint = log_joint_estimate(flags, num_imp_samples, gen, batch,
                                  l_lin_rep_style,
                                  l_lin_rep_content);
    ll['joint'] = ll_joint;
    del gen
    del l_lin_rep
    del l
    del l_dec
    torch.cuda.empty_cache()
    return ll;



