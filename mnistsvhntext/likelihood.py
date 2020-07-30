
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
from utils.constants_svhnmnist import indices as IND3
from utils.constants_svhnmnist_img import indices as IND2
from divergence_measures.mm_div import alpha_poe
from divergence_measures.mm_div import poe

LOG2PI = float(np.log(2.0 * math.pi))



#at the moment: only marginals and joint
def calc_log_likelihood_batch(flags, mod, batch, model, mod_weights, num_imp_samples=10):
    mnist_batch, svhn_batch, text_batch, labels_batch = batch;
    num_samples_batch, m1, m2, m3 = mnist_batch.shape;
    num_samples_batch, s1, s2, s3 = svhn_batch.shape;
    num_samples_batch, t1, t2 = text_batch.shape;
    #TODO: add permutation of samples in batch
    num_samples = num_samples_batch*num_imp_samples;
    mnist_batch = mnist_batch.unsqueeze(0).repeat(num_imp_samples, 1, 1, 1, 1);
    svhn_batch = svhn_batch.unsqueeze(0).repeat(num_imp_samples, 1, 1, 1, 1);
    text_batch = text_batch.unsqueeze(0).repeat(num_imp_samples, 1, 1, 1);
    mnist_batch = mnist_batch.view(num_samples, m1, m2, m3);
    svhn_batch = svhn_batch.view(num_samples, s1, s2, s3);
    text_batch = text_batch.view(num_samples, t1, t2);
    mnist_batch = mnist_batch.to(flags.device);
    svhn_batch = svhn_batch.to(flags.device);
    text_batch = text_batch.to(flags.device);
    batch_joint = {'img_mnist': mnist_batch, 'img_svhn': svhn_batch, 'text': text_batch}
    if mod == 'img_mnist':
        i_mnist = mnist_batch;
        i_svhn = None;
        i_text = None;
    elif mod == 'img_svhn':
        i_mnist = None;
        i_svhn = svhn_batch;
        i_text = None;
    elif mod == 'text':
        i_mnist = None;
        i_svhn = None;
        i_text = text_batch;
    elif mod == 'mnist_svhn' or mod == 'mnist_svhn_dp':
        i_mnist = mnist_batch;
        i_svhn = svhn_batch;
        i_text = None;
    elif mod == 'svhn_text' or mod == 'svhn_text_dp':
        i_mnist = None;
        i_svhn = svhn_batch;
        i_text = text_batch;
    elif mod == 'mnist_text' or mod == 'mnist_text_dp':
        i_mnist = mnist_batch;
        i_svhn = None;
        i_text = text_batch;
    elif mod == 'joint':
        i_mnist = mnist_batch;
        i_svhn = svhn_batch;
        i_text = text_batch;

    if model.num_modalities == 2:
        mod_names = IND2.keys();
        latents = model.inference(input_mnist=i_mnist, input_svhn=i_svhn);
    else:
        mod_names = IND3.keys();
        latents = model.inference(input_mnist=i_mnist,
                                  input_svhn=i_svhn,
                                  input_text=i_text);

    c_mu, c_logvar = latents[mod];
    if mod == 'joint':
        c_mu, c_logvar = latents['mnist_svhn_text']
    if mod.endswith('dp'):
        c_mu, c_logvar = poe(latents['mus'], latents['logvars']);
    style = dict();
    random_styles = model.get_random_style_dists(flags.batch_size*num_imp_samples);
    if flags.factorized_representation:
        for k, key in enumerate(mod_names):
            if latents[key + '_style'][0] is None and latents[key + '_style'][1] is None:
                style[key] = random_styles[key];
            else:
                style[key] = latents[key + '_style'];
    else:
        style = None;

    l_mod = {'content': [c_mu, c_logvar], 'style': style};
    l = get_latent_samples(flags, l_mod, mod_names);
    dyn_prior = None;
    use_mnist_style = False;
    use_svhn_style = False;
    use_text_style = False;
    if mod == 'img_mnist':
        use_mnist_style = True;
    elif mod == 'img_svhn':
        use_svhn_style = True;
    elif mod == 'text':
        use_text_style = True;
    else:
        if flags.modality_jsd:
            dyn_prior = get_dyn_prior(latents['weights'],
                                      latents['mus'],
                                      latents['logvars'])
        if mod == 'mnist_svhn' or mod == 'mnist_svhn_dp':
            use_mnist_style = True;
            use_svhn_style = True;
        elif mod == 'mnist_text' or mod == 'mnist_text_dp':
            use_mnist_style = True;
            use_text_style = True;
        elif mod == 'svhn_text' or mod == 'svhn_text_dp':
            use_svhn_style = True;
            use_text_style = True;
        elif mod == 'joint':
            use_mnist_style = True;
            use_svhn_style = True;
            use_text_style = True;

    m = l['style']['img_mnist'];
    s = l['style']['img_svhn'];
    c = l['content'];
    c_z_k = c['z'];
    if flags.factorized_representation:
        m_z_k = m['z'];
        s_z_k = s['z'];
        style_z_m = m_z_k.view(flags.batch_size * num_imp_samples, -1);
        style_z_s = s_z_k.view(flags.batch_size * num_imp_samples, -1);
    else:
        style_z_m = None;
        style_z_s = None;

    style_marg = {'img_mnist': style_z_m, 'img_svhn': style_z_s};
    if len(mod_weights) > 3:
        t = l['style']['text'];
        if flags.factorized_representation:
            style_z_t = t['z'].view(flags.batch_size*num_imp_samples, -1);
        else:
            style_z_t = None;
        style_marg = {'img_mnist': style_z_m, 'img_svhn': style_z_s, 'text': style_z_t};

    z_content = c_z_k.view(num_samples, -1);
    latents_dec = {'content': z_content, 'style': style_marg};
    gen = model.generate_sufficient_statistics_from_latents(latents_dec);
    suff_stats_mnist = gen['img_mnist'];
    suff_stats_svhn = gen['img_svhn'];

    # compute marginal log-likelihood
    if use_mnist_style:
        ll_mnist = log_marginal_estimate(flags, num_imp_samples, gen['img_mnist'], mnist_batch, m, c)
    else:
        ll_mnist = log_marginal_estimate(flags, num_imp_samples, gen['img_mnist'], mnist_batch, None, c)

    if use_svhn_style:
        ll_svhn = log_marginal_estimate(flags, num_imp_samples, gen['img_svhn'], svhn_batch, s, c)
    else:
        ll_svhn = log_marginal_estimate(flags, num_imp_samples, gen['img_svhn'], svhn_batch, None, c)
    ll = {'img_mnist': ll_mnist, 'img_svhn': ll_svhn};
    if len(mod_weights) > 3:
        if use_text_style:
            ll['text'] = log_marginal_estimate(flags, num_imp_samples, gen['text'], text_batch, t, c)
        else:
            ll['text'] = log_marginal_estimate(flags, num_imp_samples, gen['text'], text_batch, None, c)
    ll_joint = log_joint_estimate(flags, num_imp_samples, gen, batch_joint, l['style'], c);
    ll['joint'] = ll_joint;
    return ll;


