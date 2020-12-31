# HK, 31.12.20
from evaluation.divergence_measures.kl_div import calc_kl_divergence
from utils import utils


def calc_log_probs(exp, result, batch):
    """
    Calculates log_probs of batch
    """
    mods = exp.modalities
    log_probs = {}
    weighted_log_prob = 0.0
    for m, m_key in enumerate(mods.keys()):
        mod = mods[m_key]
        ba = batch[0][mod.name]

        log_probs[mod.name] = -mod.calc_log_prob(out_dist=result['rec'][mod.name], target=ba,
                                                 norm_value=exp.flags.batch_size)
        weighted_log_prob += exp.rec_weights[mod.name] * log_probs[mod.name]
    return log_probs, weighted_log_prob


def calc_klds(exp, result):
    latents = result['latents']['subsets']
    klds = {}
    for m, key in enumerate(latents.keys()):
        mu, logvar = latents[key]
        klds[key] = calc_kl_divergence(mu, logvar,
                                       norm_value=exp.flags.batch_size)
    return klds


def calc_klds_style(exp, result):
    latents = result['latents']['modalities']
    klds = {}
    for key in latents.keys():
        if key.endswith('style'):
            mu, logvar = latents[key]
            klds[key] = calc_kl_divergence(mu, logvar,
                                           norm_value=exp.flags.batch_size)
    return klds


def calc_style_kld(exp, klds):
    mods = exp.modalities
    style_weights = exp.style_weights
    weighted_klds = 0.0
    for m_key in mods.keys():
        weighted_klds += style_weights[m_key] * klds[m_key + '_style']
    return weighted_klds


def calc_poe_loss(exp, mods, group_divergence, klds, klds_style, batch_d, mm_vae, log_probs):
    klds_joint = {'content': group_divergence,
                  'style': {}}
    elbos = {}
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
    return sum(elbos.values())


def calc_joint_elbo_loss(exp, klds_style, group_divergence, beta_style, beta_content, weighted_log_prob, beta):
    if exp.flags.factorized_representation:
        kld_style = calc_style_kld(exp, klds_style)
    else:
        kld_style = 0.0
    kld_content = group_divergence
    kld_weighted = beta_style * kld_style + beta_content * kld_content
    rec_weight = 1.0

    return rec_weight * weighted_log_prob + beta * kld_weighted