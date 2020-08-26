import os

import torch
import torch.nn as nn

from mmnist.networks.ConvNetworksImgCMNIST import EncoderImg, DecoderImg

from divergence_measures.mm_div import calc_alphaJSD_modalities
from divergence_measures.mm_div import calc_group_divergence_poe
from divergence_measures.mm_div import calc_group_divergence_moe
from divergence_measures.mm_div import calc_kl_divergence
from divergence_measures.mm_div import poe

from utils import utils


class VAEMMNIST(nn.Module):
    def __init__(self, flags, modalities, subsets):
        super().__init__()
        self.num_modalities = len(modalities.keys())
        self.flags = flags
        self.modalities = modalities
        self.subsets = subsets
        self.encoders = [modalities["m%d" % m].encoder.to(flags.device) for m in range(self.num_modalities)]
        self.decoders = [modalities["m%d" % m].decoder.to(flags.device) for m in range(self.num_modalities)]
        self.likelihoods = [modalities["m%d" % m].likelihood for m in range(self.num_modalities)]

        weights = utils.reweight_weights(torch.Tensor(flags.alpha_modalities))
        self.weights = weights.to(flags.device)
        if flags.modality_moe or flags.modality_jsd:
            self.modality_fusion = self.moe_fusion
            if flags.modality_moe:
                self.calc_joint_divergence = self.divergence_moe
            if flags.modality_jsd:
                self.calc_joint_divergence = self.divergence_jsd
        elif flags.modality_poe:
            self.modality_fusion = self.poe_fusion
            self.calc_joint_divergence = self.divergence_poe

    def forward(self, input_batch):
        latents = self.inference(input_batch)
        
        results = dict()
        results['latents'] = latents

        results['group_distr'] = latents['joint']
        class_embeddings = utils.reparameterize(latents['joint'][0],
                                                latents['joint'][1])
        div = self.calc_joint_divergence(latents['mus'],
                                         latents['logvars'],
                                         latents['weights'])
        for k, key in enumerate(div.keys()):
            results[key] = div[key]

        enc_mods = latents['modalities']
        results_rec = dict()
        for m in range(self.num_modalities):
            x_m = input_batch['m%d' % m]
            if x_m is not None:
                style_mu, style_logvar = enc_mods['m%d_style' % m]
                if self.flags.factorized_representation:
                    style_embeddings = utils.reparameterize(mu=style_mu, logvar=style_logvar)
                else:
                    style_embeddings = None
                rec = self.likelihoods[m](*self.decoders[m](style_embeddings, class_embeddings))
                results_rec['m%d' % m] = rec
        results['rec'] = results_rec
        return results

    def divergence_poe(self, mus, logvars, weights=None):
        div_measures = calc_group_divergence_poe(self.flags,
                                         mus,
                                         logvars,
                                         norm=self.flags.batch_size)
        divs = dict()
        divs['joint_divergence'] = div_measures[0]
        divs['individual_divs'] = div_measures[1]
        divs['dyn_prior'] = None
        return divs

    def divergence_moe(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights
        weights = weights.clone()
        weights[0] = 0.0
        weights = utils.reweight_weights(weights)
        div_measures = calc_group_divergence_moe(self.flags,
                                                 mus,
                                                 logvars,
                                                 weights,
                                                 normalization=self.flags.batch_size)
        divs = dict()
        divs['joint_divergence'] = div_measures[0]
        divs['individual_divs'] = div_measures[1]
        divs['dyn_prior'] = None
        return divs

    def divergence_moe_poe(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights
        weights = weights.clone()
        weights[0] = 0.0
        weights = utils.reweight_weights(weights)
        div_measures = calc_group_divergence_moe(self.flags,
                                                 mus,
                                                 logvars,
                                                 weights,
                                                 normalization=self.flags.batch_size)
        divs = dict()
        divs['joint_divergence'] = div_measures[0]
        divs['individual_divs'] = div_measures[1]
        divs['dyn_prior'] = None
        return divs

    def divergence_jsd(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights_jsd
        div_measures = calc_alphaJSD_modalities(self.flags,
                                                mus,
                                                logvars,
                                                weights,
                                                normalization=self.flags.batch_size)
        divs = dict()
        divs['joint_divergence'] = div_measures[0]
        divs['individual_divs'] = div_measures[1]
        divs['dyn_prior'] = div_measures[2]
        return divs

    def moe_fusion(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights
        weights[0] = 0.0
        weights = utils.reweight_weights(weights)
        # num_samples = mus[0].shape[0]
        mu_moe, logvar_moe = utils.mixture_component_selection(self.flags,
                                                               mus,
                                                               logvars,
                                                               weights)
        return [mu_moe, logvar_moe]

    def poe_fusion(self, mus, logvars, weights=None):
        mu_poe, logvar_poe = poe(mus, logvars)
        return [mu_poe, logvar_poe]

    def encode(self, x_m, m):
        if x_m is not None:
            latents = self.encoders[m](x_m)
            latents_style = latents[:2]
            latents_class = latents[2:]
        else:
            latents_style = [None, None]
            latents_class = [None, None]
        return latents_style, latents_class

    def inference(self, input_batch):
        latents = dict()
        enc_mods = dict()
        for m in range(self.num_modalities):
            if "m%d" % m in input_batch.keys():
                x_m = input_batch['m%d' % m]
                num_samples = x_m.shape[0]
            else:
                x_m = None
            latents_style, latents_class = self.encode(x_m, m)
            enc_mods["m%d" % m] = latents_class
            enc_mods["m%d_style" % m] = latents_style
        latents['modalities'] = enc_mods
        mus = [torch.zeros(1, num_samples,
                           self.flags.class_dim).to(self.flags.device)]
        logvars = [torch.zeros(1, num_samples,
                               self.flags.class_dim).to(self.flags.device)]
        distr_subsets = dict()
        for k, s_key in enumerate(self.subsets.keys()):
            if s_key != '':
                mods = self.subsets[s_key]
                mus_subset = []
                logvars_subset = []
                mods_avail = True
                for m, mod in enumerate(mods):
                    if mod.name in input_batch.keys():
                        mus_subset.append(enc_mods[mod.name][0].unsqueeze(0))
                        logvars_subset.append(enc_mods[mod.name][1].unsqueeze(0))
                    else:
                        mods_avail = False
                if mods_avail:
                    mus_subset = torch.cat(mus_subset, dim=0)
                    logvars_subset = torch.cat(logvars_subset, dim=0)
                    poe_subset = poe(mus_subset, logvars_subset)
                    distr_subsets[s_key] = poe_subset
                    mus.append(poe_subset[0].unsqueeze(0))
                    logvars.append(poe_subset[1].unsqueeze(0))
        mus = torch.cat(mus, dim=0)
        logvars = torch.cat(logvars, dim=0)
        weights = (1/float(mus.shape[0]))*torch.ones(mus.shape[0]).to(self.flags.device)
        joint_mu, joint_logvar = self.modality_fusion(mus, logvars, weights)
        latents['mus'] = mus
        latents['logvars'] = logvars
        latents['weights'] = weights
        latents['joint'] = [joint_mu, joint_logvar]
        latents['subsets'] = distr_subsets
        return latents

    def get_random_styles(self, num_samples):
        styles = dict()
        for m in range(self.num_modalities):
            if self.flags.factorized_representation:
                z_style_m = torch.randn(num_samples, self.flags.style_dim)
                z_style_m = z_style_m.to(self.flags.device)
            else:
                z_style_m = None
            styles["m%d" % m] = z_style_m
        return styles

    def get_random_style_dists(self, num_samples):
        styles = dict()
        for m in range(self.num_modalities):
            s_mu_m = torch.zeros(num_samples, self.flags.style_dim).to(self.flags.device)
            s_logvar_m = torch.zeros(num_samples, self.flags.style_dim).to(self.flags.device)
            dist_m = [s_mu_m, s_logvar_m]
            styles["m%d" % m] = dist_m
        return styles

    def generate(self, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size
        z_class = torch.randn(num_samples, self.flags.class_dim)
        z_class = z_class.to(self.flags.device)

        style_latents = self.get_random_styles(num_samples)
        random_latents = {'content': z_class, 'style': style_latents}
        random_samples = self.generate_from_latents(random_latents)
        return random_samples

    def generate_from_latents(self, latents):
        cond_gen = {}
        for m in range(self.num_modalities):
            suff_stats = self.generate_sufficient_statistics_from_latents(latents)
            cond_gen_m = suff_stats["m%d" % m].mean
            cond_gen["m%d" % m] = cond_gen_m
        return cond_gen

    def generate_sufficient_statistics_from_latents(self, latents):
        cond_gen = {}
        for m in range(self.num_modalities):
            style_m = latents['style']['m%d' % m]
            content = latents['content']
            cond_gen_m = self.likelihoods[m](*self.decoders[m](style_m, content))
            cond_gen["m%d" % m] = cond_gen_m
        return cond_gen

    def cond_generation(self, latent_distributions, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size

        style_latents = self.get_random_styles(num_samples)
        cond_gen_samples = dict()
        for k, key in enumerate(latent_distributions.keys()):
            [mu, logvar] = latent_distributions[key]
            content_rep = utils.reparameterize(mu=mu, logvar=logvar)
            latents = {'content': content_rep, 'style': style_latents}
            cond_gen_samples[key] = self.generate_from_latents(latents)
        return cond_gen_samples

    def cond_generation_2a(self, latent_distribution_pairs, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size

        style_latents = self.get_random_styles(num_samples)
        cond_gen_2a = dict()
        for p, pair in enumerate(latent_distribution_pairs.keys()):
            ld_pair = latent_distribution_pairs[pair]
            mu_list = []; logvar_list = []
            for k, key in enumerate(ld_pair['latents'].keys()):
                mu_list.append(ld_pair['latents'][key][0].unsqueeze(0))
                logvar_list.append(ld_pair['latents'][key][1].unsqueeze(0))
            mus = torch.cat(mu_list, dim=0)
            logvars = torch.cat(logvar_list, dim=0)
            # weights_pair = utils.reweight_weights(torch.Tensor(ld_pair['weights']))
            # mu_joint, logvar_joint = self.modality_fusion(mus, logvars, weights_pair)
            mu_joint, logvar_joint = poe(mus, logvars)
            c_emb = utils.reparameterize(mu_joint, logvar_joint)
            l_2a = {'content': c_emb, 'style': style_latents}
            cond_gen_2a[pair] = self.generate_from_latents(l_2a)
        return cond_gen_2a

    def save_networks(self):
        for m in range(self.num_modalities):
            torch.save(self.encoders[m].state_dict(), os.path.join(self.flags.dir_checkpoints, "encoderM%d" % m))
