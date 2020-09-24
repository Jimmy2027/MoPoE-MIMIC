from abc import ABC, abstractmethod

import os

import torch
import torch.nn as nn

from divergence_measures.mm_div import calc_alphaJSD_modalities
from divergence_measures.mm_div import calc_group_divergence_moe
from divergence_measures.mm_div import poe

from utils import utils


class BaseMMVae(ABC, nn.Module):
    def __init__(self, flags, modalities, subsets):
        super(BaseMMVae, self).__init__()
        self.num_modalities = len(modalities.keys());
        self.flags = flags;
        self.modalities = modalities;
        self.subsets = subsets;
        self.set_fusion_functions();

        # assign encoders, decoders and likelihoods here # #
        #
        # ###############################################

    @abstractmethod
    def forward(self, input_batch):
        pass;

    @abstractmethod
    def encode(self, input_batch):
        pass;

    @abstractmethod
    def get_random_styles(self, num_samples):
        pass;


    @abstractmethod
    def get_random_style_dists(self, num_samples):
        pass;

    @abstractmethod
    def generate_sufficient_statistics_from_latents(self, latents):
        pass;

    @abstractmethod
    def save_networks(self):
        pass;


    def set_fusion_functions(self):
        weights = utils.reweight_weights(torch.Tensor(self.flags.alpha_modalities));
        self.weights = weights.to(self.flags.device);
        if self.flags.modality_moe:
            self.modality_fusion = self.moe_fusion;
            self.fusion_condition = self.fusion_condition_moe;
            self.calc_joint_divergence = self.divergence_static_prior;
        elif self.flags.modality_jsd:
            self.modality_fusion = self.moe_fusion;
            self.fusion_condition = self.fusion_condition_moe;
            self.calc_joint_divergence = self.divergence_dynamic_prior;
        elif self.flags.modality_poe:
            self.modality_fusion = self.poe_fusion;
            self.fusion_condition = self.fusion_condition_poe;
            self.calc_joint_divergence = self.divergence_static_prior;
        elif self.flags.joint_elbo:
            self.modality_fusion = self.poe_fusion;
            self.fusion_condition = self.fusion_condition_joint;
            self.calc_joint_divergence = self.divergence_static_prior;


    def divergence_static_prior(self, mus, logvars, weights=None):
        if weights is None:
            weights=self.weights;
        weights = weights.clone();
        weights = utils.reweight_weights(weights);
        div_measures = calc_group_divergence_moe(self.flags,
                                                 mus,
                                                 logvars,
                                                 weights,
                                                 normalization=self.flags.batch_size);
        divs = dict();
        divs['joint_divergence'] = div_measures[0];
        divs['individual_divs'] = div_measures[1];
        divs['dyn_prior'] = None;
        return divs;


    def divergence_dynamic_prior(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights;
        div_measures = calc_alphaJSD_modalities(self.flags,
                                                mus,
                                                logvars,
                                                weights,
                                                normalization=self.flags.batch_size);
        divs = dict();
        divs['joint_divergence'] = div_measures[0];
        divs['individual_divs'] = div_measures[1];
        divs['dyn_prior'] = div_measures[2];
        return divs;


    def moe_fusion(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights;
        weights = utils.reweight_weights(weights);
        #mus = torch.cat(mus, dim=0);
        #logvars = torch.cat(logvars, dim=0);
        mu_moe, logvar_moe = utils.mixture_component_selection(self.flags,
                                                               mus,
                                                               logvars,
                                                               weights);
        return [mu_moe, logvar_moe];


    def poe_fusion(self, mus, logvars, weights=None):
        if self.flags.modality_poe:
            num_samples = mus[0].shape[0];
            mus = torch.cat((mus, torch.zeros(1, num_samples,
                             self.flags.class_dim).to(self.flags.device)),
                            dim=0);
            logvars = torch.cat((logvars, torch.zeros(1, num_samples,
                                 self.flags.class_dim).to(self.flags.device)),
                                dim=0);
        #mus = torch.cat(mus, dim=0);
        #logvars = torch.cat(logvars, dim=0);
        mu_poe, logvar_poe = poe(mus, logvars);
        return [mu_poe, logvar_poe];


    def fusion_condition_moe(self, subset, input_batch=None):
        if len(subset) == 1:
            return True;
        else:
            return False;


    def fusion_condition_poe(self, subset, input_batch=None):
        if len(subset) == len(input_batch.keys()):
            return True;
        else:
            return False;


    def fusion_condition_joint(self, subset, input_batch=None):
        return True;


    def inference(self, input_batch, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size;
        latents = dict();
        enc_mods = self.encode(input_batch);
        latents['modalities'] = enc_mods;
        mus = torch.Tensor().to(self.flags.device);
        logvars = torch.Tensor().to(self.flags.device);
        distr_subsets = dict();
        for k, s_key in enumerate(self.subsets.keys()):
            if s_key != '':
                mods = self.subsets[s_key];
                mus_subset = torch.Tensor().to(self.flags.device);
                logvars_subset = torch.Tensor().to(self.flags.device);
                mods_avail = True
                for m, mod in enumerate(mods):
                    if mod.name in input_batch.keys():
                        mus_subset = torch.cat((mus_subset,
                                                enc_mods[mod.name][0].unsqueeze(0)),
                                               dim=0);
                        logvars_subset = torch.cat((logvars_subset,
                                                    enc_mods[mod.name][1].unsqueeze(0)),
                                                   dim=0);
                    else:
                        mods_avail = False;
                if mods_avail:
                    weights_subset = ((1/float(len(mus_subset)))*
                                      torch.ones(len(mus_subset)).to(self.flags.device));
                    s_mu, s_logvar = self.modality_fusion(mus_subset,
                                                          logvars_subset,
                                                          weights_subset);
                    distr_subsets[s_key] = [s_mu, s_logvar];
                    if self.fusion_condition(mods, input_batch):
                        mus = torch.cat((mus, s_mu.unsqueeze(0)), dim=0);
                        logvars = torch.cat((logvars, s_logvar.unsqueeze(0)),
                                            dim=0);
        if self.flags.modality_jsd:
            mus = torch.cat((mus, torch.zeros(1, num_samples,
                                      self.flags.class_dim).to(self.flags.device)),
                            dim=0);
            logvars = torch.cat((logvars, torch.zeros(1, num_samples,
                                          self.flags.class_dim).to(self.flags.device)),
                                dim=0);
        #weights = (1/float(len(mus)))*torch.ones(len(mus)).to(self.flags.device);
        weights = (1/float(mus.shape[0]))*torch.ones(mus.shape[0]).to(self.flags.device);
        joint_mu, joint_logvar = self.moe_fusion(mus, logvars, weights);
        #mus = torch.cat(mus, dim=0);
        #logvars = torch.cat(logvars, dim=0);
        latents['mus'] = mus;
        latents['logvars'] = logvars;
        latents['weights'] = weights;
        latents['joint'] = [joint_mu, joint_logvar];
        latents['subsets'] = distr_subsets;
        return latents;


    def generate(self, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size;

        mu = torch.zeros(num_samples,
                         self.flags.class_dim).to(self.flags.device);
        logvar = torch.zeros(num_samples,
                             self.flags.class_dim).to(self.flags.device);
        z_class = utils.reparameterize(mu, logvar);
        z_styles = self.get_random_styles(num_samples);
        random_latents = {'content': z_class, 'style': z_styles};
        random_samples = self.generate_from_latents(random_latents);
        return random_samples;


    def generate_from_latents(self, latents):
        suff_stats = self.generate_sufficient_statistics_from_latents(latents);
        cond_gen = dict();
        for m, m_key in enumerate(latents['style'].keys()):
            cond_gen_m = suff_stats[m_key].mean;
            cond_gen[m_key] = cond_gen_m;
        return cond_gen;


    def cond_generation(self, latent_distributions, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size;

        style_latents = self.get_random_styles(num_samples);
        cond_gen_samples = dict();
        for k, key in enumerate(latent_distributions.keys()):
            [mu, logvar] = latent_distributions[key];
            content_rep = utils.reparameterize(mu=mu, logvar=logvar);
            latents = {'content': content_rep, 'style': style_latents}
            cond_gen_samples[key] = self.generate_from_latents(latents);
        return cond_gen_samples;


