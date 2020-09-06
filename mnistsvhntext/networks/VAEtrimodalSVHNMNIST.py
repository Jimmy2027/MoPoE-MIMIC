import os

import torch
import torch.nn as nn

from mnistsvhntext.networks.ConvNetworksImgSVHN import EncoderSVHN, DecoderSVHN
from mnistsvhntext.networks.ConvNetworksImgMNIST import EncoderImg, DecoderImg
from mnistsvhntext.networks.ConvNetworksTextMNIST import EncoderText, DecoderText

from utils import utils
from utils.BaseMMVae import BaseMMVae


class VAEtrimodalSVHNMNIST(BaseMMVae, nn.Module):
    def __init__(self, flags, modalities, subsets):
        super().__init__(flags, modalities, subsets)
        self.encoder_m1 = modalities['mnist'].encoder;
        self.decoder_m1 = modalities['mnist'].decoder;
        self.encoder_m2 = modalities['svhn'].encoder;
        self.decoder_m2 = modalities['svhn'].decoder;
        self.encoder_m3 = modalities['text'].encoder;
        self.decoder_m3 = modalities['text'].decoder;
        self.lhood_m1 = modalities['mnist'].likelihood;
        self.lhood_m2 = modalities['svhn'].likelihood;
        self.lhood_m3 = modalities['text'].likelihood;
        self.encoder_m1 = self.encoder_m1.to(flags.device);
        self.decoder_m1 = self.decoder_m1.to(flags.device);
        self.encoder_m2 = self.encoder_m2.to(flags.device);
        self.decoder_m2 = self.decoder_m2.to(flags.device);
        self.encoder_m3 = self.encoder_m3.to(flags.device);
        self.decoder_m3 = self.decoder_m3.to(flags.device);




    def forward(self, input_batch):
        latents = self.inference(input_batch);
        results = dict();
        results['latents'] = latents;
        results['group_distr'] = latents['joint'];
        class_embeddings = utils.reparameterize(latents['joint'][0],
                                                latents['joint'][1]);
        div = self.calc_joint_divergence(latents['mus'],
                                         latents['logvars'],
                                         latents['weights']);
        for k, key in enumerate(div.keys()):
            results[key] = div[key];

        results_rec = dict();
        enc_mods = latents['modalities'];
        if 'mnist' in input_batch.keys():
            m1_s_mu, m1_s_logvar = enc_mods['mnist_style'];
            if self.flags.factorized_representation:
                m1_s_embeddings = utils.reparameterize(mu=m1_s_mu, logvar=m1_s_logvar);
            else:
                m1_s_embeddings = None;
            m1_rec = self.lhood_m1(*self.decoder_m1(m1_s_embeddings, class_embeddings));
            results_rec['mnist'] = m1_rec;
        if 'svhn' in input_batch.keys():
            m2_s_mu, m2_s_logvar = enc_mods['svhn_style'];
            if self.flags.factorized_representation:
                m2_s_embeddings = utils.reparameterize(mu=m2_s_mu, logvar=m2_s_logvar);
            else:
                m2_s_embeddings = None;
            m2_rec = self.lhood_m2(*self.decoder_m2(m2_s_embeddings, class_embeddings));
            results_rec['svhn'] = m2_rec;
        if 'text' in input_batch.keys():
            m3_s_mu, m3_s_logvar = enc_mods['text_style'];
            if self.flags.factorized_representation:
                m3_s_embeddings = utils.reparameterize(mu=m3_s_mu, logvar=m3_s_logvar);
            else:
                m3_s_embeddings = None;
            m3_rec = self.lhood_m3(*self.decoder_m3(m3_s_embeddings, class_embeddings));
            results_rec['text'] = m3_rec;
        results['rec'] = results_rec;
        return results;


    #def divergence_static_prior(self, mus, logvars, weights=None):
    #    if weights is None:
    #        weights=self.weights;
    #    weights = weights.clone();
    #    weights[0] = 0.0;
    #    weights = utils.reweight_weights(weights);
    #    div_measures = calc_group_divergence_moe(self.flags,
    #                                             mus,
    #                                             logvars,
    #                                             weights,
    #                                             normalization=self.flags.batch_size);
    #    divs = dict();
    #    divs['joint_divergence'] = div_measures[0];
    #    divs['individual_divs'] = div_measures[1];
    #    divs['dyn_prior'] = None;
    #    return divs;


    #def divergence_dynamic_prior(self, mus, logvars, weights=None):
    #    if weights is None:
    #        weights = self.weights_jsd;
    #    div_measures = calc_alphaJSD_modalities(self.flags,
    #                                            mus,
    #                                            logvars,
    #                                            weights,
    #                                            normalization=self.flags.batch_size);
    #    divs = dict();
    #    divs['joint_divergence'] = div_measures[0];
    #    divs['individual_divs'] = div_measures[1];
    #    divs['dyn_prior'] = div_measures[2];
    #    return divs;



    #def moe_fusion(self, mus, logvars, weights=None):
    #    if weights is None:
    #        weights = self.weights;
    #    weights[0] = 0.0;
    #    weights = utils.reweight_weights(weights);
    #    num_samples = mus[0].shape[0];
    #    mu_moe, logvar_moe = utils.mixture_component_selection(self.flags,
    #                                                           mus,
    #                                                           logvars,
    #                                                           weights);
    #    return [mu_moe, logvar_moe];


    #def poe_fusion(self, mus, logvars, weights=None):
    #    mu_poe, logvar_poe = poe(mus, logvars);
    #    return [mu_poe, logvar_poe];


    #def fusion_condition_moe(self, subset):
    #    if len(subset) == 1:
    #        return True;
    #    else:
    #        return False;


    #def fusion_condition_poe(self, subset):
    #    if len(subset) == len(self.modalities):
    #        return True;
    #    else:
    #        return False;


    #def fusion_condition_joint(self, subset):
    #    return True;


    def encode(self, input_batch):
        latents = dict();
        if 'mnist' in input_batch.keys():
            i_m1 = input_batch['mnist'];
            latents['mnist'] = self.encoder_m1(i_m1)
            latents['mnist_style'] = latents['mnist'][:2]
            latents['mnist'] = latents['mnist'][2:]
        else:
            latents['mnist_style'] = [None, None];
            latents['mnist'] = [None, None];
        if 'svhn' in input_batch.keys():
            i_m2 = input_batch['svhn'];
            latents['svhn'] = self.encoder_m2(i_m2);
            latents['svhn_style'] = latents['svhn'][:2];
            latents['svhn'] = latents['svhn'][2:];
        else:
            latents['svhn_style'] = [None, None];
            latents['svhn'] = [None, None];
        if 'text' in input_batch.keys():
            i_m3 = input_batch['text'];
            latents['text'] = self.encoder_m3(i_m3);
            latents['text_style'] = latents['text'][:2];
            latents['text'] = latents['text'][2:];
        else:
            latents['text_style'] = [None, None];
            latents['text'] = [None, None];
        return latents;


    #def inference(self, input_batch, num_samples=None):
    #    if num_samples is None:
    #        num_samples = self.flags.batch_size;
    #    latents = dict();
    #    enc_mods = self.encode(input_batch);
    #    latents['modalities'] = enc_mods;
    #    mus = [torch.zeros(1, num_samples,
    #                       self.flags.class_dim).to(self.flags.device)];
    #    logvars = [torch.zeros(1, num_samples,
    #                           self.flags.class_dim).to(self.flags.device)];
    #    distr_subsets = dict();
    #    for k, s_key in enumerate(self.subsets.keys()):
    #        if s_key != '':
    #            mods = self.subsets[s_key];
    #            mus_subset = [];
    #            logvars_subset = [];
    #            mods_avail = True
    #            for m, mod in enumerate(mods):
    #                if mod.name in input_batch.keys():
    #                    mus_subset.append(enc_mods[mod.name][0].unsqueeze(0));
    #                    logvars_subset.append(enc_mods[mod.name][1].unsqueeze(0));
    #                else:
    #                    mods_avail = False;
    #            if mods_avail:
    #                mus_subset = torch.cat(mus_subset, dim=0);
    #                logvars_subset = torch.cat(logvars_subset, dim=0);
    #                weights_subset = ((1/float(mus_subset.shape[0]))*
    #                                  torch.ones(mus_subset.shape[0]).to(self.flags.device));
    #                s_mu, s_logvar = self.modality_fusion(mus_subset,
    #                                                      logvars_subset,
    #                                                      weights_subset);
    #                distr_subsets[s_key] = [s_mu, s_logvar];
    #                if self.fusion_condition:
    #                    mus.append(s_mu.unsqueeze(0));
    #                    logvars.append(s_logvar.unsqueeze(0));
    #    mus = torch.cat(mus, dim=0);
    #    logvars = torch.cat(logvars, dim=0);
    #    weights = (1/float(mus.shape[0]))*torch.ones(mus.shape[0]).to(self.flags.device);
    #    joint_mu, joint_logvar = self.moe_fusion(mus, logvars, weights);
    #    latents['mus'] = mus;
    #    latents['logvars'] = logvars;
    #    latents['weights'] = weights;
    #    latents['joint'] = [joint_mu, joint_logvar];
    #    latents['subsets'] = distr_subsets;
    #    return latents;


    def get_random_styles(self, num_samples):
        if self.flags.factorized_representation:
            z_style_1 = torch.randn(num_samples, self.flags.style_m1_dim);
            z_style_2 = torch.randn(num_samples, self.flags.style_m2_dim);
            z_style_3 = torch.randn(num_samples, self.flags.style_m3_dim);
            z_style_1 = z_style_1.to(self.flags.device);
            z_style_2 = z_style_2.to(self.flags.device);
            z_style_3 = z_style_3.to(self.flags.device);
        else:
            z_style_1 = None;
            z_style_2 = None;
            z_style_3 = None;
        styles = {'mnist': z_style_1, 'svhn': z_style_2, 'text': z_style_3};
        return styles;


    def get_random_style_dists(self, num_samples):
        s1_mu = torch.zeros(num_samples,
                            self.flags.style_m1_dim).to(self.flags.device)
        s1_logvar = torch.zeros(num_samples, self.flags.style_m1_dim).to(self.flags.device); s2_mu = torch.zeros(num_samples, self.flags.style_m2_dim).to(self.flags.device)
        s2_logvar = torch.zeros(num_samples,
                                self.flags.style_m2_dim).to(self.flags.device);
        s3_mu = torch.zeros(num_samples,
                            self.flags.style_m3_dim).to(self.flags.device)
        s3_logvar = torch.zeros(num_samples,
                                self.flags.style_m3_dim).to(self.flags.device);
        m1_dist = [s1_mu, s1_logvar];
        m2_dist = [s2_mu, s2_logvar];
        m3_dist = [s3_mu, s3_logvar];
        styles = {'mnist': m1_dist, 'svhn': m2_dist, 'text': m3_dist};
        return styles;


    #def generate(self, num_samples=None):
    #    if num_samples is None:
    #        num_samples = self.flags.batch_size;
    #    z_class = torch.randn(num_samples, self.flags.class_dim);
    #    z_class = z_class.to(self.flags.device);

    #    style_latents = self.get_random_styles(num_samples);
    #    random_latents = {'content': z_class, 'style': style_latents};
    #    random_samples = self.generate_from_latents(random_latents);
    #    return random_samples;


    #def generate_from_latents(self, latents):
    #    suff_stats = self.generate_sufficient_statistics_from_latents(latents);
    #    cond_gen_m1 = suff_stats['mnist'].mean;
    #    cond_gen_m2 = suff_stats['svhn'].mean;
    #    cond_gen_m3 = suff_stats['text'].mean;
    #    cond_gen = {'mnist': cond_gen_m1,
    #                'svhn': cond_gen_m2,
    #                'text': cond_gen_m3};
    #    return cond_gen;


    def generate_sufficient_statistics_from_latents(self, latents):
        style_m1 = latents['style']['mnist'];
        style_m2 = latents['style']['svhn'];
        style_m3 = latents['style']['text'];
        content = latents['content']
        cond_gen_m1 = self.lhood_m1(*self.decoder_m1(style_m1, content));
        cond_gen_m2 = self.lhood_m2(*self.decoder_m2(style_m2, content));
        cond_gen_m3 = self.lhood_m3(*self.decoder_m3(style_m3, content));
        return {'mnist': cond_gen_m1, 'svhn': cond_gen_m2, 'text': cond_gen_m3}


    #def cond_generation(self, latent_distributions, num_samples=None):
    #    if num_samples is None:
    #        num_samples = self.flags.batch_size;

    #    style_latents = self.get_random_styles(num_samples);
    #    cond_gen_samples = dict();
    #    for k, key in enumerate(latent_distributions.keys()):
    #        [mu, logvar] = latent_distributions[key];
    #        content_rep = utils.reparameterize(mu=mu, logvar=logvar);
    #        latents = {'content': content_rep, 'style': style_latents}
    #        cond_gen_samples[key] = self.generate_from_latents(latents);
    #    return cond_gen_samples;


    def save_networks(self):
        torch.save(self.encoder_m1.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m1))
        torch.save(self.decoder_m1.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m1))
        torch.save(self.encoder_m2.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m2))
        torch.save(self.decoder_m2.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m2))
        torch.save(self.encoder_m3.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m3))
        torch.save(self.decoder_m3.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m3))
