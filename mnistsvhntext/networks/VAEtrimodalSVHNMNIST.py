import os

import torch
import torch.nn as nn

from mnistsvhntext.networks.ConvNetworksImgSVHN import EncoderSVHN, DecoderSVHN
from mnistsvhntext.networks.ConvNetworksImgMNIST import EncoderImg, DecoderImg
from mnistsvhntext.networks.ConvNetworksTextMNIST import EncoderText, DecoderText

from divergence_measures.mm_div import calc_alphaJSD_modalities
from divergence_measures.mm_div import calc_group_divergence_poe
from divergence_measures.mm_div import calc_group_divergence_moe
from divergence_measures.mm_div import calc_kl_divergence
from divergence_measures.mm_div import poe

from utils import utils


class VAEtrimodalSVHNMNIST(nn.Module):
    def __init__(self, flags, modalities, subsets):
        super(VAEtrimodalSVHNMNIST, self).__init__()
        self.num_modalities = len(modalities.keys());
        self.flags = flags;
        self.modalities = modalities;
        self.subsets = subsets;
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

        weights = utils.reweight_weights(torch.Tensor(flags.alpha_modalities));
        self.weights = weights.to(flags.device);
        if flags.modality_moe or flags.modality_jsd:
            self.modality_fusion = self.moe_fusion;
            if flags.modality_moe:
                self.calc_joint_divergence = self.divergence_moe;
            if flags.modality_jsd:
                self.calc_joint_divergence = self.divergence_jsd;
        elif flags.modality_poe:
            self.modality_fusion = self.poe_fusion;
            self.calc_joint_divergence = self.divergence_poe;


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

        input_m1 = input_batch['mnist'];
        input_m2 = input_batch['svhn'];
        input_m3 = input_batch['text'];
        results_rec = dict();
        enc_mods = latents['modalities'];
        if input_m1 is not None:
            m1_s_mu, m1_s_logvar = enc_mods['mnist_style'];
            if self.flags.factorized_representation:
                m1_s_embeddings = utils.reparameterize(mu=m1_s_mu, logvar=m1_s_logvar);
            else:
                m1_s_embeddings = None;
            m1_rec = self.lhood_m1(*self.decoder_m1(m1_s_embeddings, class_embeddings));
            results_rec['mnist'] = m1_rec;
        if input_m2 is not None:
            m2_s_mu, m2_s_logvar = enc_mods['svhn_style'];
            if self.flags.factorized_representation:
                m2_s_embeddings = utils.reparameterize(mu=m2_s_mu, logvar=m2_s_logvar);
            else:
                m2_s_embeddings = None;
            m2_rec = self.lhood_m2(*self.decoder_m2(m2_s_embeddings, class_embeddings));
            results_rec['svhn'] = m2_rec;
        if input_m3 is not None:
            m3_s_mu, m3_s_logvar = enc_mods['text_style'];
            if self.flags.factorized_representation:
                m3_s_embeddings = utils.reparameterize(mu=m3_s_mu, logvar=m3_s_logvar);
            else:
                m3_s_embeddings = None;
            m3_rec = self.lhood_m3(*self.decoder_m3(m3_s_embeddings, class_embeddings));
            results_rec['text'] = m3_rec;
        results['rec'] = results_rec;
        return results;


    def divergence_poe(self, mus, logvars, weights=None):
        div_measures = calc_group_divergence_poe(self.flags,
                                         mus,
                                         logvars,
                                         norm=self.flags.batch_size);
        divs = dict();
        divs['joint_divergence'] = div_measures[0];
        divs['individual_divs'] = div_measures[1];
        divs['dyn_prior'] = None;
        return divs;


    def divergence_moe(self, mus, logvars, weights=None):
        if weights is None:
            weights=self.weights;
        weights = weights.clone();
        weights[0] = 0.0;
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


    def divergence_moe_poe(self, mus, logvars, weights=None):
        if weights is None:
            weights=self.weights;
        weights = weights.clone();
        weights[0] = 0.0;
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

    def divergence_jsd(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights_jsd;
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
        weights[0] = 0.0;
        weights = utils.reweight_weights(weights);
        num_samples = mus[0].shape[0];
        mu_moe, logvar_moe = utils.mixture_component_selection(self.flags,
                                                               mus,
                                                               logvars,
                                                               weights);
        return [mu_moe, logvar_moe];


    def poe_fusion(self, mus, logvars, weights=None):
        mu_poe, logvar_poe = poe(mus, logvars);
        return [mu_poe, logvar_poe];


    def encode(self, i_m1=None, i_m2=None, i_m3=None):
        latents = dict();
        if i_m1 is not None:
            latents['mnist'] = self.encoder_m1(i_m1)
            latents['mnist_style'] = latents['mnist'][:2]
            latents['mnist'] = latents['mnist'][2:]
        else:
            latents['mnist_style'] = [None, None];
            latents['mnist'] = [None, None];
        if i_m2 is not None:
            latents['svhn'] = self.encoder_m2(i_m2);
            latents['svhn_style'] = latents['svhn'][:2];
            latents['svhn'] = latents['svhn'][2:];
        else:
            latents['svhn_style'] = [None, None];
            latents['svhn'] = [None, None];
        if i_m3 is not None:
            latents['text'] = self.encoder_m3(i_m3);
            latents['text_style'] = latents['text'][:2];
            latents['text'] = latents['text'][2:];
        else:
            latents['text_style'] = [None, None];
            latents['text'] = [None, None];
        return latents;


    def inference(self, input_batch):
        if 'mnist' in input_batch.keys():
            input_m1 = input_batch['mnist'];
        else:
            input_m1 = None;
        if 'svhn' in input_batch.keys():
            input_m2 = input_batch['svhn'];
        else:
            input_m2 = None;
        if 'text' in input_batch.keys():
            input_m3 = input_batch['text'];
        else:
            input_m3 = None;
        latents = dict();
        enc_mods = self.encode(i_m1=input_m1,
                              i_m2=input_m2,
                              i_m3=input_m3);
        latents['modalities'] = enc_mods;
        if input_m1 is not None:
            num_samples = input_m1.shape[0];
        if input_m2 is not None:
            num_samples = input_m2.shape[0];
        if input_m3 is not None:
            num_samples = input_m3.shape[0];
        mus = [torch.zeros(1, num_samples,
                           self.flags.class_dim).to(self.flags.device)];
        logvars = [torch.zeros(1, num_samples,
                               self.flags.class_dim).to(self.flags.device)];
        distr_subsets = dict();
        for k, s_key in enumerate(self.subsets.keys()):
            if s_key != '':
                mods = self.subsets[s_key];
                mus_subset = [];
                logvars_subset = [];
                mods_avail = True
                for m, mod in enumerate(mods):
                    if mod.name in input_batch.keys():
                        mus_subset.append(enc_mods[mod.name][0].unsqueeze(0));
                        logvars_subset.append(enc_mods[mod.name][1].unsqueeze(0));
                    else:
                        mods_avail = False;
                if mods_avail:
                    mus_subset = torch.cat(mus_subset, dim=0);
                    logvars_subset = torch.cat(logvars_subset, dim=0);
                    poe_subset = poe(mus_subset, logvars_subset);
                    distr_subsets[s_key] = poe_subset;
                    mus.append(poe_subset[0].unsqueeze(0));
                    logvars.append(poe_subset[1].unsqueeze(0));
        mus = torch.cat(mus, dim=0);
        logvars = torch.cat(logvars, dim=0);
        weights = (1/float(mus.shape[0]))*torch.ones(mus.shape[0]).to(self.flags.device);
        joint_mu, joint_logvar = self.modality_fusion(mus, logvars, weights);
        latents['mus'] = mus;
        latents['logvars'] = logvars;
        latents['weights'] = weights;
        latents['joint'] = [joint_mu, joint_logvar];
        latents['subsets'] = distr_subsets;
        return latents;


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
        s1_logvar = torch.zeros(num_samples,
                                self.flags.style_m1_dim).to(self.flags.device);
        s2_mu = torch.zeros(num_samples,
                            self.flags.style_m2_dim).to(self.flags.device)
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


    def generate(self, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size;
        z_class = torch.randn(num_samples, self.flags.class_dim);
        z_class = z_class.to(self.flags.device);

        style_latents = self.get_random_styles(num_samples);
        random_latents = {'content': z_class, 'style': style_latents};
        random_samples = self.generate_from_latents(random_latents);
        return random_samples;


    def generate_from_latents(self, latents):
        suff_stats = self.generate_sufficient_statistics_from_latents(latents);
        cond_gen_m1 = suff_stats['mnist'].mean;
        cond_gen_m2 = suff_stats['svhn'].mean;
        cond_gen_m3 = suff_stats['text'].mean;
        cond_gen = {'mnist': cond_gen_m1,
                    'svhn': cond_gen_m2,
                    'text': cond_gen_m3};
        return cond_gen;


    def generate_sufficient_statistics_from_latents(self, latents):
        style_m1 = latents['style']['mnist'];
        style_m2 = latents['style']['svhn'];
        style_m3 = latents['style']['text'];
        content = latents['content']
        cond_gen_m1 = self.lhood_m1(*self.decoder_m1(style_m1, content));
        cond_gen_m2 = self.lhood_m2(*self.decoder_m2(style_m2, content));
        cond_gen_m3 = self.lhood_m3(*self.decoder_m3(style_m3, content));
        return {'mnist': cond_gen_m1, 'svhn': cond_gen_m2, 'text': cond_gen_m3}


    def cond_generation_1a(self, latent_distributions, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size;

        style_latents = self.get_random_styles(num_samples);
        cond_gen_samples = dict();
        for k, key in enumerate(latent_distributions):
            [mu, logvar] = latent_distributions[key];
            content_rep = utils.reparameterize(mu=mu, logvar=logvar);
            latents = {'content': content_rep, 'style': style_latents}
            cond_gen_samples[key] = self.generate_from_latents(latents);
        return cond_gen_samples;


    def cond_generation_2a(self, latent_distribution_pairs, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size;

        style_latents = self.get_random_styles(num_samples);
        cond_gen_2a = dict();
        for p, pair in enumerate(latent_distribution_pairs.keys()):
            ld_pair = latent_distribution_pairs[pair];
            mu_list = []; logvar_list = [];
            for k, key in enumerate(ld_pair['latents'].keys()):
                mu_list.append(ld_pair['latents'][key][0].unsqueeze(0));
                logvar_list.append(ld_pair['latents'][key][1].unsqueeze(0));
            mus = torch.cat(mu_list, dim=0);
            logvars = torch.cat(logvar_list, dim=0);
            #weights_pair = utils.reweight_weights(torch.Tensor(ld_pair['weights']));
            #mu_joint, logvar_joint = self.modality_fusion(mus, logvars, weights_pair)
            mu_joint, logvar_joint = poe(mus, logvars);
            c_emb = utils.reparameterize(mu_joint, logvar_joint);
            l_2a = {'content': c_emb, 'style': style_latents};
            cond_gen_2a[pair] = self.generate_from_latents(l_2a);
        return cond_gen_2a;


    def save_networks(self):
        torch.save(self.encoder_m1.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m1))
        torch.save(self.decoder_m1.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m1))
        torch.save(self.encoder_m2.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m2))
        torch.save(self.decoder_m2.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m2))
        torch.save(self.encoder_m3.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m3))
        torch.save(self.decoder_m3.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m3))
