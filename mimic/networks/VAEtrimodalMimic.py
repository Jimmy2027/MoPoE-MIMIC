
import os

import torch
import torch.nn as nn

from mimic.networks.ConvNetworksImgMimic import EncoderImg, DecoderImg
from mimic.networks.ConvNetworksTextMimic import EncoderText, DecoderText

from divergence_measures.mm_div import calc_alphaJSD_modalities
from divergence_measures.mm_div import calc_group_divergence_poe
from divergence_measures.mm_div import calc_group_divergence_moe
from divergence_measures.mm_div import calc_kl_divergence
from divergence_measures.mm_div import poe

from utils import utils


class VAEtrimodalMimic(nn.Module):
    def __init__(self, flags):
        super(VAEtrimodalMimic, self).__init__()
        self.num_modalities = 3;
        self.flags = flags;
        self.encoder_pa = EncoderImg(flags, flags.style_pa_dim)
        self.encoder_lat = EncoderImg(flags, flags.style_lat_dim)
        self.encoder_text = EncoderText(flags, flags.style_text_dim)
        self.decoder_pa = DecoderImg(flags, flags.style_pa_dim);
        self.decoder_lat = DecoderImg(flags, flags.style_lat_dim);
        self.decoder_text = DecoderText(flags, flags.style_text_dim);
        self.encoder_pa = self.encoder_pa.to(flags.device);
        self.encoder_lat = self.encoder_lat.to(flags.device);
        self.encoder_text = self.encoder_text.to(flags.device);
        self.decoder_pa = self.decoder_pa.to(flags.device);
        self.decoder_lat = self.decoder_lat.to(flags.device);
        self.decoder_text = self.decoder_text.to(flags.device);
        self.lhood_pa = utils.get_likelihood(flags.likelihood_m1);
        self.lhood_lat = utils.get_likelihood(flags.likelihood_m2);
        self.lhood_text = utils.get_likelihood(flags.likelihood_m3);

        d_size_m1 = flags.img_size*flags.img_size;
        d_size_m2 = flags.img_size*flags.img_size;
        d_size_m3 = flags.len_sequence;
        total_d_size = d_size_m1 + d_size_m2 + d_size_m3;
        w1 = 1.0;
        w2 = d_size_m1/d_size_m2;
        w3 = d_size_m1/d_size_m3;
        w_total = w1+w2+w3;
        self.rec_w1 = w1;
        self.rec_w2 = w2;
        self.rec_w3 = w3;
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


    def forward(self, input_pa=None, input_lat=None, input_text=None):
        latents = self.inference(input_pa, input_lat, input_text);
        results = dict();
        results['latents'] = latents;
        div = self.calc_joint_divergence(latents['mus'],
                                         latents['logvars'],
                                         latents['weights']);
        for k, key in enumerate(div.keys()):
            results[key] = div[key];

        results['group_distr'] = latents['joint'];
        class_embeddings = utils.reparameterize(latents['joint'][0],
                                                latents['joint'][1]);

        results_rec = dict();
        if input_pa is not None:
            m1_s_mu, m1_s_logvar = latents['pa'][:2];
            if self.flags.factorized_representation:
                m1_s_embeddings = utils.reparameterize(mu=m1_s_mu, logvar=m1_s_logvar);
            else:
                m1_s_embeddings = None;
            m1_rec = self.lhood_pa(*self.decoder_pa(m1_s_embeddings, class_embeddings));
            results_rec['pa'] = m1_rec;
        if input_lat is not None:
            m2_s_mu, m2_s_logvar = latents['lateral'][:2];
            if self.flags.factorized_representation:
                m2_s_embeddings = utils.reparameterize(mu=m2_s_mu, logvar=m2_s_logvar);
            else:
                m2_s_embeddings = None;
            m2_rec = self.lhood_lat(*self.decoder_lat(m2_s_embeddings, class_embeddings));
            results_rec['lateral'] = m2_rec;
        if input_text is not None:
            m3_s_mu, m3_s_logvar = latents['text'][:2];
            if self.flags.factorized_representation:
                m3_s_embeddings = utils.reparameterize(mu=m3_s_mu, logvar=m3_s_logvar);
            else:
                m3_s_embeddings = None;
            m3_rec = self.lhood_text(*self.decoder_text(m3_s_embeddings, class_embeddings));
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
            weights=self.weights;
        weights = weights.clone();
        weights = utils.reweight_weights(weights);
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
        mu_moe, logvar_moe = utils.mixture_component_selection(self.flags,
                                                               mus,
                                                               logvars,
                                                               weights);
        return [mu_moe, logvar_moe];


    def poe_fusion(self, mus, logvars, weights=None):
        mu_poe, logvar_poe = poe(mus, logvars);
        return [mu_poe, logvar_poe];


    def encode(self, i_pa=None, i_lat=None, i_text=None):
        latents = dict();
        if i_pa is not None:
            latents['pa'] = self.encoder_pa(i_pa)
            latents['pa_style'] = latents['pa'][:2]
            latents['pa'] = latents['pa'][2:]
        else:
            latents['pa_style'] = [None, None]
            latents['pa'] = [None, None]
        if i_lat is not None:
            latents['lateral'] = self.encoder_lat(i_lat)
            latents['lateral_style'] = latents['lateral'][:2]
            latents['lateral'] = latents['lateral'][2:]
        else:
            latents['lateral_style'] = [None, None]
            latents['lateral'] = [None, None]
        if i_text is not None:
            latents['text'] = self.encoder_text(i_lat)
            latents['text_style'] = latents['text'][:2]
            latents['text'] = latents['text'][2:]
        else:
            latents['text_style'] = [None, None]
            latents['text'] = [None, None]
        return latents;


    def inference(self, input_pa=None, input_lat=None, input_text=None):
        latents = self.encode(i_pa=input_pa,
                              i_lat=input_lat,
                              i_text=input_text);
        if input_pa is not None:
            num_samples = input_pa.shape[0];
        if input_lat is not None:
            num_samples = input_lat.shape[0];
        if input_text is not None:
            num_samples = input_text.shape[0];
        weights = [self.weights[0]];
        mus = torch.zeros(1, num_samples, self.flags.class_dim).to(self.flags.device);
        logvars = torch.zeros(1, num_samples, self.flags.class_dim).to(self.flags.device);
        if input_pa is not None:
            pa_mu, pa_logvar = latents['pa'][2:];
            mus = torch.cat([mus, pa_mu.unsqueeze(0)], dim=0);
            logvars = torch.cat([logvars, pa_logvar.unsqueeze(0)], dim=0);
            weights.append(self.weights[1]);
        if input_lat is not None:
            lat_mu, lat_logvar = latents['lateral'][2:];
            mus = torch.cat([mus, lat_mu.unsqueeze(0)], dim=0);
            logvars = torch.cat([logvars, lat_logvar.unsqueeze(0)], dim=0);
            weights.append(self.weights[2]);
        if input_text is not None:
            text_mu, text_logvar = latents['text'][2:];
            mus = torch.cat([mus, text_mu.unsqueeze(0)], dim=0);
            logvars = torch.cat([logvars, text_logvar.unsqueeze(0)], dim=0);
            weights.append(self.weights[3]);
        if input_pa is not None and input_lat is not None:
            poe_pa_lat = poe(torch.cat([pa_mu.unsqueeze(0),
                                        lat_mu.unsqueeze(0)], dim=0),
                             torch.cat([pa_logvar.unsqueeze(0),
                                        lat_logvar.unsqueeze(0)], dim=0))
            latents['pa_lat'] = poe_pa_lat;
            mus = torch.cat([mus, poe_pa_lat[0].unsqueeze(0)], dim=0);
            logvars = torch.cat([logvars, poe_pa_lat[1].unsqueeze(0)], dim=0);
        if input_pa is not None and input_text is not None:
            poe_pa_text = poe(torch.cat([pa_mu.unsqueeze(0),
                                         text_mu.unsqueeze(0)], dim=0),
                              torch.cat([pa_logvar.unsqueeze(0),
                                         text_logvar.unsqueeze(0)], dim=0))
            latents['pa_text'] = poe_pa_text;
            mus = torch.cat([mus, poe_pa_text[0].unsqueeze(0)], dim=0);
            logvars = torch.cat([logvars, poe_pa_text[1].unsqueeze(0)], dim=0);
        if input_lat is not None and input_text is not None:
            poe_lat_text = poe(torch.cat([lat_mu.unsqueeze(0),
                                          text_mu.unsqueeze(0)], dim=0),
                               torch.cat([lat_logvar.unsqueeze(0),
                                          text_logvar.unsqueeze(0)], dim=0))
            latents['lat_text'] = poe_lat_text;
            mus = torch.cat([mus, poe_lat_text[0].unsqueeze(0)], dim=0);
            logvars = torch.cat([logvars, poe_lat_text[1].unsqueeze(0)], dim=0);
        if input_pa is not None and input_lat is not None and input_text is not None:
            poe_pa_lat_text = poe(torch.cat([pa_mu.unsqueeze(0),
                                             lat_mu.unsqueeze(0),
                                             text_mu.unsqueeze(0)], dim=0),
                                  torch.cat([pa_logvar.unsqueeze(0),
                                             lat_logvar.unsqueeze(0),
                                             text_logvar.unsqueeze(0)], dim=0))
            latents['pa_lat_text'] = poe_pa_lat_text;
            mus = torch.cat([mus, poe_pa_lat_text[0].unsqueeze(0)], dim=0);
            logvars = torch.cat([logvars, poe_pa_lat_text[1].unsqueeze(0)], dim=0);

        weights = (1/float(mus.shape[0]))*torch.ones(mus.shape[0]).to(self.flags.device);
        joint_mu, joint_logvar = self.modality_fusion(mus, logvars, weights);
        latents['mus'] = mus;
        latents['logvars'] = logvars;
        latents['weights'] = weights;
        latents['joint'] = [joint_mu, joint_logvar];
        return latents;


    def get_random_styles(self, num_samples):
        if self.flags.factorized_representation:
            z_style_1 = torch.randn(num_samples, self.flags.style_pa_dim);
            z_style_2 = torch.randn(num_samples, self.flags.style_lat_dim);
            z_style_3 = torch.randn(num_samples, self.flags.style_text_dim);
            z_style_1 = z_style_1.to(self.flags.device);
            z_style_2 = z_style_2.to(self.flags.device);
            z_style_3 = z_style_3.to(self.flags.device);
        else:
            z_style_1 = None;
            z_style_2 = None;
            z_style_3 = None;
        styles = {'pa': z_style_1, 'lateral': z_style_2, 'text': z_style_3};
        return styles;


    def get_random_style_dists(self, num_samples):
        s1_mu = torch.zeros(num_samples,
                            self.flags.style_pa_dim).to(self.flags.device)
        s1_logvar = torch.zeros(num_samples,
                                self.flags.style_pa_dim).to(self.flags.device);
        s2_mu = torch.zeros(num_samples,
                            self.flags.style_lat).to(self.flags.device)
        s2_logvar = torch.zeros(num_samples,
                                self.flags.style_lat_dim).to(self.flags.device);
        s3_mu = torch.zeros(num_samples,
                            self.flags.style_text_dim).to(self.flags.device)
        s3_logvar = torch.zeros(num_samples,
                                self.flags.style_text_dim).to(self.flags.device);
        m1_dist = [s1_mu, s1_logvar];
        m2_dist = [s2_mu, s2_logvar];
        m3_dist = [s3_mu, s3_logvar];
        styles = {'pa': m1_dist, 'lateral': m2_dist, 'text': m3_dist};
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
        cond_gen_pa = suff_stats['pa'].mean;
        cond_gen_lat = suff_stats['lateral'].mean;
        cond_gen_text = suff_stats['text'].mean;
        cond_gen = {'pa': cond_gen_pa,
                    'lateral': cond_gen_lat,
                    'text': cond_gen_text};
        return cond_gen;


    def generate_sufficient_statistics_from_latents(self, latents):
        style_pa = latents['style']['pa'];
        style_lat = latents['style']['lateral'];
        style_text = latents['style']['text'];
        content = latents['content']
        cond_gen_m1 = self.lhood_pa(*self.decoder_pa(style_pa, content));
        cond_gen_m2 = self.lhood_lat(*self.decoder_lat(style_lat, content));
        cond_gen_m3 = self.lhood_text(*self.decoder_text(style_text, content));
        return {'pa': cond_gen_m1, 'lateral': cond_gen_m2, 'text': cond_gen_m3}


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


    def cond_generation_2a(self, latent_distribution_pairs, num_samples=None,
                           dp_gen=False):
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
            weights_pair = utils.reweight_weights(torch.Tensor(ld_pair['weights']));
            if dp_gen:
                mu_joint, logvar_joint = poe(mus, logvars);
            else:
                mu_joint, logvar_joint = self.modality_fusion(mus, logvars, weights_pair)
            c_emb = utils.reparameterize(mu_joint, logvar_joint);
            l_2a = {'content': c_emb, 'style': style_latents};
            cond_gen_2a[pair] = self.generate_from_latents(l_2a);
        return cond_gen_2a;


    def save_networks(self):
        torch.save(self.encoder_pa.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m1))
        torch.save(self.decoder_pa.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m1))
        torch.save(self.encoder_lat.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m2))
        torch.save(self.decoder_lat.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m2))
        torch.save(self.encoder_text.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m3))
        torch.save(self.decoder_text.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m3))
