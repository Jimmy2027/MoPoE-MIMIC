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


    def generate_sufficient_statistics_from_latents(self, latents):
        style_m1 = latents['style']['mnist'];
        style_m2 = latents['style']['svhn'];
        style_m3 = latents['style']['text'];
        content = latents['content']
        cond_gen_m1 = self.lhood_m1(*self.decoder_m1(style_m1, content));
        cond_gen_m2 = self.lhood_m2(*self.decoder_m2(style_m2, content));
        cond_gen_m3 = self.lhood_m3(*self.decoder_m3(style_m3, content));
        return {'mnist': cond_gen_m1, 'svhn': cond_gen_m2, 'text': cond_gen_m3}


    def save_networks(self):
        torch.save(self.encoder_m1.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m1))
        torch.save(self.decoder_m1.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m1))
        torch.save(self.encoder_m2.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m2))
        torch.save(self.decoder_m2.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m2))
        torch.save(self.encoder_m3.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m3))
        torch.save(self.decoder_m3.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m3))
