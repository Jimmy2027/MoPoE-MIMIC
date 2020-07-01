import sys
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import label_binarize

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from utils import utils
from utils.save_samples import write_samples_text_to_file

from mimic.constants import LABELS


def generate_random_samples_plots(flags, epoch, model, alphabet=None):
    img_size = torch.Size((1, flags.img_size, flags.img_size));
    random_samples = model.generate(100)
    samples_pa = random_samples['pa'];
    samples_lateral = random_samples['lateral'];
    samples_text = utils.text_to_pil(random_samples['text'], img_size, alphabet);
    fn_pa = os.path.join(flags.dir_random_samples, 'random_epoch_' +
                            str(epoch).zfill(4) + '_pa.png');
    plot_pa = utils.create_fig(fn_pa, samples_pa, 10);
    fn_lateral = os.path.join(flags.dir_random_samples, 'random_epoch_' +
                           str(epoch).zfill(4) + '_lateral.png');
    plot_lateral = utils.create_fig(fn_lateral, samples_lateral, 10);
    fn_text = os.path.join(flags.dir_random_samples, 'random_epoch_' +
                           str(epoch).zfill(4) + '_text.png');
    plot_text = utils.create_fig(fn_text, samples_text, 10);
    plots = {'pa': plot_pa,
             'lateral': plot_lateral,
             'text': plot_text};
    return plots;



def generate_swapping_plot(flags, epoch, model, samples, alphabet):
    rec_p_in_p_out = Variable(torch.zeros([121, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_p_in_l_out = Variable(torch.zeros([121, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_p_in_t_out = Variable(torch.zeros([121, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_l_in_p_out = Variable(torch.zeros([121, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_l_in_l_out = Variable(torch.zeros([121, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_l_in_t_out = Variable(torch.zeros([121, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_t_in_p_out = Variable(torch.zeros([121, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_t_in_l_out = Variable(torch.zeros([121, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_t_in_t_out = Variable(torch.zeros([121, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_p_in_p_out = rec_p_in_p_out.to(flags.device);
    rec_p_in_l_out = rec_p_in_l_out.to(flags.device);
    rec_p_in_t_out = rec_p_in_t_out.to(flags.device);
    rec_l_in_p_out = rec_l_in_p_out.to(flags.device);
    rec_l_in_l_out = rec_l_in_l_out.to(flags.device);
    rec_l_in_t_out = rec_l_in_t_out.to(flags.device);
    rec_t_in_p_out = rec_t_in_p_out.to(flags.device);
    rec_t_in_l_out = rec_t_in_l_out.to(flags.device);
    rec_t_in_t_out = rec_t_in_t_out.to(flags.device);

    img_size = torch.Size((1, flags.img_size, flags.img_size));
    for i in range(len(samples)):
        c_sample_pa = samples[i][0].squeeze();
        c_sample_lat = samples[i][1].squeeze();
        c_sample_text = utils.text_to_pil(samples[i][2].unsqueeze(0), img_size, alphabet);
        s_sample_pa = samples[i][0].squeeze();
        s_sample_lat = samples[i][1].squeeze();
        s_sample_text = utils.text_to_pil(samples[i][2].unsqueeze(0), img_size, alphabet);
        rec_p_in_p_out[i+1, :, :, :] = c_sample_pa;
        rec_p_in_p_out[(i + 1) * 11, :, :, :] = s_sample_pa;
        rec_p_in_l_out[i+1, :, :, :] = c_sample_pa;
        rec_p_in_l_out[(i + 1) * 11, :, :, :] = s_sample_lat;
        rec_p_in_t_out[i+1, :, :, :] = c_sample_pa;
        rec_p_in_t_out[(i + 1) * 11, :, :, :] = s_sample_text;
        rec_l_in_p_out[i+1, :, :, :] = c_sample_lat;
        rec_l_in_p_out[(i+1) * 11, :, :, :] = s_sample_pa;
        rec_l_in_l_out[i+1, :, :, :] = c_sample_lat;
        rec_l_in_l_out[(i+1) * 11, :, :, :] = s_sample_lat;
        rec_l_in_t_out[i+1, :, :, :] = c_sample_lat;
        rec_l_in_t_out[(i+1) * 11, :, :, :] = s_sample_text;
        rec_t_in_p_out[i+1, :, :, :] = c_sample_text;
        rec_t_in_p_out[(i + 1) * 11, :, :, :] = s_sample_pa;
        rec_t_in_l_out[i+1, :, :, :] = c_sample_text;
        rec_t_in_l_out[(i + 1) * 11, :, :, :] = s_sample_lat;
        rec_t_in_t_out[i+1, :, :, :] = c_sample_text;
        rec_t_in_t_out[(i + 1) * 11, :, :, :] = s_sample_text;

    # style transfer
    for i in range(len(samples)):
        for j in range(len(samples)):
            latents_style= model.inference(samples[i][0].unsqueeze(0),
                                              samples[i][1].unsqueeze(0),
                                              samples[i][2].unsqueeze(0))
            latents_content = model.inference(samples[j][0].unsqueeze(0),
                                            samples[j][1].unsqueeze(0),
                                            samples[j][2].unsqueeze(0))

            l_c_pa = latents_content['pa'];
            l_c_lat = latents_content['lateral'];
            l_c_text = latents_content['text'];
            l_s_pa = latents_style['pa'];
            l_s_lat = latents_style['lateral'];
            l_s_text = latents_style['text'];
            s_pa_emb = utils.reparameterize(mu=l_s_pa[0], logvar=l_s_pa[1]);
            c_pa_emb = utils.reparameterize(mu=l_c_pa[2], logvar=l_c_pa[3]);
            s_lat_emb = utils.reparameterize(mu=l_s_lat[0], logvar=l_s_lat[1]);
            c_lat_emb = utils.reparameterize(mu=l_c_lat[2], logvar=l_c_lat[3])
            c_text_emb = utils.reparameterize(mu=l_c_text[2], logvar=l_c_text[3]);
            s_text_emb = utils.reparameterize(mu=l_s_text[0], logvar=l_s_text[1]);
            style_emb = {'pa': s_pa_emb, 'lateral': s_lat_emb, 'text': s_text_emb}

            emb_content_pa = {'content': c_pa_emb, 'style': style_emb};
            emb_content_lat = {'content': c_lat_emb, 'style': style_emb}
            emb_content_text = {'content': c_text_emb, 'style': style_emb}
            pa_content_samples = model.generate_from_latents(emb_content_pa);
            lat_content_samples = model.generate_from_latents(emb_content_lat);
            text_content_samples = model.generate_from_latents(emb_content_text);
            p_in_p_out = pa_content_samples['pa'];
            p_in_l_out = pa_content_samples['lateral'];
            p_in_t_out = pa_content_samples['text'];
            l_in_p_out = lat_content_samples['pa'];
            l_in_l_out = lat_content_samples['lateral'];
            l_in_t_out = lat_content_samples['text'];
            t_in_p_out = text_content_samples['pa'];
            t_in_l_out = text_content_samples['lateral'];
            t_in_t_out = text_content_samples['text'];

            rec_p_in_p_out[(i+1) * 11 + (j+1), :, :, :] = p_in_p_out;
            rec_p_in_l_out[(i+1) * 11 + (j+1), :, :, :] = p_in_l_out;
            rec_l_in_p_out[(i+1) * 11 + (j+1), :, :, :] = l_in_p_out;
            rec_l_in_l_out[(i+1) * 11 + (j+1), :, :, :] = l_in_l_out;
            rec_p_in_t_out[(i+1) * 11 + (j+1), :, :, :] = utils.text_to_pil(p_in_t_out, img_size, alphabet);
            rec_l_in_t_out[(i+1) * 11 + (j+1), :, :, :] = utils.text_to_pil(l_in_t_out, img_size, alphabet);
            rec_t_in_p_out[(i+1) * 11 + (j+1), :, :, :] = t_in_p_out;
            rec_t_in_l_out[(i+1) * 11 + (j+1), :, :, :] = t_in_l_out;
            rec_t_in_t_out[(i+1) * 11 + (j+1), :, :, :] = utils.text_to_pil(t_in_t_out, img_size, alphabet);

    fp_p_in_p_out = os.path.join(flags.dir_swapping, 'swap_p_to_p_epoch_' + str(epoch).zfill(4) + '.png');
    fp_p_in_l_out = os.path.join(flags.dir_swapping, 'swap_p_to_l_epoch_' + str(epoch).zfill(4) + '.png');
    fp_p_in_t_out = os.path.join(flags.dir_swapping, 'swap_p_to_t_epoch_' + str(epoch).zfill(4) + '.png');
    fp_l_in_p_out = os.path.join(flags.dir_swapping, 'swap_l_to_p_epoch_' + str(epoch).zfill(4) + '.png');
    fp_l_in_l_out = os.path.join(flags.dir_swapping, 'swap_l_to_l_epoch_' + str(epoch).zfill(4) + '.png');
    fp_l_in_t_out = os.path.join(flags.dir_swapping, 'swap_l_to_t_epoch_' + str(epoch).zfill(4) + '.png');
    fp_t_in_p_out = os.path.join(flags.dir_swapping, 'swap_t_to_p_epoch_' + str(epoch).zfill(4) + '.png');
    fp_t_in_l_out = os.path.join(flags.dir_swapping, 'swap_t_to_l_epoch_' + str(epoch).zfill(4) + '.png');
    fp_t_in_t_out = os.path.join(flags.dir_swapping, 'swap_t_to_t_epoch_' + str(epoch).zfill(4) + '.png');
    plot_p_p = utils.create_fig(fp_p_in_p_out, rec_p_in_p_out, 11);
    plot_p_l = utils.create_fig(fp_p_in_l_out, rec_p_in_l_out, 11);
    plot_p_t = utils.create_fig(fp_p_in_t_out, rec_p_in_t_out, 11);
    plot_l_p = utils.create_fig(fp_l_in_p_out, rec_l_in_p_out, 11);
    plot_l_l = utils.create_fig(fp_l_in_l_out, rec_l_in_l_out, 11);
    plot_l_t = utils.create_fig(fp_l_in_t_out, rec_l_in_t_out, 11);
    plot_t_p = utils.create_fig(fp_t_in_p_out, rec_t_in_p_out, 11);
    plot_t_l = utils.create_fig(fp_t_in_l_out, rec_t_in_l_out, 11);
    plot_t_t = utils.create_fig(fp_t_in_t_out, rec_t_in_t_out, 11);
    plots_c_pa = {'pa': plot_p_p, 'lateral': plot_p_l, 'text': plot_p_t};
    plots_c_lat = {'pa': plot_l_p, 'lateral': plot_l_l, 'text': plot_l_t};
    plots_c_text = {'pa': plot_t_p, 'lateral': plot_t_l, 'text': plot_t_t};
    plots = {'pa': plots_c_pa, 'lateral': plots_c_lat, 'text': plots_c_text};
    return plots;


def generate_conditional_fig_2a(flags, epoch, model, samples, alphabet=None):
    rec_pt_in_p_out = Variable(torch.zeros([120, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_pt_in_l_out = Variable(torch.zeros([120, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_pt_in_t_out = Variable(torch.zeros([120, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_pl_in_p_out = Variable(torch.zeros([120, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_pl_in_l_out = Variable(torch.zeros([120, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_pl_in_t_out = Variable(torch.zeros([120, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_lt_in_p_out = Variable(torch.zeros([120, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_lt_in_l_out = Variable(torch.zeros([120, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_lt_in_t_out = Variable(torch.zeros([120, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_pt_in_p_out = rec_pt_in_p_out.to(flags.device);
    rec_pt_in_l_out = rec_pt_in_l_out.to(flags.device);
    rec_pt_in_t_out = rec_pt_in_t_out.to(flags.device);
    rec_pl_in_p_out = rec_pl_in_p_out.to(flags.device);
    rec_pl_in_l_out = rec_pl_in_l_out.to(flags.device);
    rec_pl_in_t_out = rec_pl_in_t_out.to(flags.device);
    rec_lt_in_p_out = rec_lt_in_p_out.to(flags.device);
    rec_lt_in_l_out = rec_lt_in_l_out.to(flags.device);
    rec_lt_in_t_out = rec_lt_in_t_out.to(flags.device);

    img_size = torch.Size((1, flags.img_size, flags.img_size));
    for i in range(len(samples)):
        c_sample_pa = samples[i][0];
        c_sample_lat = samples[i][1];
        c_sample_text = utils.text_to_pil(samples[i][2].unsqueeze(0), img_size, alphabet);
        rec_pt_in_p_out[i, :, :, :] = c_sample_pa;
        rec_pt_in_p_out[i+10, :, :, :] = c_sample_text;
        rec_pt_in_l_out[i, :, :, :] = c_sample_pa;
        rec_pt_in_l_out[i+10, :, :, :] = c_sample_text;
        rec_pt_in_t_out[i, :, :, :] = c_sample_pa;
        rec_pt_in_t_out[i+10, :, :, :] = c_sample_text;
        rec_lt_in_p_out[i, :, :, :] = c_sample_lat;
        rec_lt_in_p_out[i+10, :, :, :] = c_sample_text;
        rec_lt_in_l_out[i, :, :, :] = c_sample_lat;
        rec_lt_in_l_out[i+10, :, :, :] = c_sample_text;
        rec_lt_in_t_out[i, :, :, :] = c_sample_lat;
        rec_lt_in_t_out[i+10, :, :, :] = c_sample_text;
        rec_pl_in_p_out[i, :, :, :] = c_sample_pa;
        rec_pl_in_p_out[i+10, :, :, :] = c_sample_lat;
        rec_pl_in_l_out[i, :, :, :] = c_sample_pa;
        rec_pl_in_l_out[i+10, :, :, :] = c_sample_lat;
        rec_pl_in_t_out[i, :, :, :] = c_sample_pa;
        rec_pl_in_t_out[i+10, :, :, :] = c_sample_lat;

    # get style from random sampling
    zi_p = Variable(torch.randn(len(samples), flags.style_pa_dim));
    zi_l = Variable(torch.randn(len(samples), flags.style_lat_dim));
    zi_t = Variable(torch.randn(len(samples), flags.style_text_dim));
    zi_p = zi_p.to(flags.device);
    zi_l = zi_l.to(flags.device);
    zi_t = zi_t.to(flags.device);

    # style transfer
    for i in range(len(samples)):
        for j in range(len(samples)):
            l_pl = model.inference(input_pa=samples[j][0].unsqueeze(0),
                                   input_lat=samples[j][1].unsqueeze(0),
                                   input_text=None)
            l_pt = model.inference(input_pa=samples[j][0].unsqueeze(0),
                                   input_lat=None,
                                   input_text=samples[j][2].unsqueeze(0))
            l_lt = model.inference(input_pa=None,
                                   input_lat=samples[j][1].unsqueeze(0),
                                   input_text=samples[j][2].unsqueeze(0))
            c_pl = l_pl['joint'];
            c_pt = l_pt['joint'];
            c_lt = l_lt['joint'];
            emb_pl_c = utils.reparameterize(c_pl[0], c_pl[1]);
            emb_pt_c = utils.reparameterize(c_pt[0], c_pt[1]);
            emb_lt_c = utils.reparameterize(c_lt[0], c_lt[1]);

            if flags.factorized_representation:
                style = {'pa': zi_p[i].unsqueeze(0),
                         'lateral': zi_l[i].unsqueeze(0),
                         'text': zi_t[i].unsqueeze(0)};
            else:
                style = {'pa': None, 'lateral': None, 'text': None};
            emb_pl = {'content': emb_pl_c, 'style': style};
            emb_pt = {'content': emb_pt_c, 'style': style};
            emb_lt = {'content': emb_lt_c, 'style': style};
            pl_cond_gen = model.generate_from_latents(emb_pl);
            pt_cond_gen = model.generate_from_latents(emb_pt);
            lt_cond_gen = model.generate_from_latents(emb_lt);

            pl_in_p_out = pl_cond_gen['pa'];
            pl_in_l_out = pl_cond_gen['lateral'];
            pl_in_t_out = utils.text_to_pil(pl_cond_gen['text'], img_size, alphabet);
            pt_in_p_out = pt_cond_gen['pa'];
            pt_in_l_out = pt_cond_gen['lateral'];
            pt_in_t_out = utils.text_to_pil(pt_cond_gen['text'], img_size, alphabet);
            lt_in_p_out = lt_cond_gen['pa'];
            lt_in_l_out = lt_cond_gen['lateral'];
            lt_in_t_out = utils.text_to_pil(lt_cond_gen['text'], img_size, alphabet);
            rec_pl_in_p_out[(i + 2) * 10 + j, :, :, :] = pl_in_p_out;
            rec_pl_in_l_out[(i + 2) * 10 + j, :, :, :] = pl_in_l_out;
            rec_pl_in_t_out[(i + 2) * 10 + j, :, :, :] = pl_in_t_out;
            rec_pt_in_p_out[(i + 2) * 10 + j, :, :, :] = pt_in_p_out;
            rec_pt_in_l_out[(i + 2) * 10 + j, :, :, :] = pt_in_l_out;
            rec_pt_in_t_out[(i + 2) * 10 + j, :, :, :] = pt_in_t_out;
            rec_lt_in_p_out[(i + 2) * 10 + j, :, :, :] = lt_in_p_out;
            rec_lt_in_l_out[(i + 2) * 10 + j, :, :, :] = lt_in_l_out;
            rec_lt_in_t_out[(i + 2) * 10 + j, :, :, :] = lt_in_t_out;

    fp_pl_p = os.path.join(flags.dir_cond_gen_2a,
                           'cond_gen_pl_p_epoch_' + str(epoch).zfill(4) + '.png');
    fp_pl_l = os.path.join(flags.dir_cond_gen_2a,
                           'cond_gen_pl_l_epoch_' + str(epoch).zfill(4) + '.png');
    fp_pl_t = os.path.join(flags.dir_cond_gen_2a,
                           'cond_gen_pl_t_epoch_' + str(epoch).zfill(4) + '.png');
    fp_pt_p = os.path.join(flags.dir_cond_gen_2a,
                           'cond_gen_pt_p_epoch_' + str(epoch).zfill(4) + '.png');
    fp_pt_l = os.path.join(flags.dir_cond_gen_2a,
                           'cond_gen_pt_l_epoch_' + str(epoch).zfill(4) + '.png');
    fp_pt_t = os.path.join(flags.dir_cond_gen_2a,
                           'cond_gen_pt_t_epoch_' + str(epoch).zfill(4) + '.png');
    fp_lt_p = os.path.join(flags.dir_cond_gen_2a,
                           'cond_gen_lt_p_epoch_' + str(epoch).zfill(4) + '.png');
    fp_lt_l = os.path.join(flags.dir_cond_gen_2a,
                           'cond_gen_lt_l_epoch_' + str(epoch).zfill(4) + '.png');
    fp_lt_t = os.path.join(flags.dir_cond_gen_2a,
                           'cond_gen_lt_t_epoch_' + str(epoch).zfill(4) + '.png');
    plot_pl_p = utils.create_fig(fp_pl_p, rec_pl_in_p_out, 10);
    plot_pl_l = utils.create_fig(fp_pl_l, rec_pl_in_l_out, 10);
    plot_pl_t = utils.create_fig(fp_pl_t, rec_pl_in_t_out, 10);
    plot_pt_p = utils.create_fig(fp_pt_p, rec_pt_in_p_out, 10);
    plot_pt_l = utils.create_fig(fp_pt_l, rec_pt_in_l_out, 10);
    plot_pt_t = utils.create_fig(fp_pt_t, rec_pt_in_t_out, 10);
    plot_lt_p = utils.create_fig(fp_lt_p, rec_lt_in_p_out, 10);
    plot_lt_l = utils.create_fig(fp_lt_l, rec_lt_in_l_out, 10);
    plot_lt_t = utils.create_fig(fp_lt_t, rec_lt_in_t_out, 10);
    plots_pl = {'pa': plot_pl_p,
                'lateral': plot_pl_l,
                'text': plot_pl_t}
    plots_pt = {'pa': plot_pt_p,
                'lateral': plot_pt_l,
                'text': plot_pt_t}
    plots_lt = {'pa': plot_lt_p,
                'lateral': plot_lt_l,
                'text': plot_lt_t}
    plots = {'pa_lateral': plots_pl,
             'pa_text': plots_pt,
             'lateral_text': plots_lt};
    return plots;


def generate_conditional_fig_1a(flags, epoch, model, samples, alphabet):
    rec_p_in_p_out = Variable(torch.zeros([110, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_p_in_l_out = Variable(torch.zeros([110, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_p_in_t_out = Variable(torch.zeros([110, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_l_in_p_out = Variable(torch.zeros([110, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_l_in_l_out = Variable(torch.zeros([110, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_l_in_t_out = Variable(torch.zeros([110, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_t_in_p_out = Variable(torch.zeros([110, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_t_in_l_out = Variable(torch.zeros([110, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_t_in_t_out = Variable(torch.zeros([110, 1, flags.img_size, flags.img_size], dtype=torch.float32));
    # get style from random sampling
    rec_p_in_p_out = rec_p_in_p_out.to(flags.device);
    rec_p_in_l_out = rec_p_in_l_out.to(flags.device);
    rec_p_in_t_out = rec_p_in_t_out.to(flags.device);
    rec_l_in_p_out = rec_l_in_p_out.to(flags.device);
    rec_l_in_l_out = rec_l_in_l_out.to(flags.device);
    rec_l_in_t_out = rec_l_in_t_out.to(flags.device);
    rec_t_in_p_out = rec_t_in_p_out.to(flags.device);
    rec_t_in_l_out = rec_t_in_l_out.to(flags.device);
    rec_t_in_t_out = rec_t_in_t_out.to(flags.device);

    img_size = torch.Size((1, flags.img_size, flags.img_size));
    for i in range(len(samples)):
        c_sample_pa = samples[i][0].squeeze();
        c_sample_lat = samples[i][1].squeeze();
        c_sample_text = utils.text_to_pil(samples[i][2].unsqueeze(0), img_size, alphabet);
        rec_p_in_p_out[i, :, :, :] = c_sample_pa;
        rec_p_in_l_out[i, :, :, :] = c_sample_pa;
        rec_p_in_t_out[i, :, :, :] = c_sample_pa;
        rec_l_in_p_out[i, :, :, :] = c_sample_lat;
        rec_l_in_l_out[i, :, :, :] = c_sample_lat;
        rec_l_in_t_out[i, :, :, :] = c_sample_lat;
        rec_t_in_p_out[i, :, :, :] = c_sample_text;
        rec_t_in_l_out[i, :, :, :] = c_sample_text;
        rec_t_in_t_out[i, :, :, :] = c_sample_text;

    # get style from random sampling
    zi_pa = Variable(torch.randn(len(samples), flags.style_pa_dim));
    zi_lat = Variable(torch.randn(len(samples), flags.style_lat_dim));
    zi_text = Variable(torch.randn(len(samples), flags.style_text_dim));
    zi_pa = zi_pa.to(flags.device);
    zi_lat = zi_lat.to(flags.device);
    zi_text = zi_text.to(flags.device);
    # style transfer
    for i in range(len(samples)):
        for j in range(len(samples)):
            # get content from given modalities
            latents = model.inference(samples[j][0].unsqueeze(0),
                                      samples[j][1].unsqueeze(0),
                                      samples[j][2].unsqueeze(0))

            c_pa = latents['pa'][2:];
            c_lat = latents['lateral'][2:];
            c_text = latents['text'][2:];
            pa_rep = utils.reparameterize(mu=c_pa[0], logvar=c_pa[1]);
            lat_rep = utils.reparameterize(mu=c_lat[0], logvar=c_lat[1]);
            text_rep = utils.reparameterize(mu=c_text[0], logvar=c_text[1]);

            if flags.factorized_representation:
                style = {'pa': zi_pa[i].unsqueeze(0),
                         'lateral': zi_lat[i].unsqueeze(0),
                         'text': zi_text[i].unsqueeze(0)};
            else:
                style = {'pa': None, 'lateral': None, 'text': None};
            cond_pa = {'content': pa_rep, 'style': style};
            cond_lat = {'content': lat_rep, 'style': style};
            cond_text = {'content': text_rep, 'style': style};
            m1_cond_gen_samples = model.generate_from_latents(cond_pa);
            m2_cond_gen_samples = model.generate_from_latents(cond_lat);
            m3_cond_gen_samples = model.generate_from_latents(cond_text);

            p_in_p_out = m1_cond_gen_samples['pa'];
            p_in_l_out = m1_cond_gen_samples['lateral'];
            p_in_t_out = m1_cond_gen_samples['text'];
            l_in_p_out = m2_cond_gen_samples['pa'];
            l_in_l_out = m2_cond_gen_samples['lateral'];
            l_in_t_out = m2_cond_gen_samples['text'];
            t_in_p_out = m3_cond_gen_samples['pa'];
            t_in_l_out = m3_cond_gen_samples['lateral'];
            t_in_t_out = m3_cond_gen_samples['text'];

            rec_p_in_p_out[(i+1) * 10 + j, :, :, :] = p_in_p_out;
            rec_p_in_l_out[(i+1) * 10 + j, :, :, :] = p_in_l_out;
            rec_p_in_t_out[(i+1) * 10 + j, :, :, :] = utils.text_to_pil(p_in_t_out, img_size, alphabet);
            rec_l_in_p_out[(i+1) * 10 + j, :, :, :] = l_in_p_out;
            rec_l_in_l_out[(i+1) * 10 + j, :, :, :] = l_in_l_out;
            rec_l_in_t_out[(i+1) * 10 + j, :, :, :] = utils.text_to_pil(l_in_t_out, img_size, alphabet);
            rec_t_in_p_out[(i+1) * 10 + j, :, :, :] = t_in_p_out;
            rec_t_in_l_out[(i+1) * 10 + j, :, :, :] = t_in_l_out;
            rec_t_in_t_out[(i+1) * 10 + j, :, :, :] = utils.text_to_pil(t_in_t_out, img_size, alphabet);

    fp_p_in_p_out = os.path.join(flags.dir_cond_gen_1a, 'cond_gen_p_to_p_epoch_' + str(epoch).zfill(4) + '.png');
    fp_p_in_l_out = os.path.join(flags.dir_cond_gen_1a, 'cond_gen_p_to_l_epoch_' + str(epoch).zfill(4) + '.png');
    fp_p_in_t_out = os.path.join(flags.dir_cond_gen_1a, 'cond_gen_p_to_t_epoch_' + str(epoch).zfill(4) + '.png');
    fp_l_in_p_out = os.path.join(flags.dir_cond_gen_1a, 'cond_gen_l_to_p_epoch_' + str(epoch).zfill(4) + '.png');
    fp_l_in_l_out = os.path.join(flags.dir_cond_gen_1a, 'cond_gen_l_to_l_epoch_' + str(epoch).zfill(4) + '.png');
    fp_l_in_t_out = os.path.join(flags.dir_cond_gen_1a, 'cond_gen_l_to_t_epoch_' + str(epoch).zfill(4) + '.png');
    fp_t_in_p_out = os.path.join(flags.dir_cond_gen_1a, 'cond_gen_t_to_p_epoch_' + str(epoch).zfill(4) + '.png');
    fp_t_in_l_out = os.path.join(flags.dir_cond_gen_1a, 'cond_gen_t_to_l_epoch_' + str(epoch).zfill(4) + '.png');
    fp_t_in_t_out = os.path.join(flags.dir_cond_gen_1a, 'cond_gen_t_to_t_epoch_' + str(epoch).zfill(4) + '.png');
    plot_p_p = utils.create_fig(fp_p_in_p_out, rec_p_in_p_out, 10);
    plot_p_l = utils.create_fig(fp_p_in_l_out, rec_p_in_l_out, 10);
    plot_p_t = utils.create_fig(fp_p_in_t_out, rec_p_in_t_out, 10);
    plot_l_p = utils.create_fig(fp_l_in_p_out, rec_l_in_p_out, 10);
    plot_l_l = utils.create_fig(fp_l_in_l_out, rec_l_in_l_out, 10);
    plot_l_t = utils.create_fig(fp_l_in_t_out, rec_l_in_t_out, 10);
    plot_t_p = utils.create_fig(fp_t_in_p_out, rec_t_in_p_out, 10);
    plot_t_l = utils.create_fig(fp_t_in_l_out, rec_t_in_l_out, 10);
    plot_t_t = utils.create_fig(fp_t_in_t_out, rec_t_in_t_out, 10);
    cond_pa = {'pa': plot_p_p, 'lateral': plot_p_l, 'text': plot_p_t};
    cond_lat = {'pa': plot_l_p, 'lateral': plot_l_l, 'text': plot_l_t};
    cond_text = {'pa': plot_t_p, 'lateral': plot_t_l, 'text': plot_t_t};
    plots = {'pa': cond_pa, 'lateral': cond_lat, 'text': cond_text};
    return plots;


def classify_cond_gen_samples(flags, epoch, models, labels, samples):
    avg_precision = dict();
    for k, key in enumerate(models.keys()):
        mod_clf = models[key];
        mod_samples = samples[key]
        mod_attr = mod_clf(mod_samples);
        mod_pred_prob = mod_attr.cpu().data.numpy();
        mod_avg_prec_all_cl= dict();
        for l, label_str in enumerate(LABELS):
            mod_pred_prob_cl = mod_pred_prob[:,l];
            gt = labels[:,l].astype(int);
            mod_avg_precision = average_precision_score(gt, mod_pred_prob_cl.ravel());
            mod_avg_prec_all_cl[label_str] = mod_avg_precision;
            mod_fn_pred = os.path.join(flags.dir_gen_eval, label_str, 'prediction_prob_' + key + '_epoch_' + str(epoch).zfill(4) + '.npy')
            np.save(mod_fn_pred, mod_pred_prob_cl);
            filename_labels = os.path.join(flags.dir_gen_eval, label_str, 'labels_epoch_' + str(epoch).zfill(4) + '.npy')
            np.save(filename_labels, gt);
        avg_precision[key] = mod_avg_prec_all_cl;
    return avg_precision;


def train_clfs_latent_representation(data, labels):
    clf_all = dict();
    for l, label_str in enumerate(LABELS):
        gt = labels[:,l].astype(int).ravel();
        clf_lr_label = dict()
        for k, key in enumerate(data.keys()):
            rep = data[key];
            clf_lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000);
            clf_lr.fit(rep, gt);
            clf_lr_label[key] = clf_lr;
        clf_all[label_str] = clf_lr_label;
    return clf_all;


def classify_latent_representations(flags, epoch, clf_lr, data, labels):
    ap_all_rep = dict();
    for k, key in enumerate(data.keys()):
        ap_all_labels = dict();
        for l, label_str in enumerate(LABELS):
            gt = labels[:, l].astype(int)
            clf_lr_label = clf_lr[label_str];
            clf_lr_rep = clf_lr_label[key];
            y_pred_rep = clf_lr_rep.predict_proba(data[key]);
            ap_all_labels[label_str] = average_precision_score(gt, y_pred_rep[:,1].ravel())
            fn_pred_rep = os.path.join(flags.dir_inference, label_str,
                                       'prediction_prob_latent_rep_' + key + '_epoch_' + str(epoch).zfill(4) + '.npy')
            np.save(fn_pred_rep, y_pred_rep);
        ap_all_rep[key] = ap_all_labels;
    return ap_all_rep


def classify_rand_gen_samples(flags, epoch, models, samples):
    model_clf_pa = models['pa'];
    model_clf_lat = models['lateral'];
    model_clf_text = models['text'];
    random_pa = samples['pa'];
    random_lat = samples['lateral'];
    random_text = samples['text'];
    attr_hat_pa = model_clf_pa(random_pa);
    attr_hat_lat = model_clf_lat(random_lat);
    attr_hat_text = model_clf_text(random_text);
    pred_prob_gen_pa = attr_hat_pa.cpu().data.numpy();
    pred_prob_gen_lat = attr_hat_lat.cpu().data.numpy();
    pred_prob_gen_text = attr_hat_text.cpu().data.numpy();

    coherence_all = dict()
    for k,label_str in enumerate(LABELS):
        pred_prob_gen_pa_cl = pred_prob_gen_pa[:,k];
        pred_prob_gen_lat_cl = pred_prob_gen_lat[:,k];
        pred_prob_gen_text_cl = pred_prob_gen_text[:,k];
        pred_pa = np.argmax(pred_prob_gen_pa_cl).astype(int);
        pred_lat = np.argmax(pred_prob_gen_lat_cl).astype(int);
        pred_text = np.argmax(pred_prob_gen_text_cl).astype(int);
        coherence_m1_m2 = (pred_pa == pred_lat);
        coherence_m1_m3 = (pred_pa == pred_text);
        coherence_cl = (np.sum(coherence_m1_m2 == coherence_m1_m3) /
                        np.sum(np.ones(pred_pa.shape)));
        filename_coherence = os.path.join(flags.dir_gen_eval, label_str, 'random_coherence_epoch_' + str(epoch).zfill(4) + '.npy')
        np.save(filename_coherence, coherence_cl);
        coherence_all[label_str] = coherence_cl;
    return coherence_all;
