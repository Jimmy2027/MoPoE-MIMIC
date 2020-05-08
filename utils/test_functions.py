import os

import numpy as np
import glob

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from scipy.misc import imread

import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from fid.inception import InceptionV3
from fid.fid_score import get_activations
from fid.fid_score import calculate_frechet_distance

from utils import text as text
import prd_score.prd_score as prd


def fig2data(fig):
    """
    Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    Args:
        fig: fig a matplotlib figure

    Returns:
        buf: numpy 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
    return buf




def generate_samples(flags, vae_bimodal, step, alphabet):
    z_style_1 = Variable(torch.randn(flags.batch_size, flags.style_m1_dim));
    z_style_2 = Variable(torch.randn(flags.batch_size, flags.style_m2_dim));
    z_class = Variable(torch.randn(flags.batch_size, flags.class_dim));
    if flags.cuda:
        z_style_1 = z_style_1.cuda()
        z_class = z_class.cuda()
        z_style_2 = z_style_2.cuda()

    if flags.dataset == 'CelebA':
        z_style_1 = z_style_1.unsqueeze(-1);
        z_style_2 = z_style_2.unsqueeze(-1)
        z_class = z_class.unsqueeze(-1)

    if flags.cuda and torch.cuda.device_count() > 1:
        m1_samples = vae_bimodal.module.decoder_img(z_style_1, z_class)
        m2_samples = vae_bimodal.module.decoder_text(z_style_2, z_class)
    else:
        m1_samples = vae_bimodal.decoder_img(z_style_1, z_class)
        m2_samples= vae_bimodal.decoder_text(z_style_2, z_class)
    decoded_samples = text.tensor_to_text(alphabet, m2_samples);

    filename_samples_m1 = os.path.join(flags.dir_samples, 'samples_' + str(step).zfill(6) + '_img.png')
    save_image(torch.sigmoid(m1_samples).data.cpu(), filename_samples_m1, nrow=16);
    filename_samples_m2 = os.path.join(flags.dir_samples, 'samples_' + str(step).zfill(6) + '_text.txt')
    text.write_samples_text_to_file(decoded_samples, filename_samples_m2);

def generate_samples_nonfactorized(flags, vae_bimodal, step, alphabet):
    z_class = Variable(torch.randn(flags.batch_size, flags.class_dim));
    if flags.cuda:
        z_class = z_class.cuda()

    if flags.dataset == 'CelebA':
        z_class = z_class.unsqueeze(-1)

    m1_samples = vae_bimodal.decoder_img(None, z_class)
    m2_samples = vae_bimodal.decoder_text(None, z_class)
    decoded_samples = text.tensor_to_text(alphabet, m2_samples);

    filename_samples_m1 = os.path.join(flags.dir_samples, 'samples_' + str(step).zfill(6) + '_img.png')
    save_image(m1_samples.data.cpu(), filename_samples_m1, nrow=16);
    filename_samples_m2 = os.path.join(flags.dir_samples, 'samples_' + str(step).zfill(6) + '_text.txt')
    text.write_samples_text_to_file(decoded_samples, filename_samples_m2);


def calculate_inception_features_for_gen_evaluation(flags, paths, modality=None, dims=2048, batch_size=128):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx], path_state_dict=flags.inception_state_dict)
    if flags.cuda:
        model.cuda()

    if 'random' in paths:
        dir_rand_gen = paths['random'];
        if not os.path.exists(dir_rand_gen):
            raise RuntimeError('Invalid path: %s' % dir_rand_gen)
        if modality is not None:
            files_rand_gen = glob.glob(os.path.join(dir_rand_gen, modality, '*' + '.png'))
            filename_random = os.path.join(flags.dir_gen_eval_fid_random, 'random_sampling_' + modality + '_activations.npy');
        else:
            files_rand_gen = glob.glob(os.path.join(dir_rand_gen, '*.png'));
            filename_random = os.path.join(flags.dir_gen_eval_fid_random, 'random_img_activations.npy')
        act_rand_gen = get_activations(files_rand_gen, model, batch_size, dims, flags.cuda, verbose=False);
        np.save(filename_random, act_rand_gen);

    if 'dynamic_prior' in paths:
        dirs_dyn_prior = paths['dynamic_prior'];
        for k, key in enumerate(dirs_dyn_prior.keys()):
            if not os.path.exists(dirs_dyn_prior[key]):
                raise RuntimeError('Invalid path: %s' % dirs_dyn_prior[key])
            files_dyn_gen = glob.glob(os.path.join(dirs_dyn_prior[key], modality, '*' + '.png'))
            filename_dyn = os.path.join(dirs_dyn_prior[key], key + '_' + modality + '_activations.npy')
            act_cond_gen = get_activations(files_dyn_gen, model, batch_size, dims, flags.cuda, verbose=False);
            np.save(filename_dyn, act_cond_gen);

    if 'conditional' in paths:
        dir_cond_gen = paths['conditional'];
        if not os.path.exists(dir_cond_gen):
            raise RuntimeError('Invalid path: %s' % dir_cond_gen)
        if modality is not None:
            files_cond_gen = glob.glob(os.path.join(dir_cond_gen, modality, '*' + '.png'))
            filename_conditional = os.path.join(dir_cond_gen, 'cond_gen_' + modality + '_activations.npy')
        else:
            files_cond_gen = glob.glob(os.path.join(dir_cond_gen, '*.png'));
            filename_conditional = os.path.join(flags.dir_gen_eval_fid_cond_gen, 'conditional_img_activations.npy')
        act_cond_gen = get_activations(files_cond_gen, model, batch_size, dims, flags.cuda, verbose=False);
        np.save(filename_conditional, act_cond_gen);

    if 'conditional_2a1m' in paths:
        dirs_cond_gen = paths['conditional_2a1m'];
        for k, key in enumerate(dirs_cond_gen.keys()):
            if not os.path.exists(dirs_cond_gen[key]):
                raise RuntimeError('Invalid path: %s' % dirs_cond_gen[key])
            files_cond_gen = glob.glob(os.path.join(dirs_cond_gen[key], modality, '*' + '.png'))
            filename_conditional = os.path.join(dirs_cond_gen[key], key + '_' + modality + '_activations.npy')
            act_cond_gen = get_activations(files_cond_gen, model, batch_size, dims, flags.cuda, verbose=False);
            np.save(filename_conditional, act_cond_gen);

    if 'conditional_1a2m' in paths:
        dirs_cond_gen = paths['conditional_1a2m'];
        for k, key in enumerate(dirs_cond_gen.keys()):
            if not os.path.exists(dirs_cond_gen[key]):
                raise RuntimeError('Invalid path: %s' % dirs_cond_gen[key])
            files_cond_gen = glob.glob(os.path.join(dirs_cond_gen[key], modality, '*' + '.png'))
            filename_conditional = os.path.join(dirs_cond_gen[key], key + '_' + modality + '_activations.npy')
            act_cond_gen = get_activations(files_cond_gen, model, batch_size, dims, flags.cuda, verbose=False);
            np.save(filename_conditional, act_cond_gen);

    if 'real' in paths:
        dir_real = paths['real'];
        if not os.path.exists(dir_real):
            raise RuntimeError('Invalid path: %s' % dir_real)
        if modality is not None:
            files_real = glob.glob(os.path.join(dir_real, modality, '*' + '.png'));
            filename_real = os.path.join(flags.dir_gen_eval_fid_real, 'real_' + modality + '_activations.npy');
        else:
            files_real = glob.glob(os.path.join(dir_real, '*.png'));
            filename_real = os.path.join(flags.dir_gen_eval_fid_real, 'real_img_activations.npy')
        act_real = get_activations(files_real, model, batch_size, dims, flags.cuda, verbose=False);
        np.save(filename_real, act_real);


def calculate_clf_img_features_for_gen_evaluation(flags, model_img, paths, batch_size, cuda):
    if 'random' in paths:
        dir_rand_gen = paths['random'];
        if not os.path.exists(dir_rand_gen):
            raise RuntimeError('Invalid path: %s' % dir_rand_gen)
        files_rand_gen = glob.glob(os.path.join(dir_rand_gen, '*.png'));
        act_rand_gen = model_img.get_activations(files_rand_gen, batch_size, cuda);
        filename_random = os.path.join(flags.dir_gen_eval_fid_random, 'random_img_activations.npy')
        np.save(filename_random, act_rand_gen);

    if 'conditional' in paths:
        dir_cond_gen = paths['conditional'];
        if not os.path.exists(dir_cond_gen):
            raise RuntimeError('Invalid path: %s' % dir_cond_gen)
        files_cond_gen = glob.glob(os.path.join(dir_cond_gen, '*.png'));
        act_cond_gen = get_activations(files_cond_gen, batch_size, cuda);
        filename_conditional = os.path.join(flags.dir_gen_eval_fid_cond_gen, 'conditional_img_activations.npy')
        np.save(filename_conditional, act_cond_gen);

    if 'real' in paths:
        dir_real = paths['real'];
        if not os.path.exists(dir_real):
            raise RuntimeError('Invalid path: %s' % dir_real)
        files_real = glob.glob(os.path.join(dir_real, '*.png'));
        act_real = get_activations(files_real, batch_size, cuda, verbose=False);
        filename_real = os.path.join(flags.dir_gen_eval_fid_real, 'real_img_activations.npy')
        np.save(filename_real, act_real);


def load_inception_activations(flags, modality=None, num_modalities=2, conditionals=None):
    if modality is None:
        filename_real = os.path.join(flags.dir_gen_eval_fid_real, 'real_img_activations.npy');
        filename_random = os.path.join(flags.dir_gen_eval_fid_random, 'random_img_activations.npy')
        filename_conditional = os.path.join(flags.dir_gen_eval_fid_cond_gen, 'conditional_img_activations.npy')
        feats_real = np.load(filename_real);
        feats_random = np.load(filename_random);
        feats_cond = np.load(filename_conditional);
        feats = [feats_real, feats_random, feats_cond];
    else:
        filename_real = os.path.join(flags.dir_gen_eval_fid_real, 'real_' + modality + '_activations.npy');
        filename_random = os.path.join(flags.dir_gen_eval_fid_random, 'random_sampling_' + modality + '_activations.npy')
        feats_real = np.load(filename_real);
        feats_random = np.load(filename_random);

        if num_modalities == 2:
            filename_cond_gen = os.path.join(flags.dir_gen_eval_fid_cond_gen, 'cond_gen_' + modality + '_activations.npy')
            feats_cond_gen = np.load(filename_cond_gen);
            feats = [feats_real, feats_random, feats_cond_gen];
        elif num_modalities > 2:
            if conditionals is None:
                raise RuntimeError('conditionals are needed for num(M) > 2...')
            feats_cond_1a2m = dict()
            for k, key in enumerate(conditionals[0].keys()):
                filename_cond_1a2m = os.path.join(conditionals[0][key], key + '_' + modality + '_activations.npy')
                feats_cond_key = np.load(filename_cond_1a2m);
                feats_cond_1a2m[key] = feats_cond_key

            feats_cond_2a1m = dict()
            for k, key in enumerate(conditionals[1].keys()):
                filename_cond_1a2m = os.path.join(conditionals[1][key], key + '_' + modality + '_activations.npy')
                feats_cond_key = np.load(filename_cond_1a2m);
                feats_cond_2a1m[key] = feats_cond_key

            if flags.modality_jsd:
                if conditionals is None:
                    raise RuntimeError('conditionals are needed for num(M) > 2...')
                feats_cond_dyn_prior_2a1m = dict()
                for k, key in enumerate(conditionals[2].keys()):
                    filename_dp_2a1m = os.path.join(conditionals[2][key], key + '_' + modality + '_activations.npy')
                    feats_dp_key = np.load(filename_dp_2a1m);
                    feats_cond_dyn_prior_2a1m[key] = feats_dp_key
            else:
                feats_cond_dyn_prior_2a1m = None;

            feats = [feats_real, feats_random, feats_cond_1a2m, feats_cond_2a1m, feats_cond_dyn_prior_2a1m];
        else:
            print('combinations of feature names and number of modalities is not correct');
    return feats;

def calculate_fid(feats_real, feats_gen):
    mu_real = np.mean(feats_real, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    mu_gen = np.mean(feats_gen, axis=0)
    sigma_gen = np.cov(feats_gen, rowvar=False)
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    return fid;


def calculate_fid_dict(feats_real, dict_feats_gen):
    dict_fid = dict();
    for k, key in enumerate(dict_feats_gen.keys()):
        feats_gen = dict_feats_gen[key];
        dict_fid[key] = calculate_fid(feats_real, feats_gen);
    return dict_fid;


def calculate_prd(feats_real, feats_gen):
    prd_val = prd.compute_prd_from_embedding(feats_real, feats_gen)
    ap = np.mean(prd_val);
    return ap;


def calculate_prd_dict(feats_real, dict_feats_gen):
    dict_fid = dict();
    for k, key in enumerate(dict_feats_gen.keys()):
        feats_gen = dict_feats_gen[key];
        dict_fid[key] = calculate_prd(feats_real, feats_gen);
    return dict_fid;


def get_clf_activations(flags, data, model):
    model.eval();
    act = model.get_activations(data);
    act = act.cpu().data.numpy().reshape(flags.batch_size, -1)
    return act;









