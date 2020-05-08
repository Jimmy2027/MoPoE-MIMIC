import sys,os
import numpy as np
import pandas as pd

import torch
from torchvision.utils import save_image

from utils.constants_svhnmnist import indices


def append_list_to_list_linear(l1, l2):
    for k in range(0, len(l2)):
        if isinstance(l2[k], str):
            l1.append(l2[k]);
        else:
            l1.append(l2[k].item());
    return l1;


def save_reparametrized_samples(samples, names, filename_samples, name_exra_column=None):
    num_features = samples.shape[1];
    columns = [];
    if name_exra_column is None:
        name_exra_column = 'image_id'
    columns.append(name_exra_column)
    for k in range(0, num_features):
        str_f = 'mu' + str(k).zfill(4);
        columns.append(str_f)

    df = pd.DataFrame(columns=columns);
    df[name_exra_column] = names;
    for k in range(0, num_features):
        df['mu' + str(k).zfill(4)] = samples[:,k];
    df.to_csv(filename_samples, index=False);


def tensor_to_text(alphabet, gen_t):
    gen_t = gen_t.cpu().data.numpy()
    gen_t = np.argmax(gen_t, axis=1)
    decoded_samples = []
    for i in range(len(gen_t)):
        decoded = seq2text(alphabet, gen_t[i])
        decoded_samples.append(tuple(decoded))
    return decoded_samples;

def seq2text(alphabet, seq):
    decoded = []
    for j in range(len(seq)):
        decoded.append(alphabet[seq[j]])
    return decoded

def write_samples_text_to_file(samples, filename):
    file_samples = open(filename, 'w');
    for k in range(0, len(samples)):
        file_samples.write(''.join(samples[k])[::-1] + '\n');
    file_samples.close();

def getText(samples):
    lines = []
    for k in range(0, len(samples)):
        lines.append(''.join(samples[k])[::-1])
    text = '\n\n'.join(lines)
    print(text)
    return text

def write_samples_img_to_file(samples, filename, img_per_row=1):
    save_image(samples.data.cpu(), filename, nrow=img_per_row);

def save_generated_samples_indiv(flags, batch_id, alphabet, real_samples, rand_samples, cond_samples):
    [imgs, txts] = real_samples;
    [img_rand_gen, text_rand_gen] = rand_samples;
    [img_cond_gen, text_cond_gen] = cond_samples;
    decoded_cond_gen_samples = tensor_to_text(alphabet, text_cond_gen);
    decoded_rand_gen_samples = tensor_to_text(alphabet, text_rand_gen);
    decoded_real_samples = tensor_to_text(alphabet, txts);
    cnt_samples = batch_id*flags.batch_size;
    for k in range(0, flags.batch_size):
        filename_sample_rand_gen_m1 = os.path.join(flags.dir_gen_eval_fid_random, 'random_sampling_' + str(cnt_samples).zfill(6) + '_img.png')
        filename_sample_rand_gen_m2 = os.path.join(flags.dir_gen_eval_fid_random, 'random_sampling_' + str(cnt_samples).zfill(6) + '_text.txt')
        filename_sample_cond_gen_m1 = os.path.join(flags.dir_gen_eval_fid_cond_gen, 'cond_gen_' + str(cnt_samples).zfill(6) + '_img.png')
        filename_sample_cond_gen_m2 = os.path.join(flags.dir_gen_eval_fid_cond_gen, 'cond_gen_' + str(cnt_samples).zfill(6) + '_text.txt')
        filename_sample_real_m1 = os.path.join(flags.dir_gen_eval_fid_real, 'real_' + str(cnt_samples).zfill(6) + '_img.png')
        filename_sample_real_m2 = os.path.join(flags.dir_gen_eval_fid_real, 'real_' + str(cnt_samples).zfill(6) + '_text.txt')

        save_image(img_rand_gen[k].data.cpu(), filename_sample_rand_gen_m1, nrow=1);
        write_samples_text_to_file(decoded_rand_gen_samples[k], filename_sample_rand_gen_m2);
        save_image(img_cond_gen[k].data.cpu(), filename_sample_cond_gen_m1, nrow=1);
        write_samples_text_to_file(decoded_cond_gen_samples[k], filename_sample_cond_gen_m2);
        save_image(imgs[k].data.cpu(), filename_sample_real_m1, nrow=1);
        write_samples_text_to_file(decoded_real_samples[k], filename_sample_real_m2);

        cnt_samples += 1;


def save_generated_samples_singlegroup(flags, batch_id, alphabet, group_name, samples):
    if group_name == 'real':
        dir_name = flags.dir_gen_eval_fid_real;
    elif group_name == 'random_sampling':
        dir_name = flags.dir_gen_eval_fid_random;
    elif group_name.startswith('dynamic_prior'):
        mod_store = flags.dir_gen_eval_fid_dynamicprior;
        dir_name = os.path.join(mod_store, '_'.join(group_name.split('_')[-2:]));
    elif group_name.startswith('cond_gen_1a2m'):
        mod_store = flags.dir_gen_eval_fid_cond_gen_1a2m;
        dir_name = os.path.join(mod_store, group_name.split('_')[-1]);
    elif group_name.startswith('cond_gen_2a1m'):
        mod_store = flags.dir_gen_eval_fid_cond_gen_2a1m;
        dir_name = os.path.join(mod_store, '_'.join(group_name.split('_')[-2:]));
    elif group_name == 'cond_gen':
        dir_name = flags.dir_gen_eval_fid_cond_gen;
    else:
        print('group name not defined....exit')
        sys.exit();

    for k, key in enumerate(samples.keys()):
        dir_f = os.path.join(dir_name, key);
        if not os.path.exists(dir_f):
            os.makedirs(dir_f);

    cnt_samples = batch_id * flags.batch_size;
    for k in range(0, flags.batch_size):
        for i, key in enumerate(samples.keys()):
            f_out = os.path.join(dir_name,  key, str(cnt_samples).zfill(6) + '.png')
            if key.startswith('img'):
                save_image(samples[key][k], f_out, nrow=1);
            elif key == 'text':
                write_samples_text_to_file(tensor_to_text(alphabet, samples[key][k].unsqueeze(0)), f_out);
        cnt_samples += 1;
