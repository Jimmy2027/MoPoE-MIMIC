import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.distributions as dist
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.transforms import Compose, ToTensor

from PIL import ImageDraw

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from utils import text as text


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def get_likelihood(str):
    if str == 'laplace':
        pz = dist.Laplace;
    elif str == 'bernoulli':
        pz = dist.Bernoulli;
    elif str == 'normal':
        pz = dist.Normal;
    elif str == 'categorical':
        pz = dist.OneHotCategorical;
    else:
        print('likelihood not implemented')
        pz = None;
    return pz;


def reparameterize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    return eps.mul(std).add_(mu)


def reparameterize_mixture(flags, training, mus, logvars, indices=None):
    uniform_mu = torch.zeros(mus[0].shape)
    uniform_logvar = torch.zeros(logvars[0].shape)
    reparams_comp = torch.zeros(mus[0].shape)
    w_modalities = torch.Tensor(flags.alpha_modalities);
    if indices is not None:
        w_modalities = w_modalities[indices];
    if w_modalities.sum() != 1.0:
        w_modalities = w_modalities / w_modalities.sum();
    if flags.cuda:
        uniform_mu = uniform_mu.cuda();
        uniform_logvar = uniform_logvar.cuda();
        w_modalities = w_modalities.cuda();
        reparams_comp = reparams_comp.cuda();
    mu = [uniform_mu];
    logvar = [uniform_logvar];
    for k in range(mus.shape[0]):
        mu.append(mus[k])
        logvar.append(logvars[k])
    for k in range(0, flags.batch_size):
        m = dist.categorical.Categorical(w_modalities.clone().detach().requires_grad_(True))
        ind_distr = m.sample()
        std = torch.exp(0.5 * logvar[ind_distr][k,:])
        eps = torch.randn_like(std)
        reparams_comp[k,:] = mu[ind_distr][k,:] + eps * std;
    return reparams_comp;


def weights_init(layer):
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()


def imshow_grid(images, shape=[2, 8], name='default', save=False):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    if save:
        plt.savefig('reconstructed_images/' + str(name) + '.png')
        plt.clf()
    else:
        plt.show()


def plot_confusion_matrix(ax, y_true, y_pred, classes,
                          normalize=False,
                          cmap=plt.cm.Blues,
                          font_size=16):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        try:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        except RuntimeWarning:
            cm = 0.0;

    # fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           )
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    size = font_size,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return ax


def reweight_weights(w):
    w = w / w.sum();
    return w;


def mixture_component_selection(flags, mus, logvars, w_modalities=None, num_samples=None):
    #if not defined, take pre-defined weights
    if num_samples is None:
        num_samples = flags.batch_size;

    if w_modalities is None:
        w_modalities = torch.Tensor(flags.alpha_modalities);
        w_modalities = w_modalities.to(flags.device);
    idx_start = [];
    idx_end = []
    for k in range(0, w_modalities.shape[0]):
        if k == 0:
            i_start = 0;
        else:
            i_start = int(idx_end[k-1]);
        if k == w_modalities.shape[0]-1:
            i_end = num_samples;
        else:
            i_end = i_start + int(torch.floor(num_samples*w_modalities[k]));
        idx_start.append(i_start);
        idx_end.append(i_end);

    idx_end[-1] = num_samples;

    mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])]);
    logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])]);
    return [mu_sel, logvar_sel];


def calc_elbo(flags, modality, recs, klds):
    kld_content = klds['content'];
    if modality == 'joint':
        weighted_style_kld = 0.0;
        weighted_rec = 0.0;
        klds_style = klds['style']
        for i, key in enumerate(klds_style.keys()):
            if key == 'img_mnist':
                weighted_style_kld += flags.beta_m1_style * klds_style['img_mnist'];
                weighted_rec += flags.rec_weight_m1 * recs['img_mnist'];
            elif key == 'img_svhn':
                weighted_style_kld += flags.beta_m2_style * klds_style['img_svhn'];
                weighted_rec += flags.rec_weight_m2 * recs['img_svhn'];
            elif key =='text':
                weighted_style_kld += flags.beta_m3_style * klds_style['text'];
                weighted_rec += flags.rec_weight_m3 * recs['text'];
        kld_style = weighted_style_kld;
        rec_error = weighted_rec;
    elif modality == 'img_mnist' or modality == 'img_svhn' or modality == 'text':
        if modality == 'img_mnist':
            beta_style_mod = flags.beta_m1_style;
            rec_weight_mod = flags.rec_weight_m1;
        elif modality == 'img_svhn':
            beta_style_mod = flags.beta_m2_style;
            rec_weight_mod = flags.rec_weight_m2;
        elif modality == 'text':
            beta_style_mod = flags.beta_m3_style;
            rec_weight_mod = flags.rec_weight_m3;
        kld_style = beta_style_mod * klds['style'][modality];
        rec_error = rec_weight_mod * recs[modality];
    div = flags.beta_content * kld_content + flags.beta_style * kld_style;
    elbo = rec_error + flags.beta * div;
    return elbo;


def calc_elbo_celeba(flags, modality, recs, klds):
    kld_content = klds['content'];
    if modality == 'joint':
        weighted_style_kld = 0.0;
        weighted_rec = 0.0;
        klds_style = klds['style']
        for i, key in enumerate(klds_style.keys()):
            if key == 'img_celeba':
                weighted_style_kld += flags.beta_m1_style * klds_style['img_celeba'];
                weighted_rec += flags.rec_weight_m1 * recs['img_celeba'];
            elif key == 'text':
                weighted_style_kld += flags.beta_m2_style * klds_style['text'];
                weighted_rec += flags.rec_weight_m2 * recs['text'];
        kld_style = weighted_style_kld;
        rec_error = weighted_rec;
    elif modality == 'img_celeba' or modality == 'text':
        if modality == 'img_celeba':
            beta_style_mod = flags.beta_m1_style;
            rec_weight_mod = flags.rec_weight_m1;
        elif modality == 'text':
            beta_style_mod = flags.beta_m2_style;
            rec_weight_mod = flags.rec_weight_m2;
        kld_style = beta_style_mod * klds['style'][modality];
        rec_error = rec_weight_mod * recs[modality];
    div = flags.beta_content * kld_content + flags.beta_style * kld_style;
    elbo = rec_error + flags.beta * div;
    return elbo;


def save_and_log_flags(flags):
    #filename_flags = os.path.join(flags.dir_experiment_run, 'flags.json')
    #with open(filename_flags, 'w') as f:
    #    json.dump(flags.__dict__, f, indent=2, sort_keys=True)

    filename_flags_rar = os.path.join(flags.dir_experiment_run, 'flags.rar')
    torch.save(flags, filename_flags_rar);

    str_args = '';
    for k, key in enumerate(sorted(flags.__dict__.keys())):
        str_args = str_args + key + ': ' + str(flags.__dict__[key]) + '\n' + ';';
    return str_args;



def create_fig(fn, img_data, num_img_row):
    save_image(img_data.data.cpu(), fn, nrow=num_img_row);
    grid = make_grid(img_data, nrow=num_img_row);
    plot = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy();
    return plot;


