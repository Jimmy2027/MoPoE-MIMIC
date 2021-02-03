# HK, 30.01.21
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import mimic
from mimic.utils import text as text
from mimic.utils.experiment import MimicExperiment
from mimic.utils.filehandling import get_config_path
from mimic.utils.flags import parser
from mimic.utils.flags import update_flags_with_config
from dataclasses import dataclass

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

FLAGS = parser.parse_args()

config_path = get_config_path(FLAGS)
flags = update_flags_with_config(config_path, testing=True)

flags.modality = 'PA'
flags.img_size = 256
flags.text_encoding = 'word'
flags.feature_extractor_img = 'resnet'
flags.batch_size = 1
flags.dataloader_workers = 0
flags.device = device
flags.normalization = False
flags.len_sequence = 128
flags.str_experiment = 'something'
flags.alpha_modalities = [flags.div_weight_uniform_content, flags.div_weight_m1_content,
                          flags.div_weight_m2_content, flags.div_weight_m3_content]
flags.use_clf = False
flags.dir_gen_eval_fid = 'fdgb'

exp = MimicExperiment(flags)
exp.plot_img_size = torch.Size([1, flags.img_size, flags.img_size])
exp.modalities['text'].plot_img_size = torch.Size([1, flags.img_size+128, flags.img_size])
mods = exp.modalities

trainloader = DataLoader(exp.dataset_train, batch_size=flags.batch_size,
                         shuffle=False,
                         num_workers=flags.dataloader_workers, pin_memory=False)

nbr_samples = 5
datas = {'PA': [], 'Lateral': [], 'text': []}
texts = []
labels_list = []
for data, labels in trainloader:
    if labels[0][-1].item() == 1:
        for mod_key, mod in mods.items():
            datas[mod_key].append(mod.plot_data(exp, data[mod_key].squeeze(0)))
            if mod_key == 'text':
                texts.append(data[mod_key])
            labels_list.append(labels[0].tolist())
        if len(datas[mod_key]) == nbr_samples:
            break

rec = torch.Tensor()

for mod in mods:
    for idx in range(nbr_samples):
        if mod == 'text':
            img = datas[f'{mod}'][idx].cpu().unsqueeze(0)
        else:

            img = datas[f'{mod}'][idx].cpu()
            # pad the non text modalities such that they fit in a wider rectangle.
            m = nn.ZeroPad2d((64, 64, 0, 0))
            img = m(img.squeeze()).unsqueeze(0).unsqueeze(0)
        rec = torch.cat((rec, img), 0)

fig = mimic.utils.plot.create_fig(f'something.png',
                                  img_data=rec,
                                  num_img_row=nbr_samples, save_figure=False)

plt.imshow(fig)
plt.show()
plt.close()

for i in range(nbr_samples):
    text_sample = text.tensor_to_text(exp, texts[i], one_hot=False)[0]
    print(' '.join(text_sample))
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(datas['PA'][i].squeeze())
    plt.subplot(1, 2, 2)
    plt.imshow(datas['Lateral'][i].squeeze())
    plt.show()
    plt.close()
