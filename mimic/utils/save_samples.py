import os

from torchvision.utils import save_image


def append_list_to_list_linear(l1, l2):  # sourcery skip
    for k in range(0, len(l2)):
        if isinstance(l2[k], str):
            l1.append(l2[k])
        else:
            l1.append(l2[k].item())
    return l1


def write_samples_text_to_file(samples, filename):
    with open(filename, 'w') as file_samples:
        for k in range(len(samples)):
            file_samples.write(''.join(samples[k]) + '\n')


def getText(samples):
    lines = [''.join(samples[k])[::-1] for k in range(len(samples))]
    text = '\n\n'.join(lines)
    print(text)
    return text


def write_samples_img_to_file(samples, filename, img_per_row=1):
    save_image(samples.data.cpu(), filename, nrow=img_per_row)


def save_generated_samples_singlegroup(exp, batch_id, group_name, samples):
    dir_save = exp.paths_fid[group_name]
    for k, key in enumerate(samples.keys()):
        dir_f = os.path.join(dir_save, key)
        if not os.path.exists(dir_f):
            os.makedirs(dir_f)

    cnt_samples = batch_id * exp.flags.batch_size
    for k in range(exp.flags.batch_size):
        for i, key in enumerate(samples.keys()):
            mod = exp.modalities[key]
            fn_out = os.path.join(dir_save, key, str(cnt_samples).zfill(6) +
                                  mod.file_suffix)
            mod.save_data(exp, samples[key][k], fn_out, {'img_per_row': 1})
        cnt_samples += 1
