import argparse
import os
import shutil
from datetime import datetime
from shutil import copyfile

import mimic


def create_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name, ignore_errors=True)
    os.makedirs(dir_name)


def get_str_experiments(flags, prefix: str = ''):
    dateTimeObj = datetime.now()
    dateStr = dateTimeObj.strftime("%Y_%m_%d_%H_%M_%S_%f")
    if prefix:
        return prefix + '_' + dateStr
    else:
        return flags.dataset + '_' + dateStr


def create_dir_structure_testing(exp):
    flags = exp.flags
    for k, label_str in enumerate(exp.labels):
        dir_gen_eval_label = os.path.join(flags.dir_gen_eval, label_str)
        create_dir(dir_gen_eval_label)
        dir_inference_label = os.path.join(flags.dir_inference, label_str)
        create_dir(dir_inference_label)


def create_dir_structure(flags: argparse.ArgumentParser(), train: bool = True) -> argparse.ArgumentParser:
    if train:
        str_experiments = get_str_experiments(flags)
        flags.dir_experiment_run = os.path.join(os.path.expanduser(flags.dir_experiment), str_experiments)
        flags.str_experiment = str_experiments
    else:
        flags.dir_experiment_run = os.path.expanduser(flags.dir_experiment)
        flags.experiment_uid = get_str_experiments(flags)
    print('dir_experiment_run: ', flags.dir_experiment_run)
    if train:
        create_dir(flags.dir_experiment_run)

    flags.dir_checkpoints = os.path.join(os.path.expanduser(flags.dir_experiment_run), 'checkpoints')
    if train:
        create_dir(os.path.expanduser(flags.dir_checkpoints))
        copyfile(get_config_path(flags), os.path.join(flags.dir_experiment_run, 'config.json'), follow_symlinks=True)

    flags.dir_logs = os.path.join(os.path.expanduser(flags.dir_experiment_run), 'logs')
    if train:
        create_dir(flags.dir_logs)
    print('dir_logs: ', flags.dir_logs)

    flags.dir_gen_eval = os.path.join(os.path.expanduser(flags.dir_experiment_run), 'generation_evaluation')
    if train:
        create_dir(flags.dir_gen_eval)

    flags.dir_inference = os.path.join(os.path.expanduser(flags.dir_experiment_run), 'inference')
    if train:
        create_dir(flags.dir_inference)

    if flags.dir_fid is None:
        flags.dir_fid = flags.dir_experiment_run;
    elif not train:
        flags.dir_fid = os.path.join(flags.dir_experiment_run, 'fid_eval');
        if not os.path.exists(flags.dir_fid):
            os.makedirs(flags.dir_fid);
    flags.dir_gen_eval_fid = os.path.join(flags.dir_fid, 'fid');
    create_dir(flags.dir_gen_eval_fid)

    flags.dir_plots = os.path.join(flags.dir_experiment_run, 'plots')
    if train:
        create_dir(flags.dir_plots)
    flags.dir_swapping = os.path.join(flags.dir_plots, 'swapping')
    if train:
        create_dir(flags.dir_swapping)

    flags.dir_random_samples = os.path.join(flags.dir_plots, 'random_samples')
    if train:
        create_dir(flags.dir_random_samples)

    flags.dir_cond_gen = os.path.join(flags.dir_plots, 'cond_gen')
    if train:
        create_dir(flags.dir_cond_gen)

    if not os.path.exists(flags.dir_clf):
        os.makedirs(flags.dir_clf)
    return flags


def expand_paths(flags: argparse.ArgumentParser()) -> argparse.ArgumentParser():
    flags.dir_data = os.path.expanduser(flags.dir_data)
    flags.dir_experiment = os.path.expanduser(flags.dir_experiment)
    flags.inception_state_dict = os.path.expanduser(flags.inception_state_dict)
    flags.dir_fid = os.path.expanduser(flags.dir_fid)
    flags.dir_clf = os.path.expanduser(flags.dir_clf)
    return flags


def get_method(flags: argparse.ArgumentParser()) -> argparse.ArgumentParser():
    if flags.method == 'poe':
        flags.modality_poe = True
        flags.poe_unimodal_elbos = True
    elif flags.method == 'moe':
        flags.modality_moe = True
    elif flags.method == 'jsd':
        flags.modality_jsd = True
    elif flags.method == 'joint_elbo':
        flags.joint_elbo = True
    else:
        NotImplementedError('method not implemented...exit!')
    return flags


def get_config_path(flags=None):
    if not flags or not flags.config_path:
        if os.path.exists('/cluster/home/klugh/'):
            return os.path.join(os.path.dirname(mimic.__file__), "configs/leomed_mimic_config.json")
        elif os.path.exists('/mnt/data/hendrik'):
            return os.path.join(os.path.dirname(mimic.__file__), "configs/bartholin_mimic_config.json")
        else:
            return os.path.join(os.path.dirname(mimic.__file__), "configs/local_mimic_config.json")
    else:
        return flags.config_path


def set_paths(flags):
    """
    expands user in paths, such as data_dir and dir_clf
    """
    if os.path.exists('/cluster/home/klugh/'):
        flags.dir_data = os.path.expanduser('~/scratch')
        flags.dir_clf = os.path.expanduser('~/scratch/mimic/trained_classifiers/Mimic128')

    elif os.path.exists('/mnt/data/hendrik'):
        flags.dir_data = os.path.expanduser('/mnt/data/hendrik/mimic_scratch')
        flags.dir_clf = os.path.expanduser('/mnt/data/hendrik/mimic_scratch/mimic/trained_classifiers/Mimic128')

    else:
        flags.dir_data = os.path.expanduser('~/Documents/master3/leomed_scratch')
        flags.dir_clf = os.path.expanduser('~/Documents/master3/leomed_scratch/mimic/trained_classifiers/Mimic128')
    return flags
