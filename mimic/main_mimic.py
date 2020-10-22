import argparse
import gc
import json
import os

import torch

from mimic.utils.experiment import MimicExperiment
from mimic.utils.flags import parser
from mimic.run_epochs import run_epochs
from mimic.utils.filehandling import create_dir_structure, expand_paths, create_dir_structure_testing, get_config_path


class Main:
    def __init__(self):
        FLAGS = parser.parse_args()
        config_path = get_config_path()
        with open(config_path, 'rt') as json_file:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(json_file))
            FLAGS = parser.parse_args(namespace=t_args)
        FLAGS = expand_paths(FLAGS)
        use_cuda = torch.cuda.is_available()
        FLAGS.device = torch.device('cuda' if use_cuda else 'cpu')
        device = 'gpu' if use_cuda else 'cpu'

        print('running on {}'.format(device))

        if FLAGS.method == 'poe':
            FLAGS.modality_poe = True
            FLAGS.poe_unimodal_elbos = True
        elif FLAGS.method == 'moe':
            FLAGS.modality_moe = True
        elif FLAGS.method == 'jsd':
            FLAGS.modality_jsd = True
        elif FLAGS.method == 'joint_elbo':
            FLAGS.joint_elbo = True
        else:
            NotImplementedError('method not implemented...exit!')

        print(FLAGS.modality_poe)
        print(FLAGS.modality_moe)
        print(FLAGS.modality_jsd)
        print(FLAGS.joint_elbo)
        print(FLAGS.dataset)

        FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content, FLAGS.div_weight_m1_content,
                                  FLAGS.div_weight_m2_content, FLAGS.div_weight_m3_content]

        self.FLAGS = create_dir_structure(FLAGS)
        alphabet_path = os.path.join(os.getcwd(), 'alphabet.json')
        with open(alphabet_path) as alphabet_file:
            self.alphabet = str(''.join(json.load(alphabet_file)))

        # because of bad initialisation, the vae might return nan values. If this is the case it is best to restart the
        # experiment.
        self.max_tries = 10  # maximum restarts of the experiment
        self.current_tries = 0

    def run_epochs(self) -> bool:
        """
        Wrapper of run_epochs.run_epochs that checks if the workflow was completed and starts it over otherwise
        """
        mimic = MimicExperiment(self.FLAGS, self.alphabet)
        create_dir_structure_testing(mimic)
        mimic.set_optimizer()
        mimic.number_restarts = self.current_tries
        try:
            run_epochs(mimic)
        except ValueError as e:
            print(e)
            return False
        return True

    def restart(self):
        """
        restarts run_epochs with new initialisation
        """
        self.current_tries += 1
        print(f'********  RESTARTING EXPERIMENT FOR THE {self.current_tries} TIME  ********')
        torch.cuda.empty_cache()
        gc.collect()
        command = f'rm -r {self.FLAGS.dir_experiment_run}'
        print(command)
        os.system(command)
        self.FLAGS = create_dir_structure(self.FLAGS)
        # self.run_epochs()


main = Main()
success = False
while not success and main.current_tries < main.max_tries:
    success = main.run_epochs()
    if not success:
        main.restart()
