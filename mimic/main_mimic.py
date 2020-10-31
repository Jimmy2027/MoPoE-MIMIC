import argparse
import gc
import json
import os
from timeit import default_timer as timer

import torch
from termcolor import colored

from mimic.run_epochs import run_epochs
from mimic.utils.experiment import MimicExperiment
from mimic.utils.filehandling import create_dir_structure, expand_paths, create_dir_structure_testing, get_config_path, \
    get_method
from mimic.utils.flags import parser


class Main:
    def __init__(self, flags: argparse.ArgumentParser, config_path: str = None):
        """
        config_path: (optional) path to the json config file
        """
        with open(config_path, 'rt') as json_file:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(json_file))
            flags = parser.parse_args(namespace=t_args)
        flags = expand_paths(flags)
        use_cuda = torch.cuda.is_available()
        flags.device = torch.device('cuda' if use_cuda else 'cpu')
        device = 'gpu' if use_cuda else 'cpu'

        flags = get_method(flags)
        print(colored(f'running on {device} with text {flags.text_encoding} encoding '
              f'with method {flags.method}, batch size: {flags.batch_size} and img size {flags.img_size}', 'red'))
        print(flags.dataset)

        flags.alpha_modalities = [flags.div_weight_uniform_content, flags.div_weight_m1_content,
                                  flags.div_weight_m2_content, flags.div_weight_m3_content]

        self.flags = create_dir_structure(flags)
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
        self.start_time = timer()
        mimic = MimicExperiment(self.flags, self.alphabet)
        create_dir_structure_testing(mimic)
        mimic.set_optimizer()
        mimic.number_restarts = self.current_tries
        try:
            run_epochs(mimic)
        except ValueError as e:
            print(e)
            return False
        mimic.update_experiments_dataframe({'experiment_duration': (timer() - self.start_time) // 60})
        return True

    def restart(self):
        """
        restarts run_epochs with new initialisation
        """
        self.current_tries += 1
        print(f'********  RESTARTING EXPERIMENT FOR THE {self.current_tries} TIME  ********')
        torch.cuda.empty_cache()
        gc.collect()
        command = f'rm -r {self.flags.dir_experiment_run}'
        print(command)
        os.system(command)
        self.flags = create_dir_structure(self.flags)


FLAGS = parser.parse_args()
CONFIG_PATH = get_config_path()
main = Main(FLAGS, CONFIG_PATH)
success = False
while not success and main.current_tries < main.max_tries:
    success = main.run_epochs()
    if not success:
        main.restart()
