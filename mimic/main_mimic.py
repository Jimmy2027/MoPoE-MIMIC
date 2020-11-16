import argparse
import gc
import os
from timeit import default_timer as timer

import torch
import torch.multiprocessing as mp
from termcolor import colored

from mimic.run_epochs import run_epochs
from mimic.utils.experiment import MimicExperiment, NaNInLatent
from mimic.utils.filehandling import create_dir_structure, create_dir_structure_testing, get_config_path, \
    get_method
from mimic.utils.flags import parser
from mimic.utils.flags import setup_flags


class Main:
    def __init__(self, flags: argparse.ArgumentParser, config_path: str = None, testing=False):
        """
        config_path: (optional) path to the json config file
        """
        flags = setup_flags(flags, config_path, testing)

        flags = get_method(flags)
        print(colored(f"running on {flags.device} with text {flags.text_encoding} encoding "
                      f'with method {flags.method}, batch size: {flags.batch_size} and img size {flags.img_size}',
                      'blue'))

        self.flags = create_dir_structure(flags)
        # because of bad initialisation, the vae might return nan values. If this is the case it is best to restart the
        # experiment.
        self.max_tries = 10  # maximum restarts of the experiment
        self.current_tries = 0
        self.start_time = 0

    def setup_distributed(self):
        self.flags.world_size = torch.cuda.device_count()
        print(f'setting up distributed computing with world size {self.flags.world_size}')
        self.flags.distributed = self.flags.world_size > 1
        self.flags.batch_size = int(self.flags.batch_size / self.flags.world_size)

    def run_epochs(self) -> bool:
        """
        Wrapper of run_epochs.run_epochs that checks if the workflow was completed and starts it over otherwise
        returns bool: true if run_epochs finishes, False if an error occurs
        """
        self.start_time = timer()
        mimic = MimicExperiment(self.flags)
        create_dir_structure_testing(mimic)
        mimic.set_optimizer()
        mimic.number_restarts = self.current_tries
        try:
            if self.flags.distributed:
                self.setup_distributed()
                mp.spawn(run_epochs, nprocs=self.flags.world_size, args=(mimic,), join=True)
            else:
                run_epochs(self.flags.device, mimic)
        except NaNInLatent as e:
            print(e)
            return False
        mimic.update_experiments_dataframe({'experiment_duration': (timer() - self.start_time) // 60})
        return True

    def restart(self) -> None:
        """
        Clears old dir_structure and creates new one
        """
        self.current_tries += 1
        print(f'********  RESTARTING EXPERIMENT FOR THE {self.current_tries} TIME  ********')
        torch.cuda.empty_cache()
        gc.collect()
        command = f'rm -r {self.flags.dir_experiment_run}'
        print(command)
        os.system(command)
        self.flags = create_dir_structure(self.flags)


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    CONFIG_PATH = get_config_path()
    main = Main(FLAGS, CONFIG_PATH)
    success = False
    while not success and main.current_tries < main.max_tries:
        success = main.run_epochs()
        if not success:
            main.restart()
