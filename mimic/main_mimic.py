import gc
import os
from argparse import Namespace
from timeit import default_timer as timer
from typing import Union

import numpy as np
import torch
import torch.multiprocessing as mp
from termcolor import colored

from mimic.run_epochs import run_epochs
from mimic.utils.exceptions import NaNInLatent, CudaOutOfMemory
from mimic.utils.experiment import MimicExperiment
from mimic.utils.filehandling import create_dir_structure, create_dir_structure_testing, get_config_path, \
    get_method
from mimic.utils.flags import parser
from mimic.utils.flags import setup_flags
from mimic.utils.utils import get_gpu_memory
import pandas as pd


class Main:
    def __init__(self, flags: Namespace, testing=False):
        """
        config_path: (optional) path to the json config file
        """
        flags = setup_flags(flags, testing)
        flags = get_method(flags)
        print(colored(f"running on {flags.device} with text {flags.text_encoding} encoding "
                      f'with method {flags.method}, batch size: {flags.batch_size} and img size {flags.img_size}',
                      'blue'))

        self.flags = create_dir_structure(flags)
        # because of bad initialisation, the vae might return nan values. If this is the case it is best to restart the
        # experiment.
        self.max_tries = 10  # maximum restarts of the experiment due to nan values
        self.current_tries = 0
        self.start_time = 0

    def setup_distributed(self):
        self.flags.world_size = torch.cuda.device_count()
        print(f'setting up distributed computing with world size {self.flags.world_size}')
        self.flags.distributed = self.flags.world_size > 1
        self.flags.batch_size = int(self.flags.batch_size / self.flags.world_size)

    def run_epochs(self) -> Union[bool, str]:
        """
        Wrapper of mimic.run_epochs.run_epochs that checks if the workflow was completed and starts it over otherwise.

        returns
            bool: true if run_epochs finishes, False if an error occurs
            string: "cuda_out_of_memory" if GPU runs out of memory
        """
        print(colored(f'current free GPU memory: {get_gpu_memory()}', 'red'))
        self.start_time = timer()
        # need to reinitialize MimicExperiment after each retry
        mimic = MimicExperiment(self.flags)
        create_dir_structure_testing(mimic)
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
        except CudaOutOfMemory as e:
            print(e)
            return 'cuda_out_of_memory'
        mimic.update_experiments_dataframe({'experiment_duration': (timer() - self.start_time) // 60})
        return True

    def restart(self) -> None:
        """
        Clears old dir_structure and creates new one, deletes corresponding row in the experiment dataframe.
        """
        exp_df = pd.read_csv('experiments_dataframe.csv')
        exp_df.drop(exp_df.index[exp_df['str_experiment'] == self.flags.str_experiment])
        exp_df.to_csv('experiments_dataframe.csv', index=False)

        torch.cuda.empty_cache()
        gc.collect()
        command = f'rm -r {self.flags.dir_experiment_run}'
        print(command)
        os.system(command)
        self.flags = create_dir_structure(self.flags)

    def main(self):
        """
        Runs "run_epochs" until it returns True. If "run_epochs" fails because of full GPU memory,
         the batch size is reduced and the workflow is started again.
         If during the training, the model returns NaNs, bad initialization is
         assumed and the workflow is started again.
        """
        success = False
        while not success and self.current_tries < self.max_tries:

            success = self.run_epochs()

            if not success:
                self.current_tries += 1
                print(f'********  RESTARTING EXPERIMENT FOR THE {self.current_tries} TIME  ********')

            if success == 'cuda_out_of_memory':
                old_bs = self.flags.batch_size
                self.flags.batch_size = int(np.floor(self.flags.batch_size * 0.8))
                print(f'********  GPU ran out of memory with batch size {old_bs}, '
                      f'trying again with batch size: {self.flags.batch_size}  ********')
                success = False

            if not success:
                self.restart()


if __name__ == '__main__':
    FLAGS: Namespace = parser.parse_args()
    FLAGS.config_path = get_config_path(FLAGS)
    main = Main(FLAGS)
    main.main()
