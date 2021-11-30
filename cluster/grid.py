import numpy as np
import sys
import pickle
import os
import yaml
import argparse
import ast
from scipy.io import savemat
from dannce.engine.io import load_sync, load_com
from dannce.engine.processing import prepare_save_metadata
from dannce import (
    _param_defaults_shared,
    _param_defaults_dannce,
    _param_defaults_com,
)

import subprocess
import time

class GridHandler:
    def __init__(
        self,
        config,
        grid_config,
        verbose=True,
        test=False,
        dannce_file=None,
    ):
        self.config = config
        self.grid_config = grid_config
        self.batch_param_file = "_grid_params.p"
        self.verbose = verbose
        self.test = test

    def load_params(self, param_path):
        """Load a params file"""
        with open(param_path, "rb") as file:
            params = yaml.safe_load(file)
        return params["batch_params"]

    def save_batch_params(self, batch_params):
        """Save the batch_param dictionary to the batch_param file"""
        out_dict = {"batch_params": batch_params}
        with open(self.batch_param_file, "wb") as file:
            pickle.dump(out_dict, file)

    def load_batch_params(self):
        with open(self.batch_param_file, "rb") as file:
            in_dict = pickle.load(file)
        return in_dict["batch_params"]

    def generate_batch_params_dannce(self):
        return self.load_params(self.grid_config)

    def submit_jobs(self, batch_params, cmd):
        """Print out description of command and issue system command"""
        if self.verbose:
            for batch_param in batch_params:
                print(batch_param)
            print("Command issued: ", cmd)
        if not self.test:
            if isinstance(cmd, list): 
                for i in range(len(cmd)):
                    os.environ["SLURM_ARRAY_TASK_ID"] = str(i)
                    os.system(cmd[i])
                    time.sleep(0.05)
            elif isinstance(cmd, str):
                os.system(cmd)
    
    def get_parent_job_id(self):
        """Return the job id in the last row of squeue -u <user> slurm command.
            The assumption here is that the last line of the squeue command would
            be the job_id of the parent sbatch job from which the array of jobs has
            to be called. This job_id will be used while iterating over the number 
            of jobs to set customized output file names.
        """
        
        get_user_cmd = "whoami"
        get_user_process = subprocess.Popen(get_user_cmd.split(), stdout=subprocess.PIPE)
        slurm_uname = get_user_process.communicate()[0].decode("utf-8").rstrip()

        get_queue_cmd = "squeue -u "+slurm_uname
        get_queue_process = subprocess.Popen(get_queue_cmd.split(), stdout=subprocess.PIPE)
        queue = get_queue_process.communicate()[0].decode("utf-8").split('\n')
        current_job_row = queue[-2].strip()
        job_id = current_job_row.split(' ')[0]

        return job_id, slurm_uname


    def submit_dannce_train_grid(self):
        """Submit dannce grid search.

        Submit a training job with parameter modifications
        listed in self.grid_config.
        """
        batch_params = self.generate_batch_params_dannce()

        use_array = True

        # with open("holy_dannce_train_grid.sh",'r') as train_script:
        #     if 'dsplus' in train_script.read():
        #         use_array = False

        if use_array :
            # The below command does not work on dsplus gpu properly.
            cmd = "sbatch --array=0-%d holy_dannce_train_grid.sh %s %s" % (
                len(batch_params) - 1,
                self.config,
                self.grid_config,
            )

        else:
            # This is a workaround to submit an array iteratively
            parent_job_id, slurm_uname = self.get_parent_job_id()
            cmd = [] 
            for i in range(0, len(batch_params) - 1):
                out_name = slurm_uname+"_"+parent_job_id+"_"+str(i)+"_%j.out"
                cmd_i = "sbatch --output=%s holy_dannce_train_grid.sh %s %s" % (
                        out_name,
                        self.config,
                        self.grid_config,
                        )
                cmd.append(cmd_i)

        if len(batch_params) > 0:
            self.save_batch_params(batch_params)
            self.submit_jobs(batch_params, cmd)
        return batch_params, cmd


def build_params_from_config_and_batch(config, batch_param, dannce_net=True):
    from dannce.interface import build_params
    from dannce.engine.processing import infer_params

    # Build final parameter dictionary
    params = build_params(config, dannce_net=dannce_net)
    for key, value in batch_param.items():
        params[key] = value
    if dannce_net:
        for key, value in _param_defaults_dannce.items():
            if key not in params:
                params[key] = value
    else:
        for key, value in _param_defaults_com.items():
            if key not in params:
                params[key] = value
    for key, value in _param_defaults_shared.items():
        if key not in params:
            params[key] = value

    params = infer_params(params, dannce_net=dannce_net, prediction=False)
    return params


def dannce_train_single_batch():
    from dannce.interface import dannce_train

    # Load in parameters to modify
    config = sys.argv[1]
    grid_config = sys.argv[2]
    handler = GridHandler(config, grid_config)
    batch_params = handler.load_batch_params()
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    print("Task ID = ", task_id)
    batch_param = batch_params[task_id]
    print(batch_param)

    # Build final parameter dictionary
    params = build_params_from_config_and_batch(config, batch_param)

    # Train
    dannce_train(params)


def dannce_train_grid():
    # Load in parameters to modify
    args = cmdline_args()
    handler = GridHandler(**args.__dict__)
    handler.submit_dannce_train_grid()


def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("config", help="Path to .yaml configuration file")
    p.add_argument(
        "grid_config", help="Path to .yaml grid search configuration file"
    )
    p.add_argument(
        "--verbose",
        dest="verbose",
        type=ast.literal_eval,
        default=True,
        help="If True, print out submission command and info.",
    )
    p.add_argument(
        "--test",
        dest="test",
        type=ast.literal_eval,
        default=False,
        help="If True, print out submission command and info, but do not submit jobs.",
    )
    return p.parse_args()
