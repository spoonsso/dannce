import sys
import pickle
import os
import yaml
import argparse
import ast
from typing import Text, List, Tuple
from cluster.multi_gpu import build_params_from_config_and_batch


import subprocess
import time

class GridHandler:
    def __init__(
        self,
        config: Text,
        grid_config: Text,
        verbose: bool = True,
        test: bool = False,
        dannce_file: Text = None,
    ):
        """Initialize grid search handler

        Args:
            config (Text): Path to base config .yaml file.
            grid_config (Text): Path to grid search config .yaml file.
            verbose (bool, optional): If True, print out batch parameters. Defaults to True.
            test (bool, optional): If True, print out system command, but do not run. Defaults to False.
            dannce_file (Text, optional): Path to dannce.mat file. Defaults to None.
        """
        self.config = config
        self.grid_config = grid_config
        self.batch_param_file = "_grid_params.p"
        self.verbose = verbose
        self.test = test

    def load_params(self, param_path: Text) -> List:
        """Load the training parameters

        Args:
            param_path (Text): Path to parameters file.

        Returns:
            List: Training parameters for each batch
        """
        with open(param_path, "rb") as file:
            params = yaml.safe_load(file)
        return params["batch_params"]

    def save_batch_params(self, batch_params: List):
        """Save the batch_param dictionary to the batch_param file

        Args:
            batch_params (List): List of batch training parameters
        """
        out_dict = {"batch_params": batch_params}
        with open(self.batch_param_file, "wb") as file:
            pickle.dump(out_dict, file)

    def load_batch_params(self) -> List:
        """Load the batch parameters

        Returns:
            List: Batch training parameters
        """
        with open(self.batch_param_file, "rb") as file:
            in_dict = pickle.load(file)
        return in_dict["batch_params"]

    def generate_batch_params_dannce(self) -> List:
        """Generate the batch parameters

        Returns:
            List: Training parameters for each batch
        """
        return self.load_params(self.grid_config)

    def submit_jobs(self, batch_params: List, cmd: Text):
        """Print out description of command and issue system command

        Args:
            batch_params (List): List of batch training parameters.
            cmd (Text): System command to be issued.
        """
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

        get_queue_cmd = "squeue -u " + slurm_uname
        get_queue_process = subprocess.Popen(get_queue_cmd.split(), stdout=subprocess.PIPE)
        queue = get_queue_process.communicate()[0].decode("utf-8").split('\n')
        current_job_row = queue[-2].strip()
        job_id = current_job_row.split(' ')[0]

        return job_id, slurm_uname


    def submit_dannce_train_grid(self) -> Tuple[List, Text]:
        """Submit dannce grid search.

        Submit a training job with parameter modifications
        listed in grid_config.

        Returns:
            Tuple[List, Text]: Batch parameters list, system command
        """
        batch_params = self.generate_batch_params_dannce()

        slurm_config = self.load_params(self.load_params(self.config)["slurm_config"])
        cmd = (
            'sbatch --wait --array=0-%d %s --wrap="%s dannce-train-single-batch %s %s"'
            % (
                len(batch_params) - 1,
                slurm_config["dannce_train_grid"],
                slurm_config["setup"],
                self.config,
                self.grid_config,
            )
        )
        if len(batch_params) > 0:
            self.save_batch_params(batch_params)
            self.submit_jobs(batch_params, cmd)
        return batch_params, cmd


def dannce_train_single_batch():
    """CLI entrypoint to train a single batch."""
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
    """CLI entrypoint to submit a set of training parameters."""
    # Load in parameters to modify
    args = cmdline_args()
    handler = GridHandler(**args.__dict__)
    handler.submit_dannce_train_grid()


def cmdline_args():
    """Parse command line arguments

    Returns:
        [type]: Argparser values
    """
    # Make parser object
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("config", help="Path to .yaml configuration file")
    p.add_argument("grid_config", help="Path to .yaml grid search configuration file")
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
