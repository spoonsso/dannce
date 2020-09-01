import numpy as np
import sys
import pickle
import os
import yaml
import argparse
import ast
from scipy.io import loadmat
from dannce.engine.io import load_sync, load_com
from dannce import _param_defaults_shared, _param_defaults_dannce, _param_defaults_com


class MultiGpuHandler:
    def __init__(
        self,
        config,
        n_samples_per_gpu=10000,
        only_unfinished=False,
        predict_path=None,
        batch_param_file="_batch_params.p",
        verbose=True,
    ):
        self.config = config
        self.n_samples_per_gpu = n_samples_per_gpu
        self.only_unfinished = only_unfinished
        self.predict_path = predict_path
        self.batch_param_file = batch_param_file
        self.verbose = verbose

    def load_params(self, param_path):
        """Load a params file"""
        with open(param_path, "rb") as file:
            params = yaml.safe_load(file)
        return params

    def save_batch_params(self, batch_params):
        """Save the batch_param dictionary to the batch_param file"""
        out_dict = {"batch_params": batch_params}
        with open(self.batch_param_file, "wb") as file:
            pickle.dump(out_dict, file)

    def load_batch_params(self):
        with open(self.batch_param_file, "rb") as file:
            in_dict = pickle.load(file)
        return in_dict["batch_params"]

    def load_dannce_file(self, path="."):
        """Return the path to the first dannce.mat file in a project folder."""
        files = os.listdir(path)
        dannce_file = [f for f in files if "dannce.mat" in f]
        if len(dannce_file) == 0:
            raise FileNotFoundError("No dannce.mat file found.")
        return dannce_file[0]

    def get_n_samples(self, dannce_file, use_com=False):
        """Get the number of samples in a project
        
        :param dannce_file: Path to dannce.mat file containing sync and com for current project. 
        """
        sync = load_sync(dannce_file)
        n_samples = len(sync[0]["data_frame"])

        if use_com:
            # Try to use the com in the dannce .mat, otherwise fall back to default.
            try:
                com = load_com(dannce_file)
                com_samples = len(com["sampleID"])
            except KeyError:
                raise KeyError("dannce.mat file needs com field.")
            n_samples = np.min([com_samples, n_samples])

        return n_samples

    def generate_batch_params(
        self, n_samples, only_unfinished=False, predict_path=None
    ):
        start_samples = np.arange(0, n_samples, self.n_samples_per_gpu, dtype=np.int)
        max_samples = start_samples + self.n_samples_per_gpu
        max_samples[-1] = n_samples
        batch_params = [
            {"start_sample": sb, "max_num_samples": mb}
            for sb, mb in zip(start_samples, max_samples)
        ]

        # Delete batch_params that were already finished
        if self.only_unfinished:
            if self.predict_path is None:
                ValueError("Predict_path must be specified if only_unfinished is true")
            if not os.path.exists(self.predict_path):
                os.makedirs(self.predict_path)
            pred_files = [
                f for f in os.listdir(self.predict_path) if "save_data_AVG" in f
            ]
            pred_files = [f for f in pred_files if f != "save_data_AVG.mat"]
            if len(pred_files) > 1:
                # TODO(Automate_batch_size)
                pred_ids = [
                    int(f.split(".")[0].split("AVG")[1]) * 4 for f in pred_files
                ]
                for i, batch_param in reversed(list(enumerate(batch_params))):
                    if batch_param["start_sample"] in pred_ids:
                        del batch_params[i]
        return batch_params

    def submit_jobs(self, batch_params, cmd):
        """Print out description of command and issue system command"""
        if self.verbose:
            for batch_param in batch_params:
                print("Start sample:", batch_param["start_sample"])
                print("End sample:", batch_param["max_num_samples"])
            print("Command issued: ", cmd)
        os.system(cmd)

    def submit_dannce_predict_multi_gpu(self):
        """Predict dannce over multiple gpus in parallel.
        
        Divide project into equal chunks of n_samples_per_gpu samples. Submit an array job 
        that predicts over each chunk in parallel. 

        """
        dannce_file = self.load_dannce_file()
        n_samples = self.get_n_samples(dannce_file, use_com=True)
        batch_params = self.generate_batch_params(n_samples)
        cmd = "sbatch --array=0-%d holy_dannce_predict_multi_gpu.sh %s" % (
            len(batch_params) - 1,
            self.config,
        )
        if len(batch_params) > 0:
            self.save_batch_params(batch_params)
            self.submit_jobs(batch_params, cmd)

    def submit_com_predict_multi_gpu(self):
        """Predict com over multiple gpus in parallel.
        
        Divide project into equal chunks of n_samples_per_gpu samples. Submit an array job 
        that predicts over each chunk in parallel. 

        """
        dannce_file = self.load_dannce_file()
        n_samples = self.get_n_samples(dannce_file, use_com=False)
        batch_params = self.generate_batch_params(n_samples)
        cmd = "sbatch --array=0-%d holy_com_predict_multi_gpu.sh %s" % (
            len(batch_params) - 1,
            self.config,
        )
        if len(batch_params) > 0:
            self.save_batch_params(batch_params)
            self.submit_jobs(batch_params, cmd)



def build_params_from_config_and_batch(config, batch_param):
    # Build final parameter dictionary
    params = build_params(config, dannce_net=True)
    for k, v in batch_param.items():
        params[k] = v
    for k, v in _param_defaults_dannce.items():
        if k not in params:
            params[k] = v
    for k, v in _param_defaults_shared.items():
        if k not in params:
            params[k] = v
    params = infer_params(params, dannce_net=True, prediction=True)
    return params

def dannce_predict_single_batch():
    # Load in parameters to modify
    config = sys.argv[1]
    handler = MultiGpuHandler(config)
    batch_params = handler.load_batch_params()
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    batch_param = batch_params[task_id]
    print(batch_param)

    # Build final parameter dictionary
    params = build_params_from_config_and_batch(config, batch_param)

    # Predict
    from dannce.interface import dannce_predict
    dannce_predict(params)

def com_predict_single_batch():
    # Load in parameters to modify
    config = sys.argv[1]
    handler = MultiGpuHandler(config)
    batch_params = handler.load_batch_params()
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    batch_param = batch_params[task_id]
    print(batch_param)

    # Build final parameter dictionary
    params = build_params_from_config_and_batch(config, batch_param)

    # Predict
    com_predict(params)


def dannce_predict_multi_gpu():
    # Load in parameters to modify
    args = cmdline_args()
    handler = MultiGpuHandler(**args.__dict__)
    handler.submit_dannce_predict_multi_gpu()


def com_predict_multi_gpu():
    # Load in parameters to modify
    args = cmdline_args()
    handler = MultiGpuHandler(**args.__dict__)
    handler.submit_com_predict_multi_gpu()


def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    p.add_argument("config", help="Path to .yaml configuration file")
    p.add_argument(
        "--n-samples-per-gpu",
        dest="n_samples_per_gpu",
        type=int,
        default=10000,
        help="Number of samples for each GPU job to handle.",
    )
    p.add_argument(
        "--only-unfinished",
        dest="only_unfinished",
        type=ast.literal_eval,
        default=False,
        help="If true, only predict chunks that have not been previously predicted.",
    )
    p.add_argument(
        "--predict-path",
        dest="predict_path",
        default=None,
        help="When using only_unfinished, check predict_path for previously predicted chunks.",
    )
    p.add_argument(
        "--com-file",
        dest="com_file",
        default=None,
        help="Use com-file to check the number of samples over which to predict rather than a dannce.mat file",
    )
    # p.add_argument(
    #     "--batch-param-file",
    #     dest="batch_param_file",
    #     default="_batch_params.p",
    #     help="Name of file in which to store submission params.",
    # )
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
    p.add_argument(
        "--dannce-file",
        dest="dannce_file",
        default=None,
        help="Path to dannce.mat file to use for determining n total samples.",
    )

    return p.parse_args()
