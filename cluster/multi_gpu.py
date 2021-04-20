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
import scipy.io as spio

DANNCE_PRED_FILE_BASE_NAME = "save_data_AVG"
COM_PRED_FILE_BASE_NAME = "com3d"


def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


class MultiGpuHandler:
    def __init__(
        self,
        config,
        n_samples_per_gpu=5000,
        only_unfinished=False,
        predict_path=None,
        com_file=None,
        # batch_param_file="_batch_params.p",
        verbose=True,
        test=False,
        dannce_file=None,
    ):
        self.config = config
        self.n_samples_per_gpu = n_samples_per_gpu
        self.only_unfinished = only_unfinished
        self.predict_path = predict_path
        self.batch_param_file = "_batch_params.p"
        self.verbose = verbose
        self.com_file = com_file
        self.test = test
        if dannce_file is None:
            self.dannce_file = self.load_dannce_file()
        else:
            self.dannce_file = dannce_file

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

    def load_com_length_from_file(self):
        """Return the length of a com file."""
        _, file_extension = os.path.splitext(self.com_file)

        if file_extension == ".pickle":
            with open(self.com_file, "rb") as file:
                in_dict = pickle.load(file)
            n_com_samples = len(in_dict.keys())
        elif file_extension == ".mat":
            com = loadmat(self.com_file)
            n_com_samples = com["com"][:].shape[0]
        else:
            raise ValueError("com_file must be a .pickle or .mat file")
        return n_com_samples

    def get_n_samples(self, dannce_file, use_com=False):
        """Get the number of samples in a project

        :param dannce_file: Path to dannce.mat file containing sync and com for current project.
        """
        sync = load_sync(dannce_file)
        n_samples = len(sync[0]["data_frame"])
        if n_samples == 1:
            n_samples = len(sync[0]["data_frame"][0])

        if use_com:
            # If a com file is specified, use it
            if self.com_file is not None:
                com_samples = self.load_com_length_from_file()
            else:
                # Try to use the com in the dannce .mat, otherwise error.
                try:
                    com = load_com(dannce_file)
                    com_samples = len(com["sampleID"][0])
                except KeyError:
                    try:
                        params = self.load_params("io.yaml")
                        self.com_file = params["com_file"]
                        com_samples = self.load_com_length_from_file()
                    except:
                        raise KeyError(
                            "dannce.mat file needs com field or com_file needs to be specified in io.yaml."
                        )
            n_samples = np.min([com_samples, n_samples])

        return n_samples

    def generate_batch_params_com(self, n_samples):
        start_samples = np.arange(
            0, n_samples, self.n_samples_per_gpu, dtype=np.int
        )
        max_samples = start_samples + self.n_samples_per_gpu
        batch_params = [
            {"start_sample": sb, "max_num_samples": self.n_samples_per_gpu}
            for sb, mb in zip(start_samples, max_samples)
        ]

        if self.only_unfinished:

            if self.predict_path is None:
                params = self.load_params("io.yaml")
                if params["com_predict_dir"] is None:
                    raise ValueError(
                        "Either predict_path (clarg) or com_predict_dir (in io.yaml) must be specified for merge"
                    )
                else:
                    self.predict_path = params["com_predict_dir"]
            if not os.path.exists(self.predict_path):
                os.makedirs(self.predict_path)
            pred_files = [
                f
                for f in os.listdir(self.predict_path)
                if COM_PRED_FILE_BASE_NAME in f
            ]
            pred_files = [
                f
                for f in pred_files
                if f != (COM_PRED_FILE_BASE_NAME + ".mat")
            ]
            if len(pred_files) > 1:
                params = self.load_params(self.config)
                pred_ids = [
                    int(f.split(".")[0].split("3d")[1]) for f in pred_files
                ]
                for i, batch_param in reversed(list(enumerate(batch_params))):
                    if batch_param["start_sample"] in pred_ids:
                        del batch_params[i]
        return batch_params

    def generate_batch_params_dannce(self, n_samples):
        start_samples = np.arange(
            0, n_samples, self.n_samples_per_gpu, dtype=np.int
        )
        max_samples = start_samples + self.n_samples_per_gpu
        max_samples[-1] = n_samples
        batch_params = [
            {"start_sample": sb, "max_num_samples": mb}
            for sb, mb in zip(start_samples, max_samples)
        ]

        # Delete batch_params that were already finished
        if self.only_unfinished:

            if self.predict_path is None:
                params = self.load_params("io.yaml")
                if params["dannce_predict_dir"] is None:
                    raise ValueError(
                        "Either predict_path (clarg) or dannce_predict_dir (in io.yaml) must be specified for merge"
                    )
                else:
                    self.predict_path = params["dannce_predict_dir"]
            if not os.path.exists(self.predict_path):
                os.makedirs(self.predict_path)
            pred_files = [
                f
                for f in os.listdir(self.predict_path)
                if DANNCE_PRED_FILE_BASE_NAME in f
            ]
            pred_files = [
                f
                for f in pred_files
                if f != (DANNCE_PRED_FILE_BASE_NAME + ".mat")
            ]
            if len(pred_files) > 1:
                params = self.load_params(self.config)
                pred_ids = [
                    int(f.split(".")[0].split("AVG")[1]) * params["batch_size"]
                    for f in pred_files
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
        if not self.test:
            sys.exit(os.WEXITSTATUS(os.system(cmd)))

    def submit_dannce_predict_multi_gpu(self):
        """Predict dannce over multiple gpus in parallel.

        Divide project into equal chunks of n_samples_per_gpu samples. Submit an array job
        that predicts over each chunk in parallel.

        """
        n_samples = self.get_n_samples(self.dannce_file, use_com=True)
        batch_params = self.generate_batch_params_dannce(n_samples)
        cmd = (
            "sbatch --wait --array=0-%d holy_dannce_predict_multi_gpu.sh %s"
            % (
                len(batch_params) - 1,
                self.config,
            )
        )
        if len(batch_params) > 0:
            self.save_batch_params(batch_params)
            self.submit_jobs(batch_params, cmd)
        return batch_params, cmd

    def submit_com_predict_multi_gpu(self):
        """Predict com over multiple gpus in parallel.

        Divide project into equal chunks of n_samples_per_gpu samples. Submit an array job
        that predicts over each chunk in parallel.

        """
        n_samples = self.get_n_samples(self.dannce_file, use_com=False)
        print(n_samples)
        batch_params = self.generate_batch_params_com(n_samples)
        cmd = "sbatch --wait --array=0-%d holy_com_predict_multi_gpu.sh %s" % (
            len(batch_params) - 1,
            self.config,
        )
        if len(batch_params) > 0:
            self.save_batch_params(batch_params)
            self.submit_jobs(batch_params, cmd)
        return batch_params, cmd

    def com_merge(self):
        # Get all of the paths
        if self.predict_path is None:
            # Try to get it from io.yaml
            params = self.load_params("io.yaml")
            if params["com_predict_dir"] is None:
                raise ValueError(
                    "Either predict_path (clarg) or com_predict_dir (in io.yaml) must be specified for merge"
                )
            else:
                self.predict_path = params["com_predict_dir"]
        pred_files = [
            f
            for f in os.listdir(self.predict_path)
            if COM_PRED_FILE_BASE_NAME in f and ".mat" in f
        ]
        pred_files = [
            f
            for f in pred_files
            if f != (COM_PRED_FILE_BASE_NAME + ".mat") and "instance" not in f
        ]
        pred_inds = [
            int(f.split(COM_PRED_FILE_BASE_NAME)[-1].split(".")[0])
            for f in pred_files
        ]
        pred_files = [pred_files[i] for i in np.argsort(pred_inds)]

        if len(pred_files) == 0:
            raise FileNotFoundError("No prediction files were found.")

        # Load all of the data and save to a single file.
        com, sampleID, metadata = [], [], []
        for pred in pred_files:
            M = loadmat(os.path.join(self.predict_path, pred))
            com.append(M["com"])
            sampleID.append(M["sampleID"])
            metadata.append(M["metadata"])

        com = np.concatenate(com, axis=0)
        sampleID = np.concatenate(sampleID, axis=0)
        metadata = metadata[0]

        # Update samples and max_num_samples
        metadata["start_sample"] = 0
        metadata["max_num_samples"] = "max"

        if len(com.shape) == 3:
            for n_instance in range(com.shape[2]):
                fn = os.path.join(
                    self.predict_path,
                    "instance"
                    + str(n_instance)
                    + COM_PRED_FILE_BASE_NAME
                    + ".mat",
                )
                savemat(
                    fn,
                    {
                        "com": com[..., n_instance].squeeze(),
                        "sampleID": sampleID,
                        "metadata": metadata,
                    },
                )
        # save to a single file.
        else:
            fn = os.path.join(
                self.predict_path, COM_PRED_FILE_BASE_NAME + ".mat"
            )
            savemat(
                fn, {"com": com, "sampleID": sampleID, "metadata": metadata}
            )

    def dannce_merge(self):
        # Get all of the paths
        if self.predict_path is None:
            # Try to get it from io.yaml
            params = self.load_params("io.yaml")
            if params["dannce_predict_dir"] is None:
                raise ValueError(
                    "Either predict_path (clarg) or dannce_predict_dir (in io.yaml) must be specified for merge"
                )
            else:
                self.predict_path = params["dannce_predict_dir"]
        pred_files = [
            f
            for f in os.listdir(self.predict_path)
            if DANNCE_PRED_FILE_BASE_NAME in f
        ]
        pred_files = [
            f for f in pred_files if f != (DANNCE_PRED_FILE_BASE_NAME + ".mat")
        ]
        pred_inds = [
            int(f.split(DANNCE_PRED_FILE_BASE_NAME)[-1].split(".")[0])
            for f in pred_files
        ]
        pred_files = [pred_files[i] for i in np.argsort(pred_inds)]
        if len(pred_files) == 0:
            raise FileNotFoundError("No prediction files were found.")

        # Load all of the data
        pred, data, p_max, sampleID, metadata = [], [], [], [], []
        for file in pred_files:
            M = loadmat(os.path.join(self.predict_path, file))
            pred.append(M["pred"])
            data.append(M["data"])
            p_max.append(M["p_max"])
            sampleID.append(M["sampleID"])
            metadata.append(M["metadata"])
        pred = np.concatenate(pred, axis=0)
        data = np.concatenate(data, axis=0)
        p_max = np.concatenate(p_max, axis=0)
        sampleID = np.concatenate(sampleID, axis=0)
        metadata = metadata[0]

        # Update samples and max_num_samples
        metadata["start_sample"] = 0
        metadata["max_num_samples"] = "max"

        # save to a single file.
        fn = os.path.join(
            self.predict_path, DANNCE_PRED_FILE_BASE_NAME + ".mat"
        )
        savemat(
            fn,
            {
                "pred": pred,
                "data": data,
                "p_max": p_max,
                "sampleID": sampleID,
                "metadata": metadata,
            },
        )


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

    params = infer_params(params, dannce_net=dannce_net, prediction=True)
    return params


def dannce_predict_single_batch():
    from dannce.interface import dannce_predict

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
    dannce_predict(params)


def com_predict_single_batch():
    from dannce.interface import com_predict

    # Load in parameters to modify
    config = sys.argv[1]
    handler = MultiGpuHandler(config)
    batch_params = handler.load_batch_params()
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    # task_id = 0
    batch_param = batch_params[task_id]
    print(batch_param)

    # Build final parameter dictionary
    params = build_params_from_config_and_batch(
        config, batch_param, dannce_net=False
    )

    # Predict
    try:
        com_predict(params)
    except OSError:
        # If a job writes to the label3d file at the same time as another reads from it
        # it throws an OSError.
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


def com_merge():
    args = cmdline_args()
    handler = MultiGpuHandler(**args.__dict__)
    handler.com_merge()


def dannce_merge():
    args = cmdline_args()
    handler = MultiGpuHandler(**args.__dict__)
    handler.dannce_merge()


def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("config", help="Path to .yaml configuration file")
    p.add_argument(
        "--n-samples-per-gpu",
        dest="n_samples_per_gpu",
        type=int,
        default=5000,
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
