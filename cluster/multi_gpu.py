import numpy as np
import sys
import pickle
import os
import yaml
import argparse
import ast
from scipy.io import savemat
from dannce.engine.io import load_sync, load_com
from dannce import (
    _param_defaults_shared,
    _param_defaults_dannce,
    _param_defaults_com,
)
import scipy.io as spio
from typing import Dict, List, Text

DANNCE_BASE_NAME = "save_data_AVG"
COM_BASE_NAME = "com3d"


def loadmat(filename: Text) -> Dict:
    """Wrapper to spio loadmat.

    This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    Returns:
        Dict: Matlab file contents
    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict: Dict) -> Dict:
    """Checks if entries in dictionary are mat-objects.

    If yes, todict is called to change them to nested dictionaries

    Returns:
        Dict: Matlab file contents
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


def load_params(param_path: Text) -> Dict:
    """Load a params file

    Args:
        param_path (Text): Path to parameters file

    Returns:
        Dict: Parameters dictionary
    """
    with open(param_path, "rb") as file:
        params = yaml.safe_load(file)
    return params


class MultiGpuHandler:
    def __init__(
        self,
        config: Text,
        n_samples_per_gpu: int = 5000,
        only_unfinished: bool = False,
        predict_path: Text = None,
        com_file: Text = None,
        # batch_param_file="_batch_params.p",
        verbose: bool = True,
        test: bool = False,
        dannce_file: Text = None,
    ):
        """Initialize multi-gpu handler

        Args:
            config (Text): Path to base configuration .yaml file
            n_samples_per_gpu (int, optional): Number of samples to evaluate for each job. Defaults to 5000.
            only_unfinished (bool, optional): If True, only evaluate unfinished jobs. Defaults to False.
            predict_path (Text, optional): Path to prediction folder. Defaults to None.
            com_file (Text, optional): Path to com file. Defaults to None.
            verbose (bool, optional): If True, print out job details. Defaults to True.
            test (bool, optional): If True, only print system commands, but do not run them. Defaults to False.
            dannce_file (Text, optional): Path to *dannce.mat file. Defaults to None.
        """
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

    def save_batch_params(self, batch_params: List):
        """Save the batch_param dictionary to the batch_param file.

        Args:
            batch_params (List): List of batch parameters.
        """
        out_dict = {"batch_params": batch_params}
        with open(self.batch_param_file, "wb") as file:
            pickle.dump(out_dict, file)

    def load_batch_params(self) -> List:
        """Load the batch parameters

        Returns:
            List: batch parameters
        """
        with open(self.batch_param_file, "rb") as file:
            in_dict = pickle.load(file)
        return in_dict["batch_params"]

    def load_dannce_file(self, path: Text = ".") -> Text:
        """Return the path to the first dannce.mat file in a project folder.

        Args:
            path (Text, optional): Path to folder in which to search for dannce file. Defaults to ".".

        Raises:
            FileNotFoundError: If no dannce.mat file is found.

        Returns:
            Text: Name of dannce.mat file
        """
        files = os.listdir(path)
        dannce_file = [f for f in files if "dannce.mat" in f]
        if len(dannce_file) == 0:
            raise FileNotFoundError("No dannce.mat file found.")
        return dannce_file[0]

    def load_com_length_from_file(self) -> int:
        """Return the length of a com file.

        Raises:
            ValueError: If file extension is not pickle or mat

        Returns:
            int: Number of samples.
        """
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

    def get_n_samples(self, dannce_file: Text, use_com=False) -> int:
        """Get the number of samples in a project

        Args:
            dannce_file (Text): Path to dannce.mat file containing sync and com for current project.
            use_com (bool, optional): If True, get n_samples from the com file. Defaults to False.

        Returns:
            int: Number of samples
        """
        n_samples = self._n_samples_from_sync(dannce_file)

        if use_com:
            com_samples = self._n_samples_from_com(dannce_file)
            n_samples = np.min([com_samples, n_samples])
        return n_samples

    def _n_samples_from_com(self, dannce_file: Text) -> int:
        """Get the number of samples from com estimates

        Args:
            dannce_file (Text): Path to dannce file

        Raises:
            KeyError: dannce.mat file needs com field or com_file needs to be specified in io.yaml.

        Returns:
            int: Number of com samples
        """
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
                    params = load_params("io.yaml")
                    self.com_file = params["com_file"]
                    com_samples = self.load_com_length_from_file()
                except:
                    raise KeyError(
                        "dannce.mat file needs com field or com_file needs to be specified in io.yaml."
                    )
        return com_samples

    def _n_samples_from_sync(self, dannce_file: Text) -> int:
        """Get the number of samples from the sync field of a dannce.mat file.

        Args:
            dannce_file (Text): Path to dannce.mat file

        Returns:
            int: Number of samples
        """
        sync = load_sync(dannce_file)
        n_samples = len(sync[0]["data_frame"])
        if n_samples == 1:
            n_samples = len(sync[0]["data_frame"][0])
        return n_samples

    def generate_batch_params_com(self, n_samples: int) -> List:
        """Generate batch parameters list for com inference

        Args:
            n_samples (int): n_samples in the recording.

        Raises:
            ValueError: If predict_path or com_predict_dir are not specified.

        Returns:
            List: Batch parameters list of dictionaries.
        """
        start_samples = np.arange(0, n_samples, self.n_samples_per_gpu, dtype=np.int)
        max_samples = start_samples + self.n_samples_per_gpu
        batch_params = [
            {"start_sample": sb, "max_num_samples": self.n_samples_per_gpu}
            for sb, mb in zip(start_samples, max_samples)
        ]

        if self.only_unfinished:
            if self.predict_path is None:
                params = load_params("io.yaml")
                if params["com_predict_dir"] is None:
                    raise ValueError(
                        "Either predict_path (clarg) or com_predict_dir (in io.yaml) must be specified for merge"
                    )
                else:
                    self.predict_path = params["com_predict_dir"]
            if not os.path.exists(self.predict_path):
                os.makedirs(self.predict_path)
            pred_files = [
                f for f in os.listdir(self.predict_path) if COM_BASE_NAME in f
            ]
            pred_files = [
                f
                for f in pred_files
                if not (
                    f.endswith(COM_BASE_NAME + ".mat")
                    or f.endswith(COM_BASE_NAME + ".pickle")
                )
            ]
            if len(pred_files) > 1:
                params = load_params(self.config)
                pred_ids = [int(f.split(".")[0].split("3d")[1]) for f in pred_files]
                for i, batch_param in reversed(list(enumerate(batch_params))):
                    if batch_param["start_sample"] in pred_ids:
                        del batch_params[i]
        return batch_params

    def generate_batch_params_dannce(self, n_samples: int) -> List:
        """Generate batch parameters list for dannce inference

        Args:
            n_samples (int): n_samples in the recording.

        Raises:
            ValueError: If predict_path or com_predict_dir are not specified.

        Returns:
            List: Batch parameters list of dictionaries.
        """
        start_samples = np.arange(0, n_samples, self.n_samples_per_gpu, dtype=np.int)
        max_samples = start_samples + self.n_samples_per_gpu
        max_samples[-1] = n_samples

        params = load_params(self.config)
        params = {**params, **load_params("io.yaml")}
        if "n_instances" not in params:
            params["n_instances"] = 1

        # If multi-instance, set the com_file and dannce predict path automatically
        if params["n_instances"] >= 2:
            batch_params = []
            for n_instance in range(params["n_instances"]):
                com_file = os.path.join(
                    params["com_predict_dir"], "instance%dcom3d.mat" % (n_instance)
                )
                dannce_predict_dir = os.path.join(
                    params["dannce_predict_dir"], "instance%d" % (n_instance)
                )
                os.makedirs(dannce_predict_dir, exist_ok=True)
                for sb, mb in zip(start_samples, max_samples):
                    batch_params.append(
                        {
                            "start_sample": sb,
                            "max_num_samples": mb,
                            "com_file": com_file,
                            "dannce_predict_dir": dannce_predict_dir,
                        }
                    )
            # Delete batch_params that were already finished
            if self.only_unfinished:
                batch_params = self.remove_finished_batches_multi_instance(batch_params)
        else:
            batch_params = [
                {"start_sample": sb, "max_num_samples": mb}
                for sb, mb in zip(start_samples, max_samples)
            ]

            # Delete batch_params that were already finished
            if self.only_unfinished:
                batch_params = self.remove_finished_batches(batch_params)
        return batch_params

    def remove_finished_batches_multi_instance(self, batch_params: List) -> List:
        """Remove finished batches from parameters list.

        Args:
            batch_params (List): Batch parameters list.

        Returns:
            (List): Updated batch parameters list.
        """
        dannce_predict_dirs = [param["dannce_predict_dir"] for param in batch_params]
        dannce_predict_dirs = list(set(dannce_predict_dirs))

        # For each instance directory, find the completed batches and delete the params.
        for pred_dir in dannce_predict_dirs:
            # Get all of the files
            pred_files = [f for f in os.listdir(pred_dir) if DANNCE_BASE_NAME in f]

            # Remove any of the default merged files.
            pred_files = [f for f in pred_files if f != (DANNCE_BASE_NAME + ".mat")]
            if len(pred_files) > 1:
                params = load_params(self.config)
                pred_ids = [
                    int(f.split(".")[0].split("AVG")[1]) * params["batch_size"]
                    for f in pred_files
                ]
                for i, batch_param in reversed(list(enumerate(batch_params))):
                    if (
                        batch_param["start_sample"] in pred_ids
                        and batch_param["dannce_predict_dir"] == pred_dir
                    ):
                        del batch_params[i]
        return batch_params

    def remove_finished_batches(self, batch_params: List) -> List:
        """Remove finished batches from parameters list.

        Args:
            batch_params (List): Batch parameters list.

        Returns:
            (List): Updated batch parameters list.
        """
        if self.predict_path is None:
            params = load_params("io.yaml")
            if params["dannce_predict_dir"] is None:
                raise ValueError(
                    "Either predict_path (clarg) or dannce_predict_dir (in io.yaml) must be specified for merge"
                )
            else:
                self.predict_path = params["dannce_predict_dir"]
        if not os.path.exists(self.predict_path):
            os.makedirs(self.predict_path)

        # Get all of the files
        pred_files = [f for f in os.listdir(self.predict_path) if DANNCE_BASE_NAME in f]

        # Remove any of the default merged files.
        pred_files = [f for f in pred_files if f != (DANNCE_BASE_NAME + ".mat")]
        if len(pred_files) > 1:
            params = load_params(self.config)
            pred_ids = [
                int(f.split(".")[0].split("AVG")[1]) * params["batch_size"]
                for f in pred_files
            ]
            for i, batch_param in reversed(list(enumerate(batch_params))):
                if batch_param["start_sample"] in pred_ids:
                    del batch_params[i]
        return batch_params

    def submit_jobs(self, batch_params: List, cmd: str):
        """Print out description of command and issue system command

        Args:
            batch_params (List): Batch parameters list
            cmd (str): System command
        """
        if self.verbose:
            for batch_param in batch_params:
                print("Start sample:", batch_param["start_sample"])
                print("End sample:", batch_param["max_num_samples"])
            print("Command issued: ", cmd)
        if not self.test:
            return os.WEXITSTATUS(os.system(cmd))

    def submit_dannce_predict_multi_gpu(self):
        """Predict dannce over multiple gpus in parallel.

        Divide project into equal chunks of n_samples_per_gpu samples. Submit an array job
        that predicts over each chunk in parallel.
        """
        n_samples = self.get_n_samples(self.dannce_file, use_com=True)
        batch_params = self.generate_batch_params_dannce(n_samples)
        slurm_config = load_params(load_params(self.config)["slurm_config"])

        cmd = (
            'sbatch --wait --array=0-%d %s --wrap="%s dannce-predict-single-batch %s"'
            % (
                len(batch_params) - 1,
                slurm_config["dannce_multi_predict"],
                slurm_config["setup"],
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
        slurm_config = load_params(load_params(self.config)["slurm_config"])
        cmd = (
            'sbatch --wait --array=0-%d %s --wrap="%s com-predict-single-batch %s"'
            % (
                len(batch_params) - 1,
                slurm_config["com_multi_predict"],
                slurm_config["setup"],
                self.config,
            )
        )
        if len(batch_params) > 0:
            self.save_batch_params(batch_params)
            self.submit_jobs(batch_params, cmd)
        return batch_params, cmd

    def com_merge(self):
        """Merge com chunks into a single file.

        Raises:
            ValueError: If predict_path or com_predict_dir are not specified.
            FileNotFoundError: If no prediction files were found in the prediction dir
        """
        # Get all of the paths
        if self.predict_path is None:
            # Try to get it from io.yaml
            params = load_params("io.yaml")
            if params["com_predict_dir"] is None:
                raise ValueError(
                    "Either predict_path (clarg) or com_predict_dir (in io.yaml) must be specified for merge"
                )
            else:
                self.predict_path = params["com_predict_dir"]
        pred_files = [
            f
            for f in os.listdir(self.predict_path)
            if COM_BASE_NAME in f and ".mat" in f
        ]
        pred_files = [
            f
            for f in pred_files
            if f != (COM_BASE_NAME + ".mat") and "instance" not in f
        ]
        pred_inds = [int(f.split(COM_BASE_NAME)[-1].split(".")[0]) for f in pred_files]
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

        # if len(com.shape == 3), there are multiple instanes
        if len(com.shape) == 3:
            for n_instance in range(com.shape[2]):
                fn = os.path.join(
                    self.predict_path,
                    "instance" + str(n_instance) + COM_BASE_NAME + ".mat",
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
            fn = os.path.join(self.predict_path, COM_BASE_NAME + ".mat")
            savemat(fn, {"com": com, "sampleID": sampleID, "metadata": metadata})

    def dannce_merge(self):
        """Merge dannce chunks into a single file.

        Raises:
            ValueError: If predict_path or com_predict_dir are not specified.
            FileNotFoundError: If no prediction files were found in the prediction dir
        """
        # Get all of the paths
        if self.predict_path is None:
            # Try to get it from io.yaml
            params = load_params("io.yaml")
            if params["dannce_predict_dir"] is None:
                raise ValueError(
                    "Either predict_path (clarg) or dannce_predict_dir (in io.yaml) must be specified for merge"
                )
            else:
                self.predict_path = params["dannce_predict_dir"]
        pred_files = [f for f in os.listdir(self.predict_path) if DANNCE_BASE_NAME in f]
        pred_files = [f for f in pred_files if f != (DANNCE_BASE_NAME + ".mat")]
        pred_inds = [
            int(f.split(DANNCE_BASE_NAME)[-1].split(".")[0]) for f in pred_files
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
        fn = os.path.join(self.predict_path, DANNCE_BASE_NAME + ".mat")
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


def build_params_from_config_and_batch(
    config: Text, batch_param: Dict, dannce_net: bool = True
) -> Dict:
    """Build parameters from configuration file and batch parameters

    Args:
        config (Text): Path to base config .yaml file.
        batch_param (Dict): batch parameters dictionary
        dannce_net (bool, optional): If True, treat with defaults for dannce nets. Defaults to True.

    Returns:
        Dict: Parameters dictionary
    """
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
    """CLI entrypoint to predict a single batch."""
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
    """CLI entrypoint to predict a single batch."""
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
    params = build_params_from_config_and_batch(config, batch_param, dannce_net=False)

    # Predict
    try:
        com_predict(params)
    except OSError:
        # If a job writes to the label3d file at the same time as another reads from it
        # it throws an OSError.
        com_predict(params)


def inference():
    """CLI entrypoint to coordinate full inference job."""
    # Make parser object
    args = inference_clargs()

    # Load in parameters to modify
    handler = MultiGpuHandler(
        args["com_config"], only_unfinished=True, test=args["test"]
    )
    handler.submit_com_predict_multi_gpu()
    handler.submit_com_predict_multi_gpu()
    handler.submit_com_predict_multi_gpu()
    handler.submit_com_predict_multi_gpu()
    handler.submit_com_predict_multi_gpu()
    if args["test"]:
        print("Skipping com merge during test.")
    else:
        handler.com_merge()

    handler = MultiGpuHandler(
        args["dannce_config"], only_unfinished=True, test=args["test"]
    )
    handler.submit_dannce_predict_multi_gpu()
    handler.submit_dannce_predict_multi_gpu()
    handler.submit_dannce_predict_multi_gpu()
    handler.submit_dannce_predict_multi_gpu()
    handler.submit_dannce_predict_multi_gpu()
    if args["test"]:
        print("Skipping dannce merge during test.")
    else:
        handler.dannce_merge()


def submit_inference():
    """CLI entrypoint to submit jobs to coordinate full inference."""
    # Make parser object
    args = inference_clargs()
    com_config = load_params(args["com_config"])
    dannce_config = load_params(args["dannce_config"])
    slurm_config = load_params(dannce_config["slurm_config"])
    io_config = load_params("io.yaml")

    # Determine whether running multi instance or single instance
    for config in [com_config, dannce_config, io_config]:
        if "n_instances" in config:
            if config["n_instances"] >= 2:
                inference_command = "dannce-multi-instance-inference"
                break
            else:
                inference_command = "dannce-inference"
        else:
            inference_command = "dannce-inference"

    cmd = 'sbatch %s --wrap="%s %s %s %s"' % (
        slurm_config["inference"],
        slurm_config["setup"],
        inference_command,
        args["com_config"],
        args["dannce_config"],
    )
    print(cmd)
    if not args["test"]:
        os.system(cmd)


def inference_clargs():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("com_config", help="Path to .yaml configuration file")
    p.add_argument("dannce_config", help="Path to .yaml configuration file")
    p.add_argument(
        "--test",
        dest="test",
        type=ast.literal_eval,
        default=False,
        help="If True, print out submission command and info, but do not submit jobs.",
    )
    return p.parse_args().__dict__


def multi_instance_inference():
    args = inference_clargs()
    # Load in parameters to modify
    handler = MultiGpuHandler(
        args["com_config"], only_unfinished=True, test=args["test"]
    )
    handler.submit_com_predict_multi_gpu()
    handler.submit_com_predict_multi_gpu()
    handler.submit_com_predict_multi_gpu()
    if args["test"]:
        print("Skipping com merge during test.")
    else:
        handler.com_merge()

    handler = MultiGpuHandler(
        args["dannce_config"], only_unfinished=True, test=args["test"]
    )
    handler.submit_dannce_predict_multi_gpu()
    handler.submit_dannce_predict_multi_gpu()
    handler.submit_dannce_predict_multi_gpu()

    params = load_params("io.yaml")
    instance_0_path = os.path.join(params["dannce_predict_dir"], "instance0")
    instance_1_path = os.path.join(params["dannce_predict_dir"], "instance1")

    handler = MultiGpuHandler(
        args["dannce_config"], predict_path=instance_0_path, test=args["test"]
    )
    if args["test"]:
        print("Skipping dannce merge during test.")
    else:
        handler.dannce_merge()
    handler = MultiGpuHandler(
        args["dannce_config"], predict_path=instance_1_path, test=args["test"]
    )
    if args["test"]:
        print("Skipping dannce merge during test.")
    else:
        handler.dannce_merge()


def dannce_predict_multi_gpu():
    """CLI entrypoint to submit batch jobs."""
    # Load in parameters to modify
    args = cmdline_args()
    handler = MultiGpuHandler(**args.__dict__)
    handler.submit_dannce_predict_multi_gpu()


def com_predict_multi_gpu():
    """CLI entrypoint to submit batch jobs."""
    # Load in parameters to modify
    args = cmdline_args()
    handler = MultiGpuHandler(**args.__dict__)
    handler.submit_com_predict_multi_gpu()


def com_merge():
    """CLI entrypoint to merge batch jobs."""
    args = cmdline_args()
    handler = MultiGpuHandler(**args.__dict__)
    handler.com_merge()


def dannce_merge():
    """CLI entrypoint to merge batch jobs."""
    args = cmdline_args()
    handler = MultiGpuHandler(**args.__dict__)
    handler.dannce_merge()


def cmdline_args():
    """Handle command line arguments

    Returns:
        [type]: argparse parser values
    """
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
