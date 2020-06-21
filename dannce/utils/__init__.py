import h5py
import numpy as np


def load_label3d_data(path, key):
    with h5py.File(path, "r") as f:
        dataset = f[key]
        dataset = [f[ref] for ref in dataset[0]]
        dataset = [{k: np.array(d[k][:]).T for k in d.keys()} for d in dataset]
    return dataset


def load_camera_params(path):
    params = load_label3d_data(path, "params")
    for p in params:
        if "r" in p:
            p["R"] = p["r"]
    return params


def load_sync(path):
    return load_label3d_data(path, "sync")


def load_labels(path):
    return load_label3d_data(path, "labelData")
