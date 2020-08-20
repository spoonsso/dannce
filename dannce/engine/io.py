import numpy as np
import scipy.io as sio


def load_label3d_data(path, key):
    d = sio.loadmat(path)[key]
    dataset = [f[0] for f in d]

    # Data are loaded in this annoying structure where the array
    # we want is at dataset[i][key][0,0], as a nested array of arrays.
    # Simplify this structure (a numpy record array) here.
    # Additionally, cannot use views here because of shape mismatches. Define
    # new dict and return.
    data = []
    for d in dataset:
        d_ = {}
        for key in d.dtype.names:
            d_[key] = d[key][0, 0]
        data.append(d_)

    return data


def load_camera_params(path):
    params = load_label3d_data(path, "params")
    for p in params:
        if "r" in p:
            p["R"] = p["r"]
    return params


def load_sync(path):
    dataset = load_label3d_data(path, "sync")
    for d in dataset:
        d["data_frame"] = d["data_frame"].astype(int)
        d["data_sampleID"] = d["data_sampleID"].astype(int)
    return dataset


def load_labels(path):
    dataset = load_label3d_data(path, "labelData")
    for d in dataset:
        d["data_frame"] = d["data_frame"].astype(int)
        d["data_sampleID"] = d["data_sampleID"].astype(int)
    return dataset


def load_com(path):
    d = sio.loadmat(path)["com"]
    data = {}
    data["com3d"] = d["com3d"][0, 0]
    data["sampleID"] = d["sampleID"][0, 0].astype(int)
    return data


def load_camnames(path):
    r = sio.loadmat(path)
    if "camnames" in r:
        s = [f[0] for f in r["camnames"][0]]
    else:
        s = None
    return s
