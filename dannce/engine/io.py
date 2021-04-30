"""Data loading and saving operations."""
import numpy as np
import scipy.io as sio
from typing import List, Dict, Text, Union


def load_label3d_data(path: Text, key: Text):
    """Load Label3D data

    Args:
        path (Text): Path to Label3D file
        key (Text): Field to access

    Returns:
        TYPE: Data from field
    """
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


def load_camera_params(path: Text) -> List[Dict]:
    """Load camera parameters from Label3D file.

    Args:
        path (Text): Path to Label3D file

    Returns:
        List[Dict]: List of camera parameter dictionaries.
    """
    params = load_label3d_data(path, "params")
    for p in params:
        if "r" in p:
            p["R"] = p["r"]
    return params


def load_sync(path: Text) -> List[Dict]:
    """Load synchronization data from Label3D file.

    Args:
        path (Text): Path to Label3D file.

    Returns:
        List[Dict]: List of synchronization dictionaries.
    """
    dataset = load_label3d_data(path, "sync")
    for d in dataset:
        d["data_frame"] = d["data_frame"].astype(int)
        d["data_sampleID"] = d["data_sampleID"].astype(int)
    return dataset


def load_labels(path: Text) -> List[Dict]:
    """Load labelData from Label3D file.

    Args:
        path (Text): Path to Label3D file.

    Returns:
        List[Dict]: List of labelData dictionaries.
    """
    dataset = load_label3d_data(path, "labelData")
    for d in dataset:
        d["data_frame"] = d["data_frame"].astype(int)
        d["data_sampleID"] = d["data_sampleID"].astype(int)
    return dataset


def load_com(path: Text) -> Dict:
    """Load COM from .mat file.

    Args:
        path (Text): Path to .mat file with "com" field

    Returns:
        Dict: Dictionary with com data
    """
    d = sio.loadmat(path)["com"]
    data = {}
    data["com3d"] = d["com3d"][0, 0]
    data["sampleID"] = d["sampleID"][0, 0].astype(int)
    return data


def load_camnames(path: Text) -> Union[List, None]:
    """Load camera names from .mat file.

    Args:
        path (Text): Path to .mat file with "camnames" field

    Returns:
        Union[List, None]: List of cameranames
    """
    label_3d_file = sio.loadmat(path)
    if "camnames" in label_3d_file:
        names = label_3d_file["camnames"][:]
        if len(names) != len(label_3d_file["labelData"]):
            camnames = [name[0] for name in names[0]]
        else:
            camnames = [name[0][0] for name in names]
    else:
        camnames = None
    return camnames
