import scipy.io as sio
import numpy as np


def load_data(path, key):
    d = sio.loadmat(path,struct_as_record=False)
    dataset = vars(d[key][0][0])

    # Data are loaded in this annoying structure where the array
    # we want is at dataset[i][key][0,0], as a nested array of arrays.
    # Simplify this structure (a numpy record array) here.
    # Additionally, cannot use views here because of shape mismatches. Define
    # new dict and return.

    import pdb;pdb.set_trace()
    data = []
    for d in dataset:
        d_ = {}
        for key in d.dtype.names:
            d_[key] = d[key][0, 0]
        data.append(d_)

    return data


def load_cameras(path):
    d = sio.loadmat(path,struct_as_record=False)
    dataset = vars(d["cameras"][0][0])

    camnames = dataset['_fieldnames']

    cameras = {}
    for i in range(len(camnames)):
        cameras[camnames[i]] = {}
        cam = vars(dataset[camnames[i]][0][0])
        fns = cam['_fieldnames']
        for fn in fns:
            cameras[camnames[i]][fn] = cam[fn]

    return cameras


def load_mocap(path):
    d = sio.loadmat(path,struct_as_record=False)
    dataset = vars(d["mocap"][0][0])

    markernames = dataset['_fieldnames']

    mocap = []
    for i in range(len(markernames)):
        mocap.append(dataset[markernames[i]])

    return np.stack(mocap, axis=2)