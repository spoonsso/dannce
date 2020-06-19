# This script loads in data mat file, takes only the "clean"
# indices, i.e. where there are no nans in the labels, and
# then selects N random samples, saving the output to a
# shorter datafile that can be loaded in entirely into
# train_DANNCE.py
#
# Usage python subselectDataFile.py path_to_datafile_folder num_samples datafile_key [max sample]
#

import sys
import numpy as np
import scipy.io as sio
import os

df = sys.argv[1]
ns = int(sys.argv[2])
dfkey = sys.argv[3]

if len(sys.argv) > 4:
    ms = int(sys.argv[4])

dfs = [f for f in os.listdir(df) if dfkey in f]

# Load in the first datafile
data = sio.loadmat(os.path.join(df, dfs[0]))
data_3d = data["data_3d"]
data_2d = data["data_2d"]
data_frame = data["data_frame"]
data_sampleID = data["data_sampleID"]

# Find indices where there are no nans in data_3d
inds = np.where(~np.isnan(np.mean(data_3d, axis=1)))[0]

# if max_sample is present, truncate inds
if len(sys.argv) > 4:
    inds = inds[inds <= ms]

# sample from inds without replacement
inds = inds[np.random.choice(np.arange(len(inds)), (ns,), replace=False)]

# sort inds so that samples can be read in faster when loading the data during training/eval
inds = np.sort(inds)

# Use these inds and apply to every datafile for each camera
for i in range(len(dfs)):
    data = sio.loadmat(os.path.join(df, dfs[i]))
    data_3d = data["data_3d"]
    data_2d = data["data_2d"]
    data_frame = data["data_frame"]
    data_sampleID = data["data_sampleID"]

    data_3d = data_3d[inds]
    data_2d = data_2d[inds]
    data_frame = data_frame[inds]
    data_sampleID = data_sampleID[inds]

    # Now save with a new filename
    dfn = os.path.join(df, dfs[i].split(".mat")[0] + "_subselected.mat")

    sio.savemat(
        dfn,
        {
            "data_3d": data_3d,
            "data_2d": data_2d,
            "data_frame": data_frame,
            "data_sampleID": data_sampleID,
        },
    )
