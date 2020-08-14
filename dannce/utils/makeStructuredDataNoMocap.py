"""
This script converts the output from predict_DANNCE into a predescribed
structured data format, with meta data, that can be used easily by our
downstream analysis pipeline

Usage:
python path_to_file/makeStructuredData.py path_to_config path_to_skeleton path_to_danncemat

path_to_template is an optional parameter for times when I don't have any labeling directory
"""
import numpy as np
import scipy.io as sio
import sys
import os
import dannce.engine.processing as processing
import dannce.engine.io as io
import ast
from dannce import _param_defaults_shared, _param_defaults_dannce, _param_defaults_com

# Set up parameters
PARENT_PARAMS = processing.read_config(sys.argv[1])
PARENT_PARAMS = processing.make_paths_safe(PARENT_PARAMS)
CONFIG_PARAMS = processing.read_config(PARENT_PARAMS["io_config"])
CONFIG_PARAMS = processing.make_paths_safe(CONFIG_PARAMS)
CONFIG_PARAMS = processing.inherit_config(CONFIG_PARAMS,
                                          PARENT_PARAMS,
                                          list(PARENT_PARAMS.keys()))

defaults = {**_param_defaults_dannce,**_param_defaults_shared}
CONFIG_PARAMS = processing.inherit_config(CONFIG_PARAMS,
                                        defaults,
                                        list(defaults.keys()))

CONFIG_PARAMS["camnames"] = None
CONFIG_PARAMS = processing.infer_params(CONFIG_PARAMS, True, False)

RESULTSDIR = CONFIG_PARAMS["dannce_predict_dir"]
print("Reading results from: " + RESULTSDIR)

# This is agnostic to the expval setting, i.e. MAX or AVG net
# However, we will eventually add the COM back in only for MAX-type results
dfiles = os.listdir(RESULTSDIR)
sfile = [f for f in dfiles if "save_data" in f]
sfile = sfile[0]

pred = sio.loadmat(os.path.join(RESULTSDIR, sfile))

pred["sampleID"] = np.squeeze(pred["sampleID"])

istherexpval = "expval" in CONFIG_PARAMS and not CONFIG_PARAMS["expval"]
istherenettype = "net_type" in CONFIG_PARAMS and CONFIG_PARAMS["net_type"] != "AVG"


if istherexpval or istherenettype:
    print("adding 3D COM back in")
    com3d = sio.loadmat(os.path.join(RESULTSDIR, "com3d_used.mat"))
    # We should make sure the sampleIDs match up
    # assert np.all(com3d['sampleID'] == pred['sampleID'])
    # We need to loop over all, double check sampleID alignment, and then add com back
    for j in range(len(pred["sampleID"].ravel())):
        comind = np.where(com3d["sampleID"] == pred["sampleID"][j])[0]
        assert len(comind) == 1
        pred["pred"][j] = pred["pred"][j] + com3d["com"][j, :, np.newaxis]

"""
We want the following to be saved:
1) animal name & data
2) network name
3) predictions in struct form, which includes the markernames, incorporate sampleID and data_frame for each camera into this.
4) camera calibration params, and links to the videos in the above struct

"""

fullpath = os.path.realpath("./")
name = os.path.split(os.path.realpath("./"))[-1]
date = os.path.split(os.path.realpath("../"))[-1]
netname = CONFIG_PARAMS["net"]

# Get the weights path, a useful piece of metadata
if "dannce_predict_model" in CONFIG_PARAMS.keys():
    mdl_file = CONFIG_PARAMS["dannce_predict_model"]
else:
    wdir = CONFIG_PARAMS["dannce_train_dir"]
    weights = os.listdir(wdir)
    weights = [f for f in weights if ".hdf5" in f]
    weights = sorted(weights, key=lambda x: int(x.split(".")[1].split("-")[0]))
    weights = weights[-1]

    mdl_file = os.path.join(wdir, weights)

weightspath = mdl_file

# Load in markernames from skeleton file
markernames = sio.loadmat(sys.argv[2])
markernames = [r[0][0] for r in markernames['joint_names']]

# In Matlab, we cannot keep parentheses in the marker names
markernames = [m.replace("(", "_") for m in markernames]
markernames = [m.replace(")", "_") for m in markernames]

# Assemble prediction struct
predictions = {}
for m in range(len(markernames)):
    predictions[markernames[m]] = pred["pred"][:, :, m]

predictions["sampleID"] = pred["sampleID"]

cameras = {}

l3dfile = sys.argv[3]
cnames = io.load_camnames(l3dfile)
syncs = io.load_sync(l3dfile)
cparams = io.load_camera_params(l3dfile)

for i in range(len(cnames)):
    cameras[cnames[i]] = {}

    # Load in this camera's matched frames file to align the sampleID
    # with that particular cameras' frame #
    mframes = syncs[i]

    # df = np.zeros((len(pred['sampleID']), 1))

    _, inds, _ = np.intersect1d(
        mframes["data_sampleID"], pred["sampleID"], return_indices=True
    )

    assert len(inds) == len(pred["sampleID"])

    df = mframes["data_frame"][inds]

    cameras[cnames[i]]["frame"] = df

    camparams = cparams[i]

    cameras[cnames[i]]["IntrinsicMatrix"] = camparams["K"]
    cameras[cnames[i]]["rotationMatrix"] = camparams["r"]
    cameras[cnames[i]]["translationVector"] = camparams["t"]
    cameras[cnames[i]]["TangentialDistortion"] = camparams["TDistort"]
    cameras[cnames[i]]["RadialDistortion"] = camparams["RDistort"]

    if "viddir" in CONFIG_PARAMS:
        viddir = CONFIG_PARAMS["viddir"]
    else:
        viddir = os.path.join(os.path.dirname(l3dfile),'videos')

    cameras[cnames[i]]["video_directory"] = os.path.realpath(
        os.path.join(viddir, cnames[i])
    )

# save everything as matlab struct
pfile = os.path.join(RESULTSDIR, "predictions.mat")
pdict = {
    "fullpath": fullpath,
    "name": name,
    "session": date,
    "netname": netname,
    "predictions": predictions,
    "cameras": cameras,
    "p_max": pred["p_max"],
    "weightspath": mdl_file,
}

sio.savemat(pfile, pdict)

print("Saved predictions to " + pfile)
