"""
This script converts the output from predict_DANNCE into a predescribed
structured data format, with meta data, that can be used easily by our
downstream analysis pipeline

Usage:
python path_to_file/makeStructuredDataNoMocap.py path_to_prediction_file path_to_skeleton_file path_to_label3d_file

"""
import numpy as np
import scipy.io as sio
import sys
import os
import dannce.engine.processing as processing
import dannce.engine.io as io
import ast
from dannce import _param_defaults_shared, _param_defaults_dannce, _param_defaults_com

# get .mat into correct format
def _gf(f):
    g = f[0][0][0]
    if isinstance(g, np.str):
        g = str(g)
    else:
        g = g[0]
    return g

    
if __name__ == "__main__":
    # Read in predictions
    pfile = sys.argv[1]
    RESULTSDIR = os.path.dirname(pfile)
    print("Reading results from: " + pfile)

    pred = sio.loadmat(pfile)
    meta = pred['metadata']
    CONFIG_PARAMS = {}
    for key in meta.dtype.names:
        CONFIG_PARAMS[key] = _gf(meta[key])

    # This is agnostic to the expval setting, i.e. MAX or AVG net
    # However, we will eventually add the COM back in only for MAX-type results
    pred["sampleID"] = np.squeeze(pred["sampleID"])

    if not CONFIG_PARAMS["expval"]:
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
        print(mdl_file)
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

    mm = markernames['joint_names']
    while np.any(np.array(mm.shape) == 1):
        mm = np.squeeze(mm)
    markernames = [r[0] for r in np.squeeze(mm)]

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
