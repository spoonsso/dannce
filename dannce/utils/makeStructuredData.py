"""
This script converts the output from predict_DANNCE into a predescribed
structured data format, with meta data, that can be used easily by our
downstream analysis pipeline

Usage:
python path_to_file/makeStructuredData.py path_to_config [cross_corr?]

the optional boolean argument cross_corr indicates whether a cross-correlation metric
should be used to further align mocap vs. prediction at the end of analysis.

path_to_template is an optional parameter for times when I don't have any labeling directory
"""
import numpy as np
import scipy.io as sio
import sys
import os
import dannce.engine.processing as processing
import ast
import h5py
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Set up parameters
    PARENT_PARAMS = processing.read_config(sys.argv[1])
    PARENT_PARAMS = processing.make_paths_safe(PARENT_PARAMS)
    CONFIG_PARAMS = processing.read_config(PARENT_PARAMS["DANNCE_CONFIG"])
    CONFIG_PARAMS = processing.make_paths_safe(CONFIG_PARAMS)

    CONFIG_PARAMS["experiment"] = PARENT_PARAMS
    RESULTSDIR = CONFIG_PARAMS["RESULTSDIR_PREDICT"]
    print("Reading results from: " + RESULTSDIR)

    # This is agnostic to the expval setting, i.e. MAX or AVG net
    # However, we will eventually add the COM back in only for MAX-type results
    dfiles = os.listdir(RESULTSDIR)
    sfile = [f for f in dfiles if "save_data" in f]
    sfile = sfile[0]

    pred = sio.loadmat(os.path.join(RESULTSDIR, sfile))

    pred["sampleID"] = np.squeeze(pred["sampleID"])

    if not CONFIG_PARAMS["expval"]:
        print("adding 3D COM back in")
        com3d = sio.loadmat(os.path.join(RESULTSDIR, "COM3D_undistorted.mat"))
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
    if "predict_model" in CONFIG_PARAMS.keys():
        mdl_file = CONFIG_PARAMS["predict_model"]
    else:
        wdir = CONFIG_PARAMS["RESULTSDIR"]
        weights = os.listdir(wdir)
        weights = [f for f in weights if ".hdf5" in f]
        weights = sorted(weights, key=lambda x: int(x.split(".")[1].split("-")[0]))
        weights = weights[-1]

        mdl_file = os.path.join(wdir, weights)

    weightspath = mdl_file

    with open(PARENT_PARAMS["lbl_template"], "r") as f:
        for line in f:
            if "labels" in line:
                line = line.strip()
                line = line[8:-1]
                markernames = ast.literal_eval(line)

    # In Matlab, we cannot keep parentheses in the marker names
    markernames = [m.replace("(", "_") for m in markernames]
    markernames = [m.replace(")", "_") for m in markernames]

    # Assemble prediction struct
    predictions = {}
    for m in range(len(markernames)):
        predictions[markernames[m]] = pred["pred"][:, :, m]

    predictions["sampleID"] = pred["sampleID"]

    cameras = {}

    cnames = PARENT_PARAMS["camnames"]
    for i in range(len(cnames)):
        cameras[cnames[i]] = {}

        # Load in this camera's matched frames file to align the sampleID
        # with that particular cameras' frame #
        mframes = sio.loadmat(
            os.path.join(PARENT_PARAMS["datadir"], PARENT_PARAMS["datafile"][i])
        )

        mframes["data_sampleID"] = np.squeeze(mframes["data_sampleID"])
        mframes["data_frame"] = np.squeeze(mframes["data_frame"])

        # df = np.zeros((len(pred['sampleID']), 1))

        _, inds, _ = np.intersect1d(
            mframes["data_sampleID"], pred["sampleID"], return_indices=True
        )

        assert len(inds) == len(pred["sampleID"])

        df = mframes["data_frame"][inds]

        # for j in range(df.shape[0]):
        #     if pred['sampleID'][j] in mframes['data_sampleID']:
        #         df[j] = mframes['data_frame'][mframes['data_sampleID'] == pred['sampleID'][j]]
        #     else:
        #         raise Exception("Could not find sampleID in matched frames")

        cameras[cnames[i]]["frame"] = df

        camparams = sio.loadmat(
            os.path.join(PARENT_PARAMS["CALIBDIR"], PARENT_PARAMS["calib_file"][i])
        )

        cameras[cnames[i]]["IntrinsicMatrix"] = camparams["K"]
        cameras[cnames[i]]["rotationMatrix"] = camparams["r"]
        cameras[cnames[i]]["translationVector"] = camparams["t"]
        cameras[cnames[i]]["TangentialDistortion"] = camparams["TDistort"]
        cameras[cnames[i]]["RadialDistortion"] = camparams["RDistort"]

        cameras[cnames[i]]["video_directory"] = os.path.realpath(
            os.path.join(PARENT_PARAMS["viddir"], cnames[i])
        )

    if "mocap" in CONFIG_PARAMS.keys():
        # Then we need to load in all diles in the mocap list, extract the mocap,
        # and align it to the predictions. One problem is that using new MatchedFrames,
        # the sampleID is no longer related to the sampleIDs inside the mocap files.
        # One thing we can try is to use a single camera's data_frame
        # Note that this assumes the order of cameras in mocap matched_frames_aligned
        # matches the order of cameras in the config files
        camn = 5
        dframe = cameras[cnames[camn]]["frame"]
        mocap = {}
        for m in range(len(markernames)):
            mocap[markernames[m]] = np.zeros((len(dframe), 3)) * np.nan

        for mfile in CONFIG_PARAMS["mocap"]:
            mfile_ = os.path.join(CONFIG_PARAMS["mocapdir"], mfile)
            print("Loading mocap data from: " + mfile)
            with h5py.File(mfile_, "r") as r:
                mframes = [r[element[0]][:] for element in r["matched_frames_aligned"]]
                mframes = np.squeeze(mframes[camn]).astype("int")
                # get indices of all mframes that are in dframe
                # lazily, we can do this in a loop, because
                # faster intersections are tricky given that matched_frames are repeated multiple times
                # another shortcut we are taking here is that we take a single mocap value rather than an average
                # over all samples for a given frame
                # Another problem is that the mframes might jump from oen file to the next, leaving a big gap
                # relative to predictions
                dfsi = np.logical_and(dframe >= np.min(mframes), dframe <= np.max(mframes))
                dframe_sub = dframe[dfsi]
                mframes_ind = np.zeros((len(dframe_sub),)) * np.nan
                for i in range(len(dframe_sub)):
                    mframe_ind = np.where(mframes == dframe_sub[i])[0]
                    if len(mframe_ind) == 0:
                        print("Detected missing frame in mocap")
                        print(i)
                        print(dframe_sub[i])
                    else:
                        mframes_ind[i] = mframe_ind[0]

                badinds = np.where(np.isnan(mframes_ind))[0]
                mframes_ind[badinds] = 0
                mframes_ind = mframes_ind.astype("int")

                for m in range(len(markernames)):
                    g = np.array(r["markers_preproc"][markernames[m]])
                    g = g[:, mframes_ind]

                    g[:, badinds] = np.nan
                    mocap[markernames[m]][dfsi] = g.T

        # Load the markernames from mouse.template
        # Get markernames from template file
        if len(sys.argv) > 2 and sys.argv[2] == "True":
            # At the end, check if there is a shift of mocap relative to predictions, and correct:
            # Use HeadF only for shift detection
            shifts = np.arange(-10, 10)
            smax = 0
            pmax = 0

            # The marker used for alignment can be changed here. E.g. when aligning using a
            # COM trace, it helps to use SpineM instead of HeadF
            m = 0
            mg = mocap[markernames[m]].T.copy()
            pg = predictions[markernames[m]].T.copy()
            mg[np.isnan(mg)] = 0
            pg[np.isnan(pg)] = 0

            for s in shifts:

                if s < 0:
                    p = np.sum((mg[:, np.abs(s) :] * pg[:, :s])[:, 100:-100])
                elif s == 0:
                    p = np.sum((mg * pg)[:, 100:-100])
                else:
                    p = np.sum((mg[:, :-s] * pg[:, s:])[:, 100:-100])

                print("shift {}, p {}".format(s, p))

                if p > pmax:
                    print("Found better shift: {}".format(s))
                    smax = s
                    pmax = p

            for m in range(len(markernames)):
                if smax < 0:
                    mocap[markernames[m]] = mocap[markernames[m]][np.abs(smax) :, :]
                    mocap[markernames[m]] = np.concatenate(
                        (mocap[markernames[m]], np.zeros((np.abs(smax), 3)) * np.nan),
                        axis=0,
                    )
                elif smax > 0:
                    # To shift positively, we need to add NaN
                    mocap[markernames[m]] = np.concatenate(
                        (np.zeros((smax, 3)) * np.nan, mocap[markernames[m]]), axis=0
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

    if "mocap" in CONFIG_PARAMS.keys():
        pdict["mocap"] = mocap
        pdict["mocapdir"] = CONFIG_PARAMS["mocapdir"]
        pdict["mocapfiles"] = CONFIG_PARAMS["mocap"]
        pdict["shift"] = smax if len(sys.argv) > 2 and sys.argv[2] == "True" else 0

    sio.savemat(pfile, pdict)

    print("Saved predictions to " + pfile)
