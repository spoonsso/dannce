"""Processing functions for dannce."""
from dataclasses import replace
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import downscale_local_mean as dsm
import imageio
import os
import dannce.engine.serve_data_DANNCE as serve_data_DANNCE
import PIL
from six.moves import cPickle
import scipy.io as sio
from scipy.ndimage.filters import maximum_filter

from dannce.engine import io
import matplotlib
import warnings

# matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml
import shutil
import time
from typing import Dict
from tensorflow.keras.models import Model


def write_debug(
    params: Dict,
    ims_train: np.ndarray,
    ims_valid: np.ndarray,
    y_train: np.ndarray,
    model: Model,
    trainData: bool = True,
):
    """Factoring re-used debug output code.

    Args:
        params (Dict): Parameters dictionary
        ims_train (np.ndarray): Training images
        ims_valid (np.ndarray): Validation images
        y_train (np.ndarray): Training targets
        model (Model): Model
        trainData (bool, optional): If True use training data for debug. Defaults to True.
    """

    def plot_out(imo, lo, imn):
        plot_markers_2d(norm_im(imo), lo, newfig=False)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        imname = imn
        plt.savefig(os.path.join(debugdir, imname), bbox_inches="tight", pad_inches=0)

    if params["debug"] and not params["multi_mode"]:

        if trainData:
            outdir = "debug_im_out"
            ims_out = ims_train
            label_out = y_train
        else:
            outdir = "debug_im_out_valid"
            ims_out = ims_valid
            label_out = model.predict(ims_valid, batch_size=1)

        # Plot all training images and save
        # create new directory for images if necessary
        debugdir = os.path.join(params["com_train_dir"], outdir)
        print("Saving debug images to: " + debugdir)
        if not os.path.exists(debugdir):
            os.makedirs(debugdir)

        plt.figure()

        for i in range(ims_out.shape[0]):
            plt.cla()
            if params["mirror"]:
                for j in range(label_out.shape[-1]):
                    plt.cla()
                    plot_out(
                        ims_out[i],
                        label_out[i, :, :, j : j + 1],
                        str(i) + "_cam_" + str(j) + ".png",
                    )
            else:
                plot_out(ims_out[i], label_out[i], str(i) + ".png")

    elif params["debug"] and params["multi_mode"]:
        print("Note: Cannot output debug information in COM multi-mode")


def initialize_vids(params, datadict, e, vids, pathonly=True, vidkey="viddir"):
    """
    Initializes video path dictionaries for a training session. This is different
        than a predict session because it operates over a single animal ("experiment")
        at a time
    """
    for i in range(len(params["experiment"][e]["camnames"])):
        # Rather than opening all vids, only open what is needed based on the
        # maximum frame ID for this experiment and Camera
        flist = []
        for key in datadict.keys():
            if int(key.split("_")[0]) == e:
                flist.append(
                    datadict[key]["frames"][params["experiment"][e]["camnames"][i]]
                )

        flist = max(flist)

        # For COM prediction, we don't prepend experiment IDs
        # So detect this case and act accordingly.
        basecam = params["experiment"][e]["camnames"][i]
        if "_" in basecam:
            basecam = basecam.split("_")[1]

        if params["vid_dir_flag"]:
            addl = ""
        else:
            addl = os.listdir(
                os.path.join(
                    params["experiment"][e][vidkey],
                    basecam,
                )
            )[0]
        r = generate_readers(
            params["experiment"][e][vidkey],
            os.path.join(basecam, addl),
            maxopt=flist,  # Large enough to encompass all videos in directory.
            extension=params["experiment"][e]["extension"],
            pathonly=pathonly,
        )

        if "_" in params["experiment"][e]["camnames"][i]:
            vids[params["experiment"][e]["camnames"][i]] = {}
            for key in r:
                vids[params["experiment"][e]["camnames"][i]][str(e) + "_" + key] = r[
                    key
                ]
        else:
            vids[params["experiment"][e]["camnames"][i]] = r

    return vids


def infer_params(params, dannce_net, prediction):
    """
    Some parameters that were previously specified in configs can just be inferred
        from others, thus relieving config bloat
    """
    # Grab the camnames from *dannce.mat if not in config
    if params["camnames"] is None:
        f = grab_predict_label3d_file()
        params["camnames"] = io.load_camnames(f)
        if params["camnames"] is None:
            raise Exception("No camnames in config or in *dannce.mat")

    # Infer vid_dir_flag and extension and n_channels_in and chunks
    # from the videos and video folder organization.
    # Look into the video directory / camnames[0]. Is there a video file?
    # If so, vid_dir_flag = True
    viddir = os.path.join(params["viddir"], params["camnames"][0])
    video_files = os.listdir(viddir)

    if any([".mp4" in file for file in video_files]) or any(
        [".avi" in file for file in video_files]
    ):

        print_and_set(params, "vid_dir_flag", True)
    else:
        print_and_set(params, "vid_dir_flag", False)
        viddir = os.path.join(viddir, video_files[0])
        video_files = os.listdir(viddir)

    extension = ".mp4" if any([".mp4" in file for file in video_files]) else ".avi"
    print_and_set(params, "extension", extension)
    video_files = [file for file in video_files if extension in file]

    # Use the camnames to find the chunks for each video
    chunks = {}
    for name in params["camnames"]:
        if params["vid_dir_flag"]:
            camdir = os.path.join(params["viddir"], name)
        else:
            camdir = os.path.join(params["viddir"], name)
            intermediate_folder = os.listdir(camdir)
            camdir = os.path.join(camdir, intermediate_folder[0])
        video_files = os.listdir(camdir)
        video_files = [f for f in video_files if extension in f]
        video_files = sorted(video_files, key=lambda x: int(x.split(".")[0]))
        chunks[name] = np.sort([int(x.split(".")[0]) for x in video_files])

    print_and_set(params, "chunks", chunks)

    firstvid = str(chunks[params["camnames"][0]][0]) + params["extension"]
    camf = os.path.join(viddir, firstvid)

    # Infer n_channels_in from the video info
    v = imageio.get_reader(camf)
    im = v.get_data(0)
    v.close()
    print_and_set(params, "n_channels_in", im.shape[-1])

    # set the raw im height and width
    print_and_set(params, "raw_im_h", im.shape[0])
    print_and_set(params, "raw_im_w", im.shape[1])

    if dannce_net and params["avg+max"] is not None:
        # To use avg+max, need to start with an AVG network
        # In case the net type is not properly specified, set it here
        print_and_set(params, "expval", True)
        print_and_set(params, "net_type", "AVG")

    if dannce_net and params["net"] is None:
        # Here we assume that if the network and expval are specified by the user
        # then there is no reason to infer anything. net + expval compatibility
        # are subsequently verified during check_config()
        #
        # If both the net and expval are unspecified, then we use the simpler
        # 'net_type' + 'train_mode' to select the correct network and set expval.
        # During prediction, the train_mode might be missing, and in any case only the
        # expval needs to be set.
        if params["net_type"] is None:
            raise Exception("Without a net name, net_type must be specified")

        if not prediction and params["train_mode"] is None:
            raise Exception("Need to specific train_mode for DANNCE training")

        if params["net_type"] == "AVG":
            print_and_set(params, "expval", True)
        elif params["net_type"] == "MAX":
            print_and_set(params, "expval", False)
        else:
            raise Exception("{} not a valid net_type".format(params["net_type"]))

        # if not prediction:
        if params["net_type"] == "AVG" and params["train_mode"] == "finetune":
            print_and_set(params, "net", "finetune_AVG")
        elif params["net_type"] == "AVG":
            # This is the network for training from scratch.
            # This will also set the network for "continued", but that network
            # will be ignored, as for continued training the full model file
            # is loaded in without a call to construct the network. However, a value
            # for params['net'] is still required for initialization
            print_and_set(params, "net", "unet3d_big_expectedvalue")
        elif params["net_type"] == "MAX" and params["train_mode"] == "finetune":
            print_and_set(params, "net", "finetune_MAX")
        elif params["net_type"] == "MAX":
            print_and_set(params, "net", "unet3d_big")

    elif dannce_net and params["expval"] is None:
        if "AVG" in params["net"] or "expected" in params["net"]:
            print_and_set(params, "expval", True)
        else:
            print_and_set(params, "expval", False)

    if dannce_net:
        # infer crop_height and crop_width if None. Just use max dims of video, as
        # DANNCE does not need to crop.
        if params["crop_height"] is None or params["crop_width"] is None:
            im_h = []
            im_w = []
            for i in range(len(params["camnames"])):
                viddir = os.path.join(params["viddir"], params["camnames"][i])
                if not params["vid_dir_flag"]:
                    # add intermediate directory to path
                    viddir = os.path.join(
                        params["viddir"], params["camnames"][i], os.listdir(viddir)[0]
                    )
                video_files = sorted(os.listdir(viddir))
                camf = os.path.join(viddir, video_files[0])
                v = imageio.get_reader(camf)
                im = v.get_data(0)
                v.close()
                im_h.append(im.shape[0])
                im_w.append(im.shape[1])

            if params["crop_height"] is None:
                print_and_set(params, "crop_height", [0, np.max(im_h)])
            if params["crop_width"] is None:
                print_and_set(params, "crop_width", [0, np.max(im_w)])

        if params["max_num_samples"] is not None:
            if params["max_num_samples"] == "max":
                print_and_set(params, "maxbatch", "max")
            elif isinstance(params["max_num_samples"], (int, np.integer)):
                print_and_set(
                    params,
                    "maxbatch",
                    int(params["max_num_samples"] // params["batch_size"]),
                )
            else:
                raise TypeError("max_num_samples must be an int or 'max'")
        else:
            print_and_set(params, "maxbatch", "max")

        if params["start_sample"] is not None:
            if isinstance(params["start_sample"], (int, np.integer)):
                print_and_set(
                    params,
                    "start_batch",
                    int(params["start_sample"] // params["batch_size"]),
                )
            else:
                raise TypeError("start_sample must be an int.")
        else:
            print_and_set(params, "start_batch", 0)

        if params["vol_size"] is not None:
            print_and_set(params, "vmin", -1 * params["vol_size"] / 2)
            print_and_set(params, "vmax", params["vol_size"] / 2)

        if params["heatmap_reg"] and not params["expval"]:
            raise Exception(
                "Heatmap regularization enabled only for AVG networks -- you are using MAX"
            )

        if params["n_rand_views"] == "None":
            print_and_set(params, "n_rand_views", None)
        
        TEMPORAL_FLAG = (params["temporal_loss_weight"] is not None) and (params["temporal_chunk_size"] is not None) 
        print_and_set(params, "use_temporal", TEMPORAL_FLAG)
        print_and_set(params, "use_silhouette", params["silhouette_loss_weight"] is not None)

        SEPARATION_FLAG = (params["separation_loss_weight"] is not None) and (params["separation_delta"] is not None)
        print_and_set(params, "use_separation", SEPARATION_FLAG)

        print_and_set(params, "use_symmetry", params["symmetry_loss_weight"] is not None)

    # There will be strange behavior if using a mirror acquisition system and are cropping images
    if params["mirror"] and params["crop_height"][-1] != params["raw_im_h"]:
        msg = "Note: You are using a mirror acquisition system with image cropping."
        msg = (
            msg
            + " All coordinates will be flipped relative to the raw image height, so ensure that your labels are also in that reference frame."
        )
        warnings.warn(msg)

    # Handle COM network name backwards compatibility
    if params["net"].lower() == "unet2d_fullbn":
        print_and_set(params, "norm_method", "batch")
    elif params["net"] == "unet2d_fullIN":
        print_and_set(params, "norm_method", "layer")

    if not dannce_net:
        print_and_set(params, "net", "unet2d_full")

    return params


def print_and_set(params, varname, value):
    # Should add new values to params in place, no need to return
    params[varname] = value
    print("Setting {} to {}.".format(varname, params[varname]))


def check_config(params, dannce_net, prediction):
    """
    Add parameter checks and restrictions here.
    """
    check_camnames(params)

    if params["exp"] is not None:
        for expdict in params["exp"]:
            check_camnames(expdict)

    if dannce_net:
        check_net_expval(params)
        check_vmin_vmax(params)


def check_vmin_vmax(params):
    for v in ["vmin", "vmax", "nvox"]:
        if params[v] is None:
            raise Exception(
                "{} not in parameters. Please add it, or use vol_size instead of vmin and vmax".format(
                    v
                )
            )


def get_ft_wt(params):
    if params["dannce_finetune_weights"] is not None:
        weights = os.listdir(params["dannce_finetune_weights"])
        weights = [f for f in weights if ".hdf5" in f]
        weights = weights[0]

        return os.path.join(params["dannce_finetune_weights"], weights)


def check_camnames(camp):
    """
    Raises an exception if camera names contain '_'
    """
    if "camnames" in camp:
        for cam in camp["camnames"]:
            if "_" in cam:
                raise Exception("Camera names cannot contain '_' ")


def check_net_expval(params):
    """
    Raise an exception if the network and expval (i.e. AVG/MAX) are incompatible
    """
    if params["net"] is None:
        raise Exception("net is None. You must set either net or net_type.")
    if params["net_type"] is not None:
        if (
            params["net_type"] == "AVG"
            and "AVG" not in params["net"]
            and "expected" not in params["net"]
        ):
            raise Exception("net_type is set to AVG, but you are using a MAX network")
        if (
            params["net_type"] == "MAX"
            and "MAX" not in params["net"]
            and params["net"] != "unet3d_big"
        ):
            raise Exception("net_type is set to MAX, but you are using a AVG network")

    if (
        params["expval"]
        and "AVG" not in params["net"]
        and "expected" not in params["net"]
    ):
        raise Exception("expval is set to True but you are using a MAX network")
    if (
        not params["expval"]
        and "MAX" not in params["net"]
        and params["net"] != "unet3d_big"
    ):
        raise Exception("expval is set to False but you are using an AVG network")


def copy_config(results_dir, main_config, io_config):
    """
    Copies config files into the results directory, and creates results
        directory if necessary
    """
    print("Saving results to: {}".format(results_dir))

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    mconfig = os.path.join(
        results_dir, "copy_main_config_" + main_config.split(os.sep)[-1]
    )
    dconfig = os.path.join(results_dir, "copy_io_config_" + io_config.split(os.sep)[-1])

    shutil.copyfile(main_config, mconfig)
    shutil.copyfile(io_config, dconfig)


def make_data_splits(samples, params, results_dir, num_experiments, temporal_chunks=None):
    """
    Make train/validation splits from list of samples, or load in a specific
        list of sampleIDs if desired.
    """
    # TODO: Switch to .mat from .pickle so that these lists are easier to read
    # and change.

    partition = {}
    if params["use_temporal"]:
        assert temporal_chunks != None, "If use temporal, do partitioning over chunks."
        v = params["num_validation_per_exp"]
        # fix random seeds
        if params["data_split_seed"] is not None:
            np.random.seed(params["data_split_seed"])
        
        valid_chunks, train_chunks = [], []
        if params["valid_exp"] is not None and v > 0:
            for e in range(num_experiments):
                if e in params["valid_exp"]:
                    valid_chunk_idx = sorted(np.random.choice(len(temporal_chunks[e]), v, replace=False))
                    valid_chunks += list(np.array(temporal_chunks[e])[valid_chunk_idx])
                    train_chunks += list(np.delete(temporal_chunks[e], valid_chunk_idx, 0))
                else:
                    train_chunks += temporal_chunks[e]
        elif v > 0:
            for e in range(num_experiments):
                valid_chunk_idx = sorted(np.random.choice(len(temporal_chunks[e]), v, replace=False))
                valid_chunks += list(np.array(temporal_chunks[e])[valid_chunk_idx])
                train_chunks += list(np.delete(temporal_chunks[e], valid_chunk_idx, 0))

        train_expts = np.arange(num_experiments)
        print("TRAIN EXPTS: {}".format(train_expts))

        train_sampleIDs, valid_sampleIDs = list(np.concatenate(train_chunks)), list(np.concatenate(valid_chunks))
        partition["train_sampleIDs"], partition["valid_sampleIDs"] = train_sampleIDs, valid_sampleIDs
        chunk_size = len(train_chunks[0])
        partition["train_chunks"] = [np.arange(i, i+chunk_size) for i in range(0, len(train_sampleIDs), chunk_size)]
        partition["valid_chunks"] = [np.arange(i, i+chunk_size) for i in range(0, len(valid_sampleIDs), chunk_size)]

        # Save train/val inds
        with open(os.path.join(results_dir, "val_samples.pickle"), "wb") as f:
            cPickle.dump(partition["valid_sampleIDs"], f)

        with open(os.path.join(results_dir, "train_samples.pickle"), "wb") as f:
            cPickle.dump(partition["train_sampleIDs"], f)

        return partition


    if params["load_valid"] is None:
        # Set random seed if included in params
        if params["data_split_seed"] is not None:
            np.random.seed(params["data_split_seed"])

        all_inds = np.arange(len(samples))

        # extract random inds from each set for validation
        v = params["num_validation_per_exp"]
        valid_inds = []
        if params["valid_exp"] is not None and v > 0:
            all_valid_inds = []
            for e in params["valid_exp"]:
                tinds = [
                    i for i in range(len(samples)) if int(samples[i].split("_")[0]) == e
                ]
                all_valid_inds = all_valid_inds + tinds
                valid_inds = valid_inds + list(
                    np.random.choice(tinds, (v,), replace=False)
                )
                valid_inds = list(np.sort(valid_inds))

            train_inds = list(
                set(all_inds) - set(all_valid_inds)
            )  # [i for i in all_inds if i not in all_valid_inds]
        elif v > 0:  # if 0, do not perform validation
            for e in range(num_experiments):
                tinds = [
                    i for i in range(len(samples)) if int(samples[i].split("_")[0]) == e
                ]
                valid_inds = valid_inds + list(
                    np.random.choice(tinds, (v,), replace=False)
                )
                valid_inds = list(np.sort(valid_inds))

            train_inds = [i for i in all_inds if i not in valid_inds]
        elif params["valid_exp"] is not None:
            raise Exception("Need to set num_validation_per_exp in using valid_exp")
        else:
            train_inds = all_inds

        assert (set(valid_inds) & set(train_inds)) == set()

        train_samples = samples[train_inds]
        train_inds = []
        if params["valid_exp"] is not None:
            train_expts = [f for f in range(num_experiments) if f not in params["valid_exp"]]
        else:
            train_expts = np.arange(num_experiments)

        print("TRAIN EXPTS: {}".format(train_expts))

        if params["num_train_per_exp"] is not None:
            # Then sample randomly without replacement from training sampleIDs
            for e in train_expts:
                tinds = [
                    i
                    for i in range(len(train_samples))
                    if int(train_samples[i].split("_")[0]) == e
                ]
                print(e)
                print(len(tinds))
                train_inds = train_inds + list(
                    np.random.choice(
                        tinds, (params["num_train_per_exp"],), replace=False
                    )
                )
                train_inds = list(np.sort(train_inds))
        else:
            train_inds = np.arange(len(train_samples))

        partition["valid_sampleIDs"] = samples[valid_inds]
        partition["train_sampleIDs"] = train_samples[train_inds]

        # Save train/val inds
        with open(os.path.join(results_dir, "val_samples.pickle"), "wb") as f:
            cPickle.dump(partition["valid_sampleIDs"], f)

        with open(os.path.join(results_dir, "train_samples.pickle"), "wb") as f:
            cPickle.dump(partition["train_sampleIDs"], f)
    else:
        # Load validation samples from elsewhere
        with open(
            os.path.join(params["load_valid"], "val_samples.pickle"),
            "rb",
        ) as f:
            partition["valid_sampleIDs"] = cPickle.load(f)
        partition["train_sampleIDs"] = [
            f for f in samples if f not in partition["valid_sampleIDs"]
        ]

    # Reset any seeding so that future batch shuffling, etc. are not tied to this seed
    if params["data_split_seed"] is not None:
        np.random.seed()
    
    return partition


def __initAvgMax(t, g, o, params):
    """
    Helper function for creating 3D targets
    """
    gridsize = tuple([params["nvox"]] * 3)
    g = np.reshape(
        g,
        (-1, *gridsize, 3),
    )

    for i in range(o.shape[0]):
        for j in range(o.shape[-1]):
            o[i, ..., j] = np.exp(
                -(
                    (g[i, ..., 1] - t[i, 1, j]) ** 2
                    + (g[i, ..., 0] - t[i, 0, j]) ** 2
                    + (g[i, ..., 2] - t[i, 2, j]) ** 2
                )
                / (2 * params["sigma"] ** 2)
            )

    return o


def initAvgMax(y_train, y_valid, Xtg, Xvg, params):
    """
    Converts 3D coordinate targets into 3D volumes, for AVG+MAX training
    """
    gridsize = tuple([params["nvox"]] * 3)
    y_train_aux = np.zeros(
        (
            y_train.shape[0],
            *gridsize,
            params["new_n_channels_out"],
        ),
        dtype="float32",
    )

    y_valid_aux = np.zeros(
        (
            y_valid.shape[0],
            *gridsize,
            params["new_n_channels_out"],
        ),
        dtype="float32",
    )

    return (
        __initAvgMax(y_train, Xtg, y_train_aux, params),
        __initAvgMax(y_valid, Xvg, y_valid_aux, params),
    )


def remove_samples_npy(npydir, samples, params):
    """
    Remove any samples from sample list if they do not have corresponding volumes in the image
        or grid directories
    """
    # image_volumes
    # grid_volumes
    samps = []
    for e in npydir.keys():
        imvol = os.path.join(npydir[e], "image_volumes")
        gridvol = os.path.join(npydir[e], "grid_volumes")
        ims = os.listdir(imvol)
        grids = os.listdir(gridvol)
        npysamps = [
            "0_" + f.split("_")[1] + ".npy"
            for f in samples
            if int(f.split("_")[0]) == e
        ]

        goodsamps = list(set(npysamps) & set(ims) & set(grids))

        samps = samps + [
            str(e) + "_" + f.split("_")[1].split(".")[0] for f in goodsamps
        ]

        sampdiff = len(npysamps) - len(goodsamps)

        # import pdb; pdb.set_trace()
        print(
            "Removed {} samples from {} because corresponding image or grid files could not be found".format(
                sampdiff, params["experiment"][e]["label3d_file"]
            )
        )

    return np.array(samps)


def rename_weights(traindir, kkey, mon):
    """
    At the end of DANNCe or COM training, rename the best weights file with the epoch #
        and value of the monitored quantity
    """
    # First load in the training.csv
    r = np.genfromtxt(os.path.join(traindir, "training.csv"), delimiter=",", names=True)
    e = r["epoch"]
    q = r[mon]
    minq = np.min(q)
    if e.size == 1:
        beste = e
    else:
        beste = e[np.argmin(q)]

    newname = "weights." + str(int(beste)) + "-" + "{:.5f}".format(minq) + ".hdf5"

    os.rename(os.path.join(traindir, kkey), os.path.join(traindir, newname))


def make_paths_safe(params):
    """Given a parameter dictionary, loops through the keys and replaces any \\ or / with os.sep
    to promote OS agnosticism
    """
    for key in params.keys():
        if isinstance(params[key], str):
            params[key] = params[key].replace("/", os.sep)
            params[key] = params[key].replace("\\", os.sep)

    return params


def trim_COM_pickle(fpath, start_sample, end_sample, opath=None):
    """Trim dictionary entries to the range [start_sample, end_sample].

    spath is the output path for saving the trimmed COM dictionary, if desired
    """
    with open(fpath, "rb") as f:
        save_data = cPickle.load(f)
    sd = {}

    for key in save_data:
        if key >= start_sample and key <= end_sample:
            sd[key] = save_data[key]

    with open(opath, "wb") as f:
        cPickle.dump(sd, f)
    return sd


def save_params(outdir, params):
    """
    Save copy of params to outdir as .mat file
    """
    sio.savemat(os.path.join(outdir, "copy_params.mat"), prepare_save_metadata(params))

    return True


def make_none_safe(pdict):
    if isinstance(pdict, dict):
        for key in pdict:
            pdict[key] = make_none_safe(pdict[key])
    else:
        if (
            pdict is None
            or (isinstance(pdict, list) and None in pdict)
            or (isinstance(pdict, tuple) and None in pdict)
        ):
            return "None"
        else:
            return pdict
    return pdict


def prepare_save_metadata(params):
    """
    To save metadata, i.e. the prediction param values associated with COM or DANNCE
        output, we need to convert loss and metrics and net into names, and remove
        the 'experiment' field
    """

    # Need to convert None to string but still want to conserve the metadat structure
    # format, so we don't want to convert the whole dict to a string
    meta = params.copy()
    if "experiment" in meta:
        del meta["experiment"]
    if "loss" in meta:
        try: 
            meta["loss"] = [loss.__name__ for loss in meta["loss"]]
        except:
            meta["loss"] = meta["loss"].__name__
    if "net" in meta:
        meta["net"] = meta["net"].__name__
    if "metric" in meta:
        meta["metric"] = [
            f.__name__ if not isinstance(f, str) else f for f in meta["metric"]
        ]

    meta = make_none_safe(meta.copy())
    return meta


def save_COM_dannce_mat(params, com3d, sampleID):
    """
    Instead of saving 3D COM to com3d.mat, save it into the dannce.mat file, which
    streamlines subsequent dannce access.
    """
    com = {}
    com["com3d"] = com3d
    com["sampleID"] = sampleID
    com["metadata"] = prepare_save_metadata(params)

    # Open dannce.mat file, add com and re-save
    print("Saving COM predictions to " + params["label3d_file"])
    rr = sio.loadmat(params["label3d_file"])
    # For safety, save old file to temp and delete it at the end
    sio.savemat(params["label3d_file"] + ".temp", rr)
    rr["com"] = com
    sio.savemat(params["label3d_file"], rr)

    os.remove(params["label3d_file"] + ".temp")


def save_COM_checkpoint(
    save_data, results_dir, datadict_, cameras, params, file_name="com3d"
):
    """
    Saves COM pickle and matfiles

    """
    # Save undistorted 2D COMs and their 3D triangulations
    f = open(os.path.join(results_dir, file_name + ".pickle"), "wb")
    cPickle.dump(save_data, f)
    f.close()

    # We need to remove the eID in front of all the keys in datadict
    # for prepare_COM to run properly
    datadict_save = {}
    for key in datadict_:
        datadict_save[int(float(key.split("_")[-1]))] = datadict_[key]

    if params["n_instances"] > 1:
        if params["n_channels_out"] > 1:
            linking_method = "multi_channel"
        else:
            linking_method = "euclidean"
        _, com3d_dict = serve_data_DANNCE.prepare_COM_multi_instance(
            os.path.join(results_dir, file_name + ".pickle"),
            datadict_save,
            comthresh=0,
            weighted=False,
            camera_mats=cameras,
            linking_method=linking_method,
        )
    else:
        prepare_func = serve_data_DANNCE.prepare_COM
        _, com3d_dict = serve_data_DANNCE.prepare_COM(
            os.path.join(results_dir, file_name + ".pickle"),
            datadict_save,
            comthresh=0,
            weighted=False,
            camera_mats=cameras,
            method="median",
        )

    cfilename = os.path.join(results_dir, file_name + ".mat")
    print("Saving 3D COM to {}".format(cfilename))
    samples_keys = list(com3d_dict.keys())

    if params["n_instances"] > 1:
        c3d = np.zeros((len(samples_keys), 3, params["n_instances"]))
    else:
        c3d = np.zeros((len(samples_keys), 3))

    for i in range(len(samples_keys)):
        c3d[i] = com3d_dict[samples_keys[i]]

    sio.savemat(
        cfilename,
        {
            "sampleID": samples_keys,
            "com": c3d,
            "metadata": prepare_save_metadata(params),
        },
    )

    # Also save a copy into the label3d file
    # save_COM_dannce_mat(params, c3d, samples_keys)


def inherit_config(child, parent, keys):
    """
    If a key in keys does not exist in child, assigns the key-value in parent to
        child.
    """
    for key in keys:
        if key not in child.keys():
            child[key] = parent[key]
            print(
                "{} not found in io.yaml file, falling back to main config".format(key)
            )

    return child


def grab_predict_label3d_file(defaultdir=""):
    """
    Finds the paths to the training experiment yaml files.
    """
    def_ep = os.path.join(".", defaultdir)
    label3d_files = os.listdir(def_ep)
    label3d_files = [
        os.path.join(def_ep, f) for f in label3d_files if "dannce.mat" in f
    ]
    label3d_files.sort()

    if len(label3d_files) == 0:
        raise Exception("Did not find any *dannce.mat file in {}".format(def_ep))
    print("Using the following *dannce.mat files: {}".format(label3d_files[0]))
    return label3d_files[0]

def load_expdict(params, e, expdict, _DEFAULT_VIDDIR, _DEFAULT_VIDDIR_SIL):
    """
    Load in camnames and video directories and label3d files for a single experiment
        during training.
    """
    _DEFAULT_NPY_DIR = "npy_volumes"
    exp = params.copy()
    exp = make_paths_safe(exp)
    exp["label3d_file"] = expdict["label3d_file"]
    exp["base_exp_folder"] = os.path.dirname(exp["label3d_file"])

    if "viddir" not in expdict:
        # if the videos are not at the _DEFAULT_VIDDIR, then it must
        # be specified in the io.yaml experiment portion
        exp["viddir"] = os.path.join(exp["base_exp_folder"], _DEFAULT_VIDDIR)
    else:
        exp["viddir"] = expdict["viddir"]

    print("Experiment {} using videos in {}".format(e, exp["viddir"]))

    if params["use_silhouette"]:
        exp["viddir_sil"] = os.path.join(exp["base_exp_folder"], _DEFAULT_VIDDIR_SIL) if "viddir_sil" not in expdict else expdict["viddir_sil"]
        print("Experiment {} also using masked videos in {}".format(e, exp["viddir_sil"]))

    l3d_camnames = io.load_camnames(expdict["label3d_file"])
    if "camnames" in expdict:
        exp["camnames"] = expdict["camnames"]
    elif l3d_camnames is not None:
        exp["camnames"] = l3d_camnames
    print("Experiment {} using camnames: {}".format(e, exp["camnames"]))

    # Use the camnames to find the chunks for each video
    chunks = {}
    for name in exp["camnames"]:
        if exp["vid_dir_flag"]:
            camdir = os.path.join(exp["viddir"], name)
        else:
            camdir = os.path.join(exp["viddir"], name)
            intermediate_folder = os.listdir(camdir)
            camdir = os.path.join(camdir, intermediate_folder[0])
        video_files = os.listdir(camdir)
        video_files = [f for f in video_files if ".mp4" in f]
        video_files = sorted(video_files, key=lambda x: int(x.split(".")[0]))
        chunks[str(e) + "_" + name] = np.sort(
            [int(x.split(".")[0]) for x in video_files]
        )
    exp["chunks"] = chunks
    print(chunks)

    # For npy volume training
    if params["use_npy"]:
        exp["npy_vol_dir"] = os.path.join(exp["base_exp_folder"], _DEFAULT_NPY_DIR)
    return exp


def batch_rgb2gray(imstack):
    """Convert to gray image-wise.

    batch dimension is first.
    """
    grayim = np.zeros((imstack.shape[0], imstack.shape[1], imstack.shape[2]), "float32")
    for i in range(grayim.shape[0]):
        grayim[i] = rgb2gray(imstack[i].astype("uint8"))
    return grayim


def return_tile(imstack, fac=2):
    """Crop a larger image into smaller tiles without any overlap."""
    height = imstack.shape[1] // fac
    width = imstack.shape[2] // fac
    out = np.zeros(
        (imstack.shape[0] * fac * fac, height, width, imstack.shape[3]), "float32"
    )
    cnt = 0
    for i in range(imstack.shape[0]):
        for j in np.arange(0, imstack.shape[1], height):
            for k in np.arange(0, imstack.shape[2], width):
                out[cnt, :, :, :] = imstack[i, j : j + height, k : k + width, :]
                cnt = cnt + 1
    return out


def tile2im(imstack, fac=2):
    """Reconstruct lagrer image from tiled data."""
    height = imstack.shape[1]
    width = imstack.shape[2]
    out = np.zeros(
        (imstack.shape[0] // (fac * fac), height * fac, width * fac, imstack.shape[3]),
        "float32",
    )
    cnt = 0
    for i in range(out.shape[0]):
        for j in np.arange(0, out.shape[1], height):
            for k in np.arange(0, out.shape[2], width):
                out[i, j : j + height, k : k + width, :] = imstack[cnt]
                cnt += 1
    return out


def downsample_batch(imstack, fac=2, method="PIL"):
    """Downsample each image in a batch."""

    if method == "PIL":
        out = np.zeros(
            (
                imstack.shape[0],
                imstack.shape[1] // fac,
                imstack.shape[2] // fac,
                imstack.shape[3],
            ),
            "float32",
        )
        if out.shape[-1] == 3:
            # this is just an RGB image, so no need to loop over channels with PIL
            for i in range(imstack.shape[0]):
                out[i] = np.array(
                    PIL.Image.fromarray(imstack[i].astype("uint8")).resize(
                        (out.shape[2], out.shape[1]), resample=PIL.Image.LANCZOS
                    )
                )
        else:
            for i in range(imstack.shape[0]):
                for j in range(imstack.shape[3]):
                    out[i, :, :, j] = np.array(
                        PIL.Image.fromarray(imstack[i, :, :, j]).resize(
                            (out.shape[2], out.shape[1]), resample=PIL.Image.LANCZOS
                        )
                    )

    elif method == "dsm":
        out = np.zeros(
            (
                imstack.shape[0],
                imstack.shape[1] // fac,
                imstack.shape[2] // fac,
                imstack.shape[3],
            ),
            "float32",
        )
        for i in range(imstack.shape[0]):
            for j in range(imstack.shape[3]):
                out[i, :, :, j] = dsm(imstack[i, :, :, j], (fac, fac))

    elif method == "nn":
        out = imstack[:, ::fac, ::fac]

    elif fac > 1:
        raise Exception("Downfac > 1. Not a valid downsampling method")

    return out


def batch_maximum(imstack):
    """Find the location of the maximum for each image in a batch."""
    maxpos = np.zeros((imstack.shape[0], 2))
    for i in range(imstack.shape[0]):
        if np.isnan(imstack[i, 0, 0]):
            maxpos[i, 0] = np.nan
            maxpos[i, 1] = np.nan
        else:
            ind = np.unravel_index(
                np.argmax(np.squeeze(imstack[i]), axis=None),
                np.squeeze(imstack[i]).shape,
            )
            maxpos[i, 0] = ind[1]
            maxpos[i, 1] = ind[0]
    return maxpos


def generate_readers(
    viddir, camname, minopt=0, maxopt=300000, pathonly=False, extension=".mp4"
):
    """Open all mp4 objects with imageio, and return them in a dictionary."""
    out = {}
    mp4files = [
        os.path.join(camname, f)
        for f in os.listdir(os.path.join(viddir, camname))
        if extension in f
        and int(f.rsplit(extension)[0]) <= maxopt
        and int(f.rsplit(extension)[0]) >= minopt
    ]

    # This is a trick (that should work) for getting rid of
    # awkward sub-directory folder names when they are being used
    mp4files_scrub = [
        os.path.join(
            os.path.normpath(f).split(os.sep)[0], os.path.normpath(f).split(os.sep)[-1]
        )
        for f in mp4files
    ]

    pixelformat = "yuv420p"
    input_params = []
    output_params = []

    for i in range(len(mp4files)):
        if pathonly:
            out[mp4files_scrub[i]] = os.path.join(viddir, mp4files[i])
        else:
            print(
                "NOTE: Ignoring {} files numbered above {}".format(extensions, maxopt)
            )
            out[mp4files_scrub[i]] = imageio.get_reader(
                os.path.join(viddir, mp4files[i]),
                pixelformat=pixelformat,
                input_params=input_params,
                output_params=output_params,
            )

    return out


def cropcom(im, com, size=512):
    """Crops single input image around the coordinates com."""
    minlim_r = int(np.round(com[1])) - size // 2
    maxlim_r = int(np.round(com[1])) + size // 2
    minlim_c = int(np.round(com[0])) - size // 2
    maxlim_c = int(np.round(com[0])) + size // 2

    out = im[np.max([minlim_r, 0]) : maxlim_r, np.max([minlim_c, 0]) : maxlim_c, :]

    dim = out.shape[2]

    # pad with zeros if region ended up outside the bounds of the original image
    if minlim_r < 0:
        out = np.concatenate(
            (np.zeros((abs(minlim_r), out.shape[1], dim)), out), axis=0
        )
    if maxlim_r > im.shape[0]:
        out = np.concatenate(
            (out, np.zeros((maxlim_r - im.shape[0], out.shape[1], dim))), axis=0
        )
    if minlim_c < 0:
        out = np.concatenate(
            (np.zeros((out.shape[0], abs(minlim_c), dim)), out), axis=1
        )
    if maxlim_c > im.shape[1]:
        out = np.concatenate(
            (out, np.zeros((out.shape[0], maxlim_c - im.shape[1], dim))), axis=1
        )

    return out


def write_config(results_dir, configdict, message, filename="modelconfig.cfg"):
    """Write a dictionary of k-v pairs to file.

    A much more customizable configuration writer. Accepts a dictionary of
    key-value pairs and just writes them all to file,
    together with a custom message
    """
    f = open(results_dir + filename, "w")
    for key in configdict:
        f.write("{}: {}\n".format(key, configdict[key]))
    f.write("message:" + message)


def read_config(filename):
    """Read configuration file.

    :param filename: Path to configuration file.
    """
    with open(filename) as f:
        params = yaml.safe_load(f)

    return params


def plot_markers_2d(im, markers, newfig=True):
    """Plot markers in two dimensions."""

    if newfig:
        plt.figure()
    plt.imshow((im - np.min(im)) / (np.max(im) - np.min(im)))

    for mark in range(markers.shape[-1]):
        ind = np.unravel_index(
            np.argmax(markers[:, :, mark], axis=None), markers[:, :, mark].shape
        )
        plt.plot(ind[1], ind[0], ".r")


def preprocess_3d(im_stack):
    """Easy inception-v3 style image normalization across a set of images."""
    im_stack /= 127.5
    im_stack -= 1.0
    return im_stack


def norm_im(im):
    """Normalize image."""
    return (im - np.min(im)) / (np.max(im) - np.min(im))

def plot_markers_3d_torch(stack, nonan=True):
    """Return the 3d coordinates for each of the peaks in probability maps."""
    import torch

    n_mark = stack.shape[-1]
    index = stack.flatten(0, 2).argmax(dim=0).to(torch.int32)
    inds = unravel_index(index, stack.shape[:-1])
    if ~torch.any(torch.isnan(stack[0, 0, 0, :])) and (nonan or not nonan):
        x = inds[1]
        y = inds[0]
        z = inds[2]
    elif not nonan:
        x = inds[1]
        y = inds[0]
        z = inds[2]
        for mark in range(0, n_mark):
            if torch.isnan(stack[:, :, :, mark]):
                x[mark] = torch.nan
                y[mark] = torch.nan
                z[mark] = torch.nan
    return x, y, z


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def grid_channelwise_max(grid_):
    """Return the max value in each channel over a 3D volume.

    input--
        grid_: shape (nvox, nvox, nvox, nchannels)

    output--
        shape (nchannels,)
    """
    return np.max(np.max(np.max(grid_, axis=0), axis=0), axis=0)


def moment_3d(im, mesh, thresh=0):
    """Get the normalized spatial moments of the 3d image stack.

    inputs--
        im: 3d volume confidence map, one for each channel (marker)
            i.e. shape (nvox,nvox,nvox,nchannels)
        mesh: spatial coordinates for every position on im
        thresh: threshold applied to im before calculating moments
    """
    x = []
    y = []
    z = []
    for mark in range(im.shape[3]):
        # get normalized probabilities
        im_norm = (im[:, :, :, mark] * (im[:, :, :, mark] >= thresh)) / np.sum(
            im[:, :, :, mark] * (im[:, :, :, mark] >= thresh)
        )
        x.append(np.sum(mesh[0] * im_norm))
        y.append(np.sum(mesh[1] * im_norm))
        z.append(np.sum(mesh[2] * im_norm))
    return x, y, z


def get_peak_inds(map_):
    """Return the indices of the peak value of an n-d map."""
    return np.unravel_index(np.argmax(map_, axis=None), map_.shape)


def get_peak_inds_multi_instance(im, n_instances, window_size=10):
    """Return top n_instances local peaks through non-max suppression."""
    bw = im == maximum_filter(im, footprint=np.ones((window_size, window_size)))
    inds = np.argwhere(bw)
    vals = im[inds[:, 0], inds[:, 1]]
    idx = np.argsort(vals)[::-1]
    return inds[idx[:n_instances], :]


def get_marker_peaks_2d(stack):
    """Return the concatenated coordinates of all peaks for each map/marker."""
    x = []
    y = []
    for i in range(stack.shape[-1]):
        inds = get_peak_inds(stack[:, :, i])
        x.append(inds[1])
        y.append(inds[0])
    return x, y


def savedata_expval(
    fname, params, write=True, data=None, num_markers=20, tcoord=True, pmax=False
):
    """Save the expected values."""
    if data is None:
        f = open(fname, "rb")
        data = cPickle.load(f)
        f.close()

    d_coords = np.zeros((len(list(data.keys())), 3, num_markers))
    t_coords = np.zeros((len(list(data.keys())), 3, num_markers))
    sID = np.zeros((len(list(data.keys())),))
    p_max = np.zeros((len(list(data.keys())), num_markers))

    for (i, key) in enumerate(data.keys()):
        d_coords[i] = data[key]["pred_coord"]
        if tcoord:
            t_coords[i] = np.reshape(data[key]["true_coord_nogrid"], (3, num_markers))
        if pmax:
            p_max[i] = data[key]["pred_max"]
        sID[i] = data[key]["sampleID"]

        sdict = {
            "pred": d_coords,
            "data": t_coords,
            "p_max": p_max,
            "sampleID": sID,
            "metadata": prepare_save_metadata(params),
        }
    if write and data is None:
        sio.savemat(
            fname.split(".pickle")[0] + ".mat",
            sdict,
        )
    elif write and data is not None:
        sio.savemat(fname, sdict)

    return d_coords, t_coords, p_max, sID


def savedata_tomat(
    fname,
    params,
    vmin,
    vmax,
    nvox,
    write=True,
    data=None,
    num_markers=20,
    tcoord=True,
    tcoord_scale=True,
    addCOM=None,
):
    """Save pickled data to a mat file.

    From a save_data structure saved to a *.pickle file, save a matfile
        with useful variables for easier manipulation in matlab.
    Also return pred_out_world and other variables for plotting within jupyter
    """
    if data is None:
        f = open(fname, "rb")
        data = cPickle.load(f)
        f.close()

    d_coords = np.zeros((list(data.keys())[-1] + 1, 3, num_markers))
    t_coords = np.zeros((list(data.keys())[-1] + 1, 3, num_markers))
    p_max = np.zeros((list(data.keys())[-1] + 1, num_markers))
    log_p_max = np.zeros((list(data.keys())[-1] + 1, num_markers))
    sID = np.zeros((list(data.keys())[-1] + 1,))
    for (i, key) in enumerate(data.keys()):
        d_coords[i] = data[key]["pred_coord"]
        if tcoord:
            t_coords[i] = np.reshape(data[key]["true_coord_nogrid"], (3, num_markers))
        p_max[i] = data[key]["pred_max"]
        log_p_max[i] = data[key]["logmax"]
        sID[i] = data[key]["sampleID"]

    vsize = (vmax - vmin) / nvox
    # First, need to move coordinates over to centers of voxels
    pred_out_world = vmin + d_coords * vsize + vsize / 2

    if tcoord and tcoord_scale:
        t_coords = vmin + t_coords * vsize + vsize / 2

    if addCOM is not None:
        # We use the passed comdict to add back in the com, this is useful
        # if one wnats to bootstrap on these values for COMnet or otherwise
        for i in range(len(sID)):
            pred_out_world[i] = pred_out_world[i] + addCOM[int(sID)][:, np.newaxis]

    sdict = {
        "pred": pred_out_world,
        "data": t_coords,
        "p_max": p_max,
        "sampleID": sID,
        "log_pmax": log_p_max,
        "metadata": prepare_save_metadata(params),
    }
    if write and data is None:
        sio.savemat(
            fname.split(".pickle")[0] + ".mat",
            sdict,
        )
    elif write and data is not None:
        sio.savemat(
            fname,
            sdict,
        )
    return pred_out_world, t_coords, p_max, log_p_max, sID


def spatial_expval(map_):
    """Calculate the spatial expected value of the input.

    Note there is probably underflow here that I am ignoring, because this
    doesn't need to be *that* accurate
    """
    map_ = map_ / np.sum(map_)
    x, y = np.meshgrid(np.arange(map_.shape[1]), np.arange(map_.shape[0]))

    return np.sum(map_ * x), np.sum(map_ * y)


def spatial_var(map_):
    """Calculate the spatial variance of the input."""
    expx, expy = spatial_expval(map_)
    map_ = map_ / np.sum(map_)
    x, y = np.meshgrid(np.arange(map_.shape[1]), np.arange(map_.shape[0]))

    return np.sum(map_ * ((x - expx) ** 2 + (y - expy) ** 2))


def spatial_entropy(map_):
    """Calculate the spatial entropy of the input."""
    map_ = map_ / np.sum(map_)
    return -1 * np.sum(map_ * np.log(map_))


def write_npy(uri, gen):
    """
    Creates a new image folder and grid folder at the uri and uses
    the generator to generate samples and save them as npy files
    """
    imdir = os.path.join(uri, "image_volumes")
    if not os.path.exists(imdir):
        os.makedirs(imdir)

    griddir = os.path.join(uri, "grid_volumes")
    if not os.path.exists(griddir):
        os.makedirs(griddir)

    # Make sure rotation and shuffle are turned off
    gen.channel_combo = None
    gen.shuffle = False
    gen.rotation = False
    gen.expval = True

    # Turn normalization off so that we can save as uint8
    gen.norm_im = False

    bs = gen.batch_size
    for i in range(len(gen)):
        if i % 1000 == 0:
            print(i)
        # Generate batch
        bch = gen.__getitem__(i)
        # loop over all examples in batch and save volume
        for j in range(bs):
            # get the frame name / unique ID
            fname = gen.list_IDs[gen.indexes[i * bs + j]]

            # and save
            print(fname)
            np.save(os.path.join(imdir, fname + ".npy"), bch[0][0][j].astype("uint8"))
            np.save(os.path.join(griddir, fname + ".npy"), bch[0][1][j])

def extract_3d_sil(vol, upper_thres):
    vol[vol > 0] = 1
    vol = np.sum(vol, axis=-1, keepdims=True)

    vol[vol < upper_thres] = -1 #0
    vol[vol > 0] = 1

    print("{}\% of silhouette training voxels are occupied".format(
            100*np.sum(vol)/len(vol.ravel())))
    return vol

def extract_3d_sil_soft(vol, upper_thres, chan_num, keeprange=3):
    vol[vol > 0] = 1
    vol = np.sum(vol, axis=-1, keepdims=True)

    lower_thres = upper_thres - keeprange*chan_num
    vol[vol <= lower_thres] = -1 #0
    vol[vol > 0] = (vol[vol > 0] - lower_thres) / (keeprange*chan_num)

    print("{}\% of silhouette training voxels are occupied".format(
            100*np.sum((vol > 0))/len(vol.ravel())))
    return vol

def load_volumes_into_mem(params, partition, n_cams, generator, train=True, silhouette=False):
    n_samples = len(partition["train_sampleIDs"]) if train else len(partition["valid_sampleIDs"]) 
    message = "Loading training data into memory" if train else "Loading validation data into memory"
    gridsize = tuple([params["nvox"]] * 3)

    X = np.empty((n_samples, *gridsize, params["chan_num"]*n_cams), dtype="float32")
    print(message)

    X_grid = None
    if params["expval"]:
        if not silhouette: 
            y = np.empty((n_samples, 3, params["new_n_channels_out"]), dtype="float32")
            X_grid = np.empty((n_samples, params["nvox"] ** 3, 3), dtype="float32")
    else:
        y = np.empty((n_samples, *gridsize, params["new_n_channels_out"]), dtype="float32")

    for i in range(n_samples):
        print(i, end="\r")
        rr = generator.__getitem__(i)
        if params["expval"]:
            X[i] = rr[0][0]
            if not silhouette: 
                X_grid[i], y[i] = rr[0][1], rr[1]
        else:
            X[i], y[i] = rr[0], rr[1]

    if silhouette:
        print("Now loading silhouettes")

        if params["soft_silhouette"]:
            X = extract_3d_sil_soft(X, params["chan_num"]*n_cams, params["chan_num"])
        else:
            X = extract_3d_sil(X, params["chan_num"]*n_cams)
        
        return None, None, X
    
    return X, X_grid, y

