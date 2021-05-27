"""Handle training and prediction for DANNCE and COM networks."""
import sys
import numpy as np
import os
from copy import deepcopy
import scipy.io as sio
import imageio
import time
import gc
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.losses as keras_losses
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
import dannce.engine.serve_data_DANNCE as serve_data_DANNCE
from dannce.engine.generator import DataGenerator_3Dconv
from dannce.engine.generator import DataGenerator_3Dconv_frommem
from dannce.engine.generator import DataGenerator_3Dconv_npy
from dannce.engine.generator import DataGenerator_3Dconv_torch
from dannce.engine.generator import DataGenerator_3Dconv_tf
from dannce.engine.generator_aux import (
    DataGenerator_downsample,
    DataGenerator_downsample_multi_instance,
)
from dannce.engine.generator_aux import DataGenerator_downsample_frommem
import dannce.engine.processing as processing
from dannce.engine.processing import savedata_tomat, savedata_expval
from dannce.engine import nets, losses, ops, io
from dannce import (
    _param_defaults_dannce,
    _param_defaults_shared,
    _param_defaults_com,
)
import dannce.engine.inference as inference
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Text

_DEFAULT_VIDDIR = "videos"
_DEFAULT_COMSTRING = "COM"
_DEFAULT_COMFILENAME = "com3d.mat"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def check_unrecognized_params(params: Dict):
    """Check for invalid keys in the params dict against param defaults.

    Args:
        params (Dict): Parameters dictionary.

    Raises:
        ValueError: Error if there are unrecognized keys in the configs.
    """
    # Check if key in any of the defaults
    invalid_keys = []
    for key in params:
        in_com = key in _param_defaults_com
        in_dannce = key in _param_defaults_dannce
        in_shared = key in _param_defaults_shared
        if not (in_com or in_dannce or in_shared):
            invalid_keys.append(key)

    # If there are any keys that are invalid, throw an error and print them out
    if len(invalid_keys) > 0:
        invalid_key_msg = [" %s," % key for key in invalid_keys]
        msg = "Unrecognized keys in the configs: %s" % "".join(invalid_key_msg)
        raise ValueError(msg)


def build_params(base_config: Text, dannce_net: bool):
    """Build parameters dictionary from base config and io.yaml

    Args:
        base_config (Text): Path to base configuration .yaml.
        dannce_net (bool): If True, use dannce net defaults.

    Returns:
        Dict: Parameters dictionary.
    """
    base_params = processing.read_config(base_config)
    base_params = processing.make_paths_safe(base_params)
    params = processing.read_config(base_params["io_config"])
    params = processing.make_paths_safe(params)
    params = processing.inherit_config(
        params, base_params, list(base_params.keys())
    )
    check_unrecognized_params(params)
    return params


def make_folder(key: Text, params: Dict):
    """Make the prediction or training directories.

    Args:
        key (Text): Folder descriptor.
        params (Dict): Parameters dictionary.

    Raises:
        ValueError: Error if key is not defined.
    """
    # Make the prediction directory if it does not exist.
    if params[key] is not None:
        if not os.path.exists(params[key]):
            os.makedirs(params[key])
    else:
        raise ValueError(key + " must be defined.")


def com_predict(params: Dict):
    """Predict the COM for a given parameters dictionary.

    Args:
        params (Dict): Parameters dictionary.

    """
    # Make the prediction directory if it does not exist.
    make_folder("com_predict_dir", params)

    # Load the appropriate loss function and network
    try:
        params["loss"] = getattr(losses, params["loss"])
    except AttributeError:
        params["loss"] = getattr(keras_losses, params["loss"])
    params["net"] = getattr(nets, params["net"])

    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]

    # If params['n_channels_out'] is greater than one, we enter a mode in
    # which we predict all available labels + the COM
    MULTI_MODE = params["n_channels_out"] > 1 & params["n_instances"] == 1
    params["n_channels_out"] = params["n_channels_out"] + int(MULTI_MODE)

    # channels out is equal to the number of views when using a single video stream with mirrors
    eff_n_channels_out = int(params["n_views"]) if params["mirror"] else params["n_channels_out"]

    # Grab the input file for prediction
    params["label3d_file"] = processing.grab_predict_label3d_file()

    print("Using camnames: {}".format(params["camnames"]))

    # Also add parent params under the 'experiment' key for compatibility
    # with DANNCE's video loading function
    params["experiment"] = {}
    params["experiment"][0] = params

    # For real mono training
    params["chan_num"] = 1 if params["mono"] else params["n_channels_in"]

    # Build net
    print("Initializing Network...")
    model = params["net"](
        params["loss"],
        float(params["lr"]),
        params["chan_num"],
        eff_n_channels_out,
        ["mse"],
        multigpu=False,
    )

    # If the weights are not specified, use the train directory.
    if params["com_predict_weights"] is None:
        wdir = params["com_train_dir"]
        weights = os.listdir(wdir)
        weights = [f for f in weights if ".hdf5" in f]
        weights = sorted(
            weights, key=lambda x: int(x.split(".")[1].split("-")[0])
        )
        weights = weights[-1]
        params["com_predict_weights"] = os.path.join(wdir, weights)

    print("Loading weights from " + params["com_predict_weights"])
    model.load_weights(params["com_predict_weights"])

    print("COMPLETE\n")

    (
        samples,
        datadict,
        datadict_3d,
        cameras,
        camera_mats,
    ) = serve_data_DANNCE.prepare_data(
        params,
        multimode=MULTI_MODE,
        prediction=True,
        return_cammat=True,
        nanflag=False,
    )

    # Zero any negative frames
    for key in datadict.keys():
        for key_ in datadict[key]["frames"].keys():
            if datadict[key]["frames"][key_] < 0:
                datadict[key]["frames"][key_] = 0

    # The generator expects an experimentID in front of each sample key
    samples = ["0_" + str(f) for f in samples]
    datadict_ = {}
    for key in datadict.keys():
        datadict_["0_" + str(key)] = datadict[key]

    datadict = datadict_

    # Initialize video dictionary. paths to videos only.
    vids = {}
    vids = processing.initialize_vids(params, datadict, 0, vids, pathonly=True)

    # Parameters
    predict_params = {
        "dim_in": (
            params["crop_height"][1] - params["crop_height"][0],
            params["crop_width"][1] - params["crop_width"][0],
        ),
        "n_channels_in": params["n_channels_in"],
        "batch_size": 1,
        "n_channels_out": params["n_channels_out"],
        "out_scale": params["sigma"],
        "camnames": {0: params["camnames"]},
        "crop_width": params["crop_width"],
        "crop_height": params["crop_height"],
        "downsample": params["downfac"],
        "labelmode": "coord",
        "chunks": params["chunks"],
        "shuffle": False,
        "dsmode": params["dsmode"],
        "preload": False,
        "mono": params["mono"],
        "mirror": params["mirror"],
    }

    partition = {}
    partition["valid_sampleIDs"] = samples
    labels = datadict

    save_data = {}

    # If multi-instance mode is on, use the correct generator
    # and eval function.
    if params["n_instances"] > 1:
        predict_generator = DataGenerator_downsample_multi_instance(
            params["n_instances"],
            partition["valid_sampleIDs"],
            labels,
            vids,
            **predict_params
        )
    else:
        predict_generator = DataGenerator_downsample(
            partition["valid_sampleIDs"], labels, vids, **predict_params
        )

    # If we just want to analyze a chunk of video...
    if params["max_num_samples"] == "max":
        save_data = inference.infer_com(
            params["start_sample"],
            len(predict_generator),
            predict_generator,
            params,
            model,
            partition,
            save_data,
            camera_mats,
            cameras,
        )
        processing.save_COM_checkpoint(
            save_data, params["com_predict_dir"], datadict_, cameras, params
        )
    else:
        endIdx = np.min(
            [
                params["start_sample"] + params["max_num_samples"],
                len(predict_generator),
            ]
        )
        save_data = inference.infer_com(
            params["start_sample"],
            endIdx,
            predict_generator,
            params,
            model,
            partition,
            save_data,
            camera_mats,
            cameras,
        )
        processing.save_COM_checkpoint(
            save_data,
            params["com_predict_dir"],
            datadict_,
            cameras,
            params,
            file_name="com3d%d" % (params["start_sample"]),
        )

    print("done!")


def com_train(params: Dict):
    """Train COM network

    Args:
        params (Dict): Parameters dictionary.
    """
    # Make the train directory if it does not exist.
    make_folder("com_train_dir", params)

    params["loss"] = getattr(losses, params["loss"])
    params["net"] = getattr(nets, params["net"])

    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]

    # MULTI_MODE is where the full set of markers is trained on, rather than
    # the COM only. In some cases, this can help improve COMfinder performance.
    MULTI_MODE = params["n_channels_out"] > 1 & params["n_instances"] == 1
    params["n_channels_out"] = params["n_channels_out"] + int(MULTI_MODE)

    samples = []
    datadict = {}
    datadict_3d = {}
    cameras = {}
    camnames = {}

    # Use the same label files and experiment settings as DANNCE unless
    # indicated otherwise by using a 'com_exp' block in io.yaml.
    #
    # This can be useful for introducing additional COM-only label files.
    if params["com_exp"] is not None:
        exps = params["com_exp"]
    else:
        exps = params["exp"]
    num_experiments = len(exps)
    params["experiment"] = {}
    total_chunks = {}
    for e, expdict in enumerate(exps):

        exp = processing.load_expdict(params, e, expdict, _DEFAULT_VIDDIR)

        params["experiment"][e] = exp
        (
            samples_,
            datadict_,
            datadict_3d_,
            cameras_,
        ) = serve_data_DANNCE.prepare_data(
            params["experiment"][e],
            nanflag=False,
            com_flag=not MULTI_MODE,
            multimode=MULTI_MODE,
        )

        # No need to prepare any COM file (they don't exist yet).
        # We call this because we want to support multiple experiments,
        # which requires appending the experiment ID to each data object and key
        samples, datadict, datadict_3d, ddd = serve_data_DANNCE.add_experiment(
            e,
            samples,
            datadict,
            datadict_3d,
            {},
            samples_,
            datadict_,
            datadict_3d_,
            {},
        )
        cameras[e] = cameras_
        camnames[e] = params["experiment"][e]["camnames"]
        for name, chunk in exp["chunks"].items():
            total_chunks[name] = chunk

    com_train_dir = params["com_train_dir"]

    # Dump the params into file for reproducibility
    processing.save_params(com_train_dir, params)

    # Additionally, to keep videos unique across experiments, need to add
    # experiment labels in other places. E.g. experiment 0 CameraE's "camname"
    # Becomes 0_CameraE.
    cameras, datadict, params = serve_data_DANNCE.prepend_experiment(
        params, datadict, num_experiments, camnames, cameras
    )

    samples = np.array(samples)

    e = 0

    # Initialize video objects
    vids = {}
    for e in range(num_experiments):
        vids = processing.initialize_vids(
            params, datadict, e, vids, pathonly=True
        )

    print("Using {} downsampling".format(params["dsmode"]))

    train_params = {
        "dim_in": (
            params["crop_height"][1] - params["crop_height"][0],
            params["crop_width"][1] - params["crop_width"][0],
        ),
        "n_channels_in": params["n_channels_in"],
        "batch_size": 1,
        "n_channels_out": params["n_channels_out"],
        "out_scale": params["sigma"],
        "camnames": camnames,
        "crop_width": params["crop_width"],
        "crop_height": params["crop_height"],
        "downsample": params["downfac"],
        "shuffle": False,
        "chunks": total_chunks,
        "dsmode": params["dsmode"],
        "preload": False,
        "mono": params["mono"],
        "mirror": params["mirror"],
    }

    valid_params = deepcopy(train_params)
    valid_params["shuffle"] = False

    partition = processing.make_data_splits(
        samples, params, com_train_dir, num_experiments
    )

    labels = datadict

    # For real mono training
    params["chan_num"] = 1 if params["mono"] else params["n_channels_in"]

    # effective n_channels, which is different if using a mirror arena configuration
    eff_n_channels_out = len(camnames[0]) if params["mirror"] else params["n_channels_out"]

    # Build net
    print("Initializing Network...")

    model = params["net"](
        params["loss"],
        float(params["lr"]),
        params["chan_num"],
        eff_n_channels_out,
        ["mse"],
        multigpu=False,
    )
    print("COMPLETE\n")

    if params["com_finetune_weights"] is not None:
        weights = os.listdir(params["com_finetune_weights"])
        weights = [f for f in weights if ".hdf5" in f]
        weights = weights[0]

        try:
            model.load_weights(
                os.path.join(params["com_finetune_weights"], weights)
            )
        except:
            print(
                "Note: model weights could not be loaded due to a mismatch in dimensions.\
                   Assuming that this is a fine-tune with a different number of outputs and removing \
                  the top of the net accordingly"
            )
            model.layers[-1].name = "top_conv"
            model.load_weights(
                os.path.join(params["com_finetune_weights"], weights),
                by_name=True,
            )

    if params["lockfirst"]:
        for layer in model.layers[:2]:
            layer.trainable = False

    model.compile(
        optimizer=Adam(lr=float(params["lr"])),
        loss=params["loss"],
    )

    # Create checkpoint and logging callbacks
    kkey = "weights.hdf5"
    mon = "val_loss" if params["num_validation_per_exp"] > 0 else "loss"

    # Create checkpoint and logging callbacks
    model_checkpoint = ModelCheckpoint(
        os.path.join(com_train_dir, kkey),
        monitor=mon,
        save_best_only=True,
        save_weights_only=True,
    )
    csvlog = CSVLogger(os.path.join(com_train_dir, "training.csv"))
    tboard = TensorBoard(
        log_dir=os.path.join(com_train_dir, "logs"), write_graph=False, update_freq=100
    )

    # Initialize data structures
    if params["mirror"]:
        ncams = 1 # Effectively, for the purpose of batch indexing
    else:
        ncams = len(camnames[0])

    dh = (params["crop_height"][1] - params["crop_height"][0]) // params["downfac"]
    dw = (params["crop_width"][1] - params["crop_width"][0]) // params["downfac"]


    ims_train = np.zeros(
        (
            ncams * len(partition["train_sampleIDs"]),
            dh,
            dw,
            params["chan_num"],
        ),
        dtype="float32",
    )
    y_train = np.zeros(
        (ncams * len(partition["train_sampleIDs"]), dh, dw, eff_n_channels_out),
        dtype="float32",
    )
    ims_valid = np.zeros(
        (
            ncams * len(partition["valid_sampleIDs"]),
            dh,
            dw,
            params["chan_num"],
        ),
        dtype="float32",
    )
    y_valid = np.zeros(
        (ncams * len(partition["valid_sampleIDs"]), dh, dw, eff_n_channels_out),
        dtype="float32",
    )

    # Set up generators
    if params["n_instances"] > 1:
        train_generator = DataGenerator_downsample_multi_instance(
            params["n_instances"],
            partition["train_sampleIDs"],
            labels,
            vids,
            **train_params
        )
        valid_generator = DataGenerator_downsample_multi_instance(
            params["n_instances"],
            partition["valid_sampleIDs"],
            labels,
            vids,
            **valid_params
        )
    else:
        train_generator = DataGenerator_downsample(
            partition["train_sampleIDs"], labels, vids, **train_params
        )
        valid_generator = DataGenerator_downsample(
            partition["valid_sampleIDs"], labels, vids, **valid_params
        )

    print("Loading data")
    for i in range(len(partition["train_sampleIDs"])):
        print(i, end="\r")
        ims = train_generator.__getitem__(i)
        ims_train[i * ncams : (i + 1) * ncams] = ims[0]
        y_train[i * ncams : (i + 1) * ncams] = ims[1]

    for i in range(len(partition["valid_sampleIDs"])):
        print(i, end="\r")
        ims = valid_generator.__getitem__(i)
        ims_valid[i * ncams : (i + 1) * ncams] = ims[0]
        y_valid[i * ncams : (i + 1) * ncams] = ims[1]

    train_generator = DataGenerator_downsample_frommem(
        np.arange(ims_train.shape[0]),
        ims_train,
        y_train,
        batch_size=params["batch_size"] * ncams,
        augment_hue=params["augment_hue"],
        augment_brightness=params["augment_brightness"],
        augment_rotation=params["augment_rotation"],
        augment_shear=params["augment_hue"],
        augment_shift=params["augment_brightness"],
        augment_zoom=params["augment_rotation"],
        bright_val=params["augment_bright_val"],
        hue_val=params["augment_hue_val"],
        shift_val=params["augment_shift_val"],
        rotation_val=params["augment_rotation_val"],
        shear_val=params["augment_shear_val"],
        zoom_val=params["augment_zoom_val"],
        chan_num=params["chan_num"],
    )
    valid_generator = DataGenerator_downsample_frommem(
        np.arange(ims_valid.shape[0]),
        ims_valid,
        y_valid,
        batch_size=ncams,
        shuffle=False,
        chan_num=params["chan_num"],
    )

    def write_debug(trainData=True):
        """Factoring re-used debug output code.

        Writes training or validation images to an output directory, together
        with the ground truth COM labels and predicted COM labels, respectively.

        Args:
            trainData (bool, optional): If True use training data for debug.
        """
        def plot_out(imo, lo, imn):
            processing.plot_markers_2d(
                processing.norm_im(imo), lo, newfig=False
            )
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            imname = imn
            plt.savefig(
                os.path.join(debugdir, imname), bbox_inches="tight", pad_inches=0
            )

        if params["debug"] and not MULTI_MODE:

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
                        plot_out(ims_out[i], label_out[i, :, :, j:j+1],
                                 str(i) + "_cam_" + str(j) + ".png")
                else:
                    plot_out(ims_out[i], label_out[i], str(i) + ".png")

        elif params["debug"] and MULTI_MODE:
            print("Note: Cannot output debug information in COM multi-mode")

    write_debug(trainData=True)

    model.fit(
        x=train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        verbose=params["verbose"],
        epochs=params["epochs"],
        workers=6,
        callbacks=[csvlog, model_checkpoint, tboard],
    )

    write_debug(trainData=False)

    print("Renaming weights file with best epoch description")
    processing.rename_weights(com_train_dir, kkey, mon)

    print("Saving full model at end of training")
    sdir = os.path.join(params["com_train_dir"], "fullmodel_weights")
    if not os.path.exists(sdir):
        os.makedirs(sdir)
    model.save(os.path.join(sdir, "fullmodel_end.hdf5"))


def dannce_train(params: Dict):
    """Train dannce network.

    Args:
        params (Dict): Parameters dictionary.

    Raises:
        Exception: Error if training mode is invalid.
    """
    # Depth disabled until next release.
    params["depth"] = False

    # Make the training directory if it does not exist.
    make_folder("dannce_train_dir", params)

    params["loss"] = getattr(losses, params["loss"])
    params["net"] = getattr(nets, params["net"])

    # Default to 6 views but a smaller number of views can be specified in the
    # DANNCE config. If the legnth of the camera files list is smaller than
    # n_views, relevant lists will be duplicated in order to match n_views, if
    # possible.
    n_views = int(params["n_views"])

    # Convert all metric strings to objects
    metrics = []
    for m in params["metric"]:
        try:
            m_obj = getattr(losses, m)
        except AttributeError:
            m_obj = getattr(keras.losses, m)
        metrics.append(m_obj)

    # set GPU ID
    if not params["multi_gpu_train"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]

    # find the weights given config path
    if params["dannce_finetune_weights"] is not None:
        params["dannce_finetune_weights"] = processing.get_ft_wt(params)
        print("Fine-tuning from {}".format(params["dannce_finetune_weights"]))

    samples = []
    datadict = {}
    datadict_3d = {}
    com3d_dict = {}
    cameras = {}
    camnames = {}
    exps = params["exp"]
    num_experiments = len(exps)
    params["experiment"] = {}
    total_chunks = {}

    for e, expdict in enumerate(exps):

        exp = processing.load_expdict(params, e, expdict, _DEFAULT_VIDDIR)

        (
            exp,
            samples_,
            datadict_,
            datadict_3d_,
            cameras_,
            com3d_dict_,
        ) = do_COM_load(exp, expdict, n_views, e, params)

        print("Using {} samples total.".format(len(samples_)))

        (
            samples,
            datadict,
            datadict_3d,
            com3d_dict,
        ) = serve_data_DANNCE.add_experiment(
            e,
            samples,
            datadict,
            datadict_3d,
            com3d_dict,
            samples_,
            datadict_,
            datadict_3d_,
            com3d_dict_,
        )

        cameras[e] = cameras_
        camnames[e] = exp["camnames"]
        print("Using the following cameras: {}".format(camnames[e]))
        params["experiment"][e] = exp
        for name, chunk in exp["chunks"].items():
            total_chunks[name] = chunk

    dannce_train_dir = params["dannce_train_dir"]

    # Dump the params into file for reproducibility
    processing.save_params(dannce_train_dir, params)

    # Additionally, to keep videos unique across experiments, need to add
    # experiment labels in other places. E.g. experiment 0 CameraE's "camname"
    # Becomes 0_CameraE. *NOTE* This function modified camnames in place
    # to add the appropriate experiment ID
    cameras, datadict, params = serve_data_DANNCE.prepend_experiment(
        params, datadict, num_experiments, camnames, cameras
    )

    samples = np.array(samples)

    if params["use_npy"]:
        # Add all npy volume directories to list, to be used by generator
        npydir = {}
        for e in range(num_experiments):
            npydir[e] = params["experiment"][e]["npy_vol_dir"]

        samples = processing.remove_samples_npy(npydir, samples, params)
    else:
        # Initialize video objects
        vids = {}
        for e in range(num_experiments):
            if params["immode"] == "vid":
                vids = processing.initialize_vids(
                    params, datadict, e, vids, pathonly=True
                     )

    # Parameters
    if params["expval"]:
        outmode = "coordinates"
    else:
        outmode = "3dprob"

    gridsize = tuple([params["nvox"]] * 3)

    # When this true, the data generator will shuffle the cameras and then select the first 3,
    # to feed to a native 3 camera model
    cam3_train = params["cam3_train"]
    if params["cam3_train"]:
        cam3_train = True
    else:
        cam3_train = False

    partition = processing.make_data_splits(
            samples, params, dannce_train_dir, num_experiments
        )

    if params["use_npy"]:
        # mono conversion will happen from RGB npy files, and the generator
        # needs to b aware that the npy files contain RGB content
        params["chan_num"] = params["n_channels_in"]
    else:
        # Used to initialize arrays for mono, and also in *frommem (the final generator)
        params["chan_num"] = 1 if params["mono"] else params["n_channels_in"]

        valid_params = {
            "dim_in": (
                params["crop_height"][1] - params["crop_height"][0],
                params["crop_width"][1] - params["crop_width"][0],
            ),
            "n_channels_in": params["n_channels_in"],
            "batch_size": 1,
            "n_channels_out": params["new_n_channels_out"],
            "out_scale": params["sigma"],
            "crop_width": params["crop_width"],
            "crop_height": params["crop_height"],
            "vmin": params["vmin"],
            "vmax": params["vmax"],
            "nvox": params["nvox"],
            "interp": params["interp"],
            "depth": params["depth"],
            "channel_combo": params["channel_combo"],
            "mode": outmode,
            "camnames": camnames,
            "immode": params["immode"],
            "shuffle": False,  # We will shuffle later
            "rotation": False,  # We will rotate later if desired
            "vidreaders": vids,
            "distort": True,
            "expval": params["expval"],
            "crop_im": False,
            "chunks": total_chunks,
            "preload": False,
            "mono": params["mono"],
            "mirror": params["mirror"],
        }

        # Setup a generator that will read videos and labels
        tifdirs = []  # Training from single images not yet supported in this demo

        train_generator = DataGenerator_3Dconv(
            partition["train_sampleIDs"],
            datadict,
            datadict_3d,
            cameras,
            partition["train_sampleIDs"],
            com3d_dict,
            tifdirs,
            **valid_params
        )
        valid_generator = DataGenerator_3Dconv(
            partition["valid_sampleIDs"],
            datadict,
            datadict_3d,
            cameras,
            partition["valid_sampleIDs"],
            com3d_dict,
            tifdirs,
            **valid_params
        )

        # We should be able to load everything into memory...
        gridsize = tuple([params["nvox"]] * 3)
        X_train = np.zeros(
            (
                len(partition["train_sampleIDs"]),
                *gridsize,
                params["chan_num"] * len(camnames[0]),
            ),
            dtype="float32",
        )

        X_valid = np.zeros(
            (
                len(partition["valid_sampleIDs"]),
                *gridsize,
                params["chan_num"] * len(camnames[0]),
            ),
            dtype="float32",
        )

        X_train_grid = None
        X_valid_grid = None
        if params["expval"]:
            y_train = np.zeros(
                (
                    len(partition["train_sampleIDs"]),
                    3,
                    params["new_n_channels_out"],
                ),
                dtype="float32",
            )
            X_train_grid = np.zeros(
                (len(partition["train_sampleIDs"]), params["nvox"] ** 3, 3),
                dtype="float32",
            )

            y_valid = np.zeros(
                (
                    len(partition["valid_sampleIDs"]),
                    3,
                    params["new_n_channels_out"],
                ),
                dtype="float32",
            )
            X_valid_grid = np.zeros(
                (len(partition["valid_sampleIDs"]), params["nvox"] ** 3, 3),
                dtype="float32",
            )
        else:
            y_train = np.zeros(
                (
                    len(partition["train_sampleIDs"]),
                    *gridsize,
                    params["new_n_channels_out"],
                ),
                dtype="float32",
            )

            y_valid = np.zeros(
                (
                    len(partition["valid_sampleIDs"]),
                    *gridsize,
                    params["new_n_channels_out"],
                ),
                dtype="float32",
            )

        print(
            "Loading training data into memory. This can take a while to seek through",
            "large sets of video. This process is much faster if the frame indices",
            "are sorted in ascending order in your label data file.",
        )
        for i in range(len(partition["train_sampleIDs"])):
            print(i, end="\r")
            rr = train_generator.__getitem__(i)
            if params["expval"]:
                X_train[i] = rr[0][0]
                X_train_grid[i] = rr[0][1]
            else:
                X_train[i] = rr[0]
            y_train[i] = rr[1]

        if params["debug_volume_tifdir"] is not None:
            # When this option is toggled in the config, rather than
            # training, the image volumes are dumped to tif stacks.
            # This can be used for debugging problems with calibration or
            # COM estimation
            tifdir = params["debug_volume_tifdir"]
            print("Dump training volumes to {}".format(tifdir))
            for i in range(X_train.shape[0]):
                for j in range(len(camnames[0])):
                    im = X_train[
                        i,
                        :,
                        :,
                        :,
                        j * params["chan_num"] : (j + 1) * params["chan_num"],
                    ]
                    im = processing.norm_im(im) * 255
                    im = im.astype("uint8")
                    of = os.path.join(
                        tifdir,
                        partition["train_sampleIDs"][i] + "_cam" + str(j) + ".tif",
                    )
                    imageio.mimwrite(of, np.transpose(im, [2, 0, 1, 3]))
            print("Done! Exiting.")
            sys.exit()

        print("Loading validation data into memory")
        for i in range(len(partition["valid_sampleIDs"])):
            print(i, end="\r")
            rr = valid_generator.__getitem__(i)
            if params["expval"]:
                X_valid[i] = rr[0][0]
                X_valid_grid[i] = rr[0][1]
            else:
                X_valid[i] = rr[0]
            y_valid[i] = rr[1]

    # Now we can generate from memory with shuffling, rotation, etc.
    randflag = params["channel_combo"] == "random"

    if cam3_train:
        params["n_rand_views"] = 3
        params["rand_view_replace"] = False
        randflag = True

    shared_args = {'chan_num': params["chan_num"],
                   'expval': params["expval"],
                   'nvox': params["nvox"]}
    shared_args_train = {'batch_size': params["batch_size"],
                         'rotation': params["rotate"],
                         'augment_hue': params["augment_hue"],
                         'augment_brightness': params["augment_brightness"],
                         'augment_continuous_rotation': params["augment_continuous_rotation"],
                         'bright_val': params["augment_bright_val"],
                         'hue_val': params["augment_hue_val"],
                         'rotation_val': params["augment_rotation_val"],
                         'replace': params["rand_view_replace"],
                         'random': randflag,
                         'n_rand_views': params["n_rand_views"],
                         }
    shared_args_valid = {'batch_size': 4,
                         'rotation': False,
                         'augment_hue': False,
                         'augment_brightness': False,
                         'augment_continuous_rotation': False,
                         'shuffle': False,
                         'replace': False,
                         'n_rand_views': params["n_rand_views"] if cam3_train else None,
                         'random': True if cam3_train else False}
    if params["use_npy"]:
        genfunc = DataGenerator_3Dconv_npy
        args_train = {'list_IDs': partition["train_sampleIDs"],
                      'labels_3d': datadict_3d,
                      'npydir': npydir,
                      }
        args_train = {**args_train,
                      **shared_args_train,
                      **shared_args,
                      'sigma': params["sigma"],
                      'mono': params["mono"]}

        args_valid = {'list_IDs': partition["valid_sampleIDs"],
                      'labels_3d': datadict_3d,
                      'npydir': npydir,
                      }
        args_valid = {**args_valid,
                      **shared_args_valid,
                      **shared_args,
                      'sigma': params["sigma"],
                      'mono': params["mono"]}
    else:
        genfunc = DataGenerator_3Dconv_frommem
        args_train = {'list_IDs': np.arange(len(partition["train_sampleIDs"])),
                      'data': X_train,
                      'labels': y_train,
                      }
        args_train = {**args_train,
                      **shared_args_train,
                      **shared_args,
                      'xgrid': X_train_grid}

        args_valid = {'list_IDs': np.arange(len(partition["valid_sampleIDs"])),
                      'data': X_valid,
                      'labels': y_valid,
                      }
        args_valid = {**args_valid,
                      **shared_args_valid,
                      **shared_args,
                      'xgrid': X_valid_grid}

    train_generator = genfunc(**args_train)
    valid_generator = genfunc(**args_valid)

    # Build net
    print("Initializing Network...")

    # Currently, we expect four modes of use:
    # 1) Training a new network from scratch
    # 2) Fine-tuning a network trained on a diff. dataset (transfer learning)
    # 3) Continuing to train 1) or 2) from a full model checkpoint (including optimizer state)

    # if params["multi_gpu_train"]:
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    scoping = strategy.scope()
    # else:
    #     scoping = True

    print("NUM CAMERAS: {}".format(len(camnames[0])))

    with scoping:
        if params["train_mode"] == "new":
            model = params["net"](
                params["loss"],
                float(params["lr"]),
                params["chan_num"] + params["depth"],
                params["n_channels_out"],
                len(camnames[0]),
                batch_norm=False,
                instance_norm=True,
                include_top=True,
                gridsize=gridsize,
            )
        elif params["train_mode"] == "finetune":
            model = params["net"](
                params["loss"],
                float(params["lr"]),
                params["chan_num"] + params["depth"],
                params["n_channels_out"],
                len(camnames[0]),
                params["new_last_kernel_size"],
                params["new_n_channels_out"],
                params["dannce_finetune_weights"],
                params["n_layers_locked"],
                batch_norm=False,
                instance_norm=True,
                gridsize=gridsize,
            )
        elif params["train_mode"] == "continued":
            model = load_model(
                params["dannce_finetune_weights"],
                custom_objects={
                    "ops": ops,
                    "slice_input": nets.slice_input,
                    "mask_nan_keep_loss": losses.mask_nan_keep_loss,
                    "euclidean_distance_3D": losses.euclidean_distance_3D,
                    "centered_euclidean_distance_3D": losses.centered_euclidean_distance_3D,
                },
            )
        elif params["train_mode"] == "continued_weights_only":
            # This does not work with models created in 'finetune' mode, but will work with models
            # started from scratch ('new' train_mode)
            model = params["net"](
                params["loss"],
                float(params["lr"]),
                params["chan_num"] + params["depth"],
                params["n_channels_out"],
                3 if cam3_train else len(camnames[0]),
                batch_norm=False,
                instance_norm=True,
                include_top=True,
                gridsize=gridsize,
            )
            model.load_weights(params["dannce_finetune_weights"])
        else:
            raise Exception("Invalid training mode")

        model.compile(
            optimizer=Adam(lr=float(params["lr"])),
            loss=params["loss"],
            metrics=metrics,
        )

    print("COMPLETE\n")

    # Create checkpoint and logging callbacks
    kkey = "weights.hdf5"
    mon = "val_loss" if params["num_validation_per_exp"] > 0 else "loss"

    model_checkpoint = ModelCheckpoint(
        os.path.join(dannce_train_dir, kkey),
        monitor=mon,
        save_best_only=True,
        save_weights_only=False,
    )
    csvlog = CSVLogger(os.path.join(dannce_train_dir, "training.csv"))
    tboard = TensorBoard(
        log_dir=os.path.join(dannce_train_dir,"logs"), write_graph=False, update_freq=100
    )

    class savePredTargets(keras.callbacks.Callback):
        def __init__(self, total_epochs, td, tgrid, vd, vgrid, tID, vID, odir, tlabel, vlabel):
            self.td = td
            self.vd = vd
            self.tID = tID
            self.vID = vID
            self.total_epochs = total_epochs
            self.val_loss = 1e10
            self.odir = odir
            self.tgrid = tgrid
            self.vgrid = vgrid
            self.tlabel = tlabel
            self.vlabel = vlabel
        def on_epoch_end(self, epoch, logs=None):
            lkey = 'val_loss' if 'val_loss' in logs else 'loss'
            if epoch == self.total_epochs-1 or logs[lkey] < self.val_loss and epoch > 25:
                print("Saving predictions on train and validation data, after epoch {}".format(epoch))
                self.val_loss = logs[lkey]
                pred_t = model.predict([self.td, self.tgrid], batch_size=1)
                pred_v = model.predict([self.vd, self.vgrid], batch_size=1)
                ofile = os.path.join(self.odir,'checkpoint_predictions_e{}.mat'.format(epoch))
                sio.savemat(ofile, {'pred_train': pred_t,
                                    'pred_valid': pred_v,
                                    'target_train': self.tlabel,
                                    'target_valid': self.vlabel,
                                    'train_sampleIDs': self.tID,
                                    'valid_sampleIDs': self.vID})

    class saveCheckPoint(keras.callbacks.Callback):
        def __init__(self, odir, total_epochs):
            self.odir = odir
            self.saveE = np.arange(0, total_epochs, 250)
        def on_epoch_end(self, epoch, logs=None):
            lkey = 'val_loss' if 'val_loss' in logs else 'loss'
            val_loss = logs[lkey]
            if epoch in self.saveE:
                # Do a garbage collect to combat keras memory leak
                gc.collect()
                print("Saving checkpoint weights at epoch {}".format(epoch))
                savename = 'weights.checkpoint.epoch{}.{}{:.5f}.hdf5'.format(epoch,
                                                                        lkey,
                                                                        val_loss)
                self.model.save(os.path.join(self.odir, savename))

                
    callbacks = [csvlog, model_checkpoint, tboard, saveCheckPoint(params['dannce_train_dir'], params["epochs"])]

    if params['expval'] and not params["use_npy"]:
        save_callback = savePredTargets(params['epochs'],X_train,
            X_train_grid,
            X_valid,
            X_valid_grid,
            partition['train_sampleIDs'],
            partition['valid_sampleIDs'],
            params['dannce_train_dir'],
            y_train,
            y_valid)
        callbacks = callbacks + [save_callback]

    model.fit(
        x=train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        verbose=params["verbose"],
        epochs=params["epochs"],
        callbacks=callbacks,
        workers=6,
    )

    print("Renaming weights file with best epoch description")
    processing.rename_weights(dannce_train_dir, kkey, mon)

    print("Saving full model at end of training")
    sdir = os.path.join(params["dannce_train_dir"], "fullmodel_weights")
    if not os.path.exists(sdir):
        os.makedirs(sdir)
    model.save(os.path.join(sdir, "fullmodel_end.hdf5"))

    print("done!")


def dannce_predict(params: Dict):
    """Predict with dannce network

    Args:
        params (Dict): Paremeters dictionary.
    """
    # Depth disabled until next release.
    params["depth"] = False
    # Make the prediction directory if it does not exist.
    make_folder("dannce_predict_dir", params)

    # Load the appropriate loss function and network
    try:
        params["loss"] = getattr(losses, params["loss"])
    except AttributeError:
        params["loss"] = getattr(keras_losses, params["loss"])
    netname = params["net"]
    params["net"] = getattr(nets, params["net"])
    # Default to 6 views but a smaller number of views can be specified in the DANNCE config.
    # If the legnth of the camera files list is smaller than n_views, relevant lists will be
    # duplicated in order to match n_views, if possible.
    n_views = int(params["n_views"])

    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]
    gpu_id = params["gpu_id"]

    # While we can use experiment files for DANNCE training,
    # for prediction we use the base data files present in the main config
    # Grab the input file for prediction
    params["label3d_file"] = processing.grab_predict_label3d_file()
    params["base_exp_folder"] = os.path.dirname(params["label3d_file"])

    # default to slow numpy backend if there is no predict_mode in config file. I.e. legacy support
    predict_mode = (
        params["predict_mode"]
        if params["predict_mode"] is not None
        else "numpy"
    )
    print("Using {} predict mode".format(predict_mode))

    print("Using camnames: {}".format(params["camnames"]))
    # Also add parent params under the 'experiment' key for compatibility
    # with DANNCE's video loading function
    params["experiment"] = {}
    params["experiment"][0] = params

    (
        params["experiment"][0],
        samples_,
        datadict_,
        datadict_3d_,
        cameras_,
        com3d_dict_,
    ) = do_COM_load(
        params["experiment"][0],
        params["experiment"][0],
        n_views,
        0,
        params,
        training=False,
    )

    # Write 3D COM to file. This might be different from the input com3d file
    # if arena thresholding was applied.
    cfilename = os.path.join(params["dannce_predict_dir"], "com3d_used.mat")
    print("Saving 3D COM to {}".format(cfilename))
    c3d = np.zeros((len(samples_), 3))
    for i in range(len(samples_)):
        c3d[i] = com3d_dict_[samples_[i]]
    sio.savemat(cfilename, {"sampleID": samples_, "com": c3d})

    # The library is configured to be able to train over multiple animals ("experiments")
    # at once. Because supporting code expects to see an experiment ID# prepended to
    # each of these data keys, we need to add a token experiment ID here.
    samples = []
    datadict = {}
    datadict_3d = {}
    com3d_dict = {}
    (
        samples,
        datadict,
        datadict_3d,
        com3d_dict,
    ) = serve_data_DANNCE.add_experiment(
        0,
        samples,
        datadict,
        datadict_3d,
        com3d_dict,
        samples_,
        datadict_,
        datadict_3d_,
        com3d_dict_,
    )
    cameras = {}
    cameras[0] = cameras_
    camnames = {}
    camnames[0] = params["experiment"][0]["camnames"]

    # Need a '0' experiment ID to work with processing functions.
    # *NOTE* This function modified camnames in place
    # to add the appropriate experiment ID
    cameras, datadict, params = serve_data_DANNCE.prepend_experiment(
        params, datadict, 1, camnames, cameras, dannce_prediction=True
    )

    samples = np.array(samples)

    # For real mono prediction
    params["chan_num"] = 1 if params["mono"] else params["n_channels_in"]

    # Initialize video dictionary. paths to videos only.
    # TODO: Remove this immode option if we decide not
    # to support tifs
    if params["immode"] == "vid":
        vids = {}
        vids = processing.initialize_vids(
            params, datadict, 0, vids, pathonly=True
        )

    # Parameters
    valid_params = {
        "dim_in": (
            params["crop_height"][1] - params["crop_height"][0],
            params["crop_width"][1] - params["crop_width"][0],
        ),
        "n_channels_in": params["n_channels_in"],
        "batch_size": params["batch_size"],
        "n_channels_out": params["n_channels_out"],
        "out_scale": params["sigma"],
        "crop_width": params["crop_width"],
        "crop_height": params["crop_height"],
        "vmin": params["vmin"],
        "vmax": params["vmax"],
        "nvox": params["nvox"],
        "interp": params["interp"],
        "depth": params["depth"],
        "channel_combo": params["channel_combo"],
        "mode": "coordinates",
        "camnames": camnames,
        "immode": params["immode"],
        "shuffle": False,
        "rotation": False,
        "vidreaders": vids,
        "distort": True,
        "expval": params["expval"],
        "crop_im": False,
        "chunks": params["chunks"],
        "preload": False,
        "mono": params["mono"],
        "mirror": params["mirror"],
        "predict_flag": True,
    }

    # Datasets
    partition = {}
    valid_inds = np.arange(len(samples))
    partition["valid_sampleIDs"] = samples[valid_inds]

    # TODO: Remove tifdirs arguments, which are deprecated
    tifdirs = []

    # Generators
    if predict_mode == "torch":
        import torch

        # Because CUDA_VISBILE_DEVICES is already set to a single GPU, the gpu_id here should be "0"
        device = "cuda:0"
        genfunc = DataGenerator_3Dconv_torch
    elif predict_mode == "tf":
        device = "/GPU:0"
        genfunc = DataGenerator_3Dconv_tf
    else:
        genfunc = DataGenerator_3Dconv

    valid_generator = genfunc(
        partition["valid_sampleIDs"],
        datadict,
        datadict_3d,
        cameras,
        partition["valid_sampleIDs"],
        com3d_dict,
        tifdirs,
        **valid_params
    )

    # Build net
    print("Initializing Network...")

    # This requires that the network be saved as a full model, not just weights.
    # As a precaution, we import all possible custom objects that could be used
    # by a model and thus need declarations

    if params["dannce_predict_model"] is not None:
        mdl_file = params["dannce_predict_model"]
    else:
        wdir = params["dannce_train_dir"]
        weights = os.listdir(wdir)
        weights = [f for f in weights if ".hdf5" in f]
        weights = sorted(
            weights, key=lambda x: int(x.split(".")[1].split("-")[0])
        )
        weights = weights[-1]

        mdl_file = os.path.join(wdir, weights)
        # if not using dannce_predict model (thus taking the final weights in train_results),
        # set this file to dannce_predict_model so that it will still get saved with metadata
        params["dannce_predict_model"] = mdl_file

    print("Loading model from " + mdl_file)

    if (
        netname == "unet3d_big_tiedfirstlayer_expectedvalue"
        or params["from_weights"] is not None
    ):
        gridsize = tuple([params["nvox"]] * 3)
        params["dannce_finetune_weights"] = processing.get_ft_wt(params)

        if params["train_mode"] == "finetune":

            print("Initializing a finetune network from {}, into which weights from {} will be loaded.".format(
                params["dannce_finetune_weights"], mdl_file))
            model = params["net"](
                params["loss"],
                float(params["lr"]),
                params["chan_num"] + params["depth"],
                params["n_channels_out"],
                len(camnames[0]),
                params["new_last_kernel_size"],
                params["new_n_channels_out"],
                params["dannce_finetune_weights"],
                params["n_layers_locked"],
                batch_norm=False,
                instance_norm=True,
                gridsize=gridsize,
            )
        else:
            # This network is too "custom" to be loaded in as a full model, until I
            # figure out how to unroll the first tied weights layer
            model = params["net"](
                params["loss"],
                float(params["lr"]),
                params["chan_num"] + params["depth"],
                params["n_channels_out"],
                len(camnames[0]),
                batch_norm=False,
                instance_norm=True,
                include_top=True,
                gridsize=gridsize,
            )
        model.load_weights(mdl_file)
    else:
        model = load_model(
            mdl_file,
            custom_objects={
                "ops": ops,
                "slice_input": nets.slice_input,
                "mask_nan_keep_loss": losses.mask_nan_keep_loss,
                "euclidean_distance_3D": losses.euclidean_distance_3D,
                "centered_euclidean_distance_3D": losses.centered_euclidean_distance_3D,
            },
        )

    # To speed up expval prediction, rather than doing two forward passes: one for the 3d coordinate
    # and one for the probability map, here we splice on a new output layer after
    # the softmax on the last convolutional layer
    if params["expval"]:
        from tensorflow.keras.layers import GlobalMaxPooling3D

        o2 = GlobalMaxPooling3D()(model.layers[-3].output)
        model = Model(
            inputs=[model.layers[0].input, model.layers[-2].input],
            outputs=[model.layers[-1].output, o2],
        )

    save_data = {}

    max_eval_batch = params["maxbatch"]

    if max_eval_batch != "max" and max_eval_batch > len(valid_generator):
        print("Maxbatch was set to a larger number of matches than exist in the video. Truncating")
        max_eval_batch = len(valid_generator)
        processing.print_and_set(params, "maxbatch", max_eval_batch)

    if max_eval_batch == "max":
        max_eval_batch = len(valid_generator)

    if params["start_batch"] is not None:
        start_batch = params["start_batch"]
    else:
        start_batch = 0

    if params["new_n_channels_out"] is not None:
        n_chn = params["new_n_channels_out"]
    else:
        n_chn = params["n_channels_out"]

    if params["write_npy"] is not None:
        # Instead of running inference, generate all samples
        # from valid_generator and save them to npy files. Useful
        # for working with large datasets (such as Rat 7M) because
        # .npy files can be loaded in quickly with random access
        # during training.
        print("Writing samples to .npy files")
        processing.write_npy(params["write_npy"], valid_generator)
        print("Done, exiting program")
        sys.exit()


    save_data = inference.infer_dannce(
        start_batch,
        max_eval_batch,
        valid_generator,
        params,
        model,
        partition,
        save_data,
        device,
        n_chn,
    )

    if params["expval"]:
        if params["start_batch"] is not None:
            path = os.path.join(
                params["dannce_predict_dir"], "save_data_AVG%d.mat" % (start_batch)
            )
        else:
            path = os.path.join(params["dannce_predict_dir"], "save_data_AVG.mat")
        p_n = savedata_expval(
            path,
            params,
            write=True,
            data=save_data,
            tcoord=False,
            num_markers=n_chn,
            pmax=True,
        )
    else:
        if params["start_batch"] is not None:
            path = os.path.join(
                params["dannce_predict_dir"], "save_data_MAX%d.mat" % (start_batch)
            )
        else:
            path = os.path.join(params["dannce_predict_dir"], "save_data_MAX.mat")
        p_n = savedata_tomat(
            path,
            params,
            params["vmin"],
            params["vmax"],
            params["nvox"],
            write=True,
            data=save_data,
            num_markers=n_chn,
            tcoord=False,
        )


def do_COM_load(
    exp: Dict, expdict: Dict, n_views: int, e, params: Dict, training=True
):
    """Load and process COMs.

    Args:
        exp (Dict): Parameters dictionary for experiment
        expdict (Dict): Experiment specific overrides (e.g. com_file, vid_dir)
        n_views (int): Number of views
        e (TYPE): Description
        params (Dict): Parameters dictionary.
        training (bool, optional): If true, load COM for training frames.

    Returns:
        TYPE: Description
        exp, samples_, datadict_, datadict_3d_, cameras_, com3d_dict_

    Raises:
        Exception: Exception when invalid com file format.
    """
    (
        samples_,
        datadict_,
        datadict_3d_,
        cameras_,
    ) = serve_data_DANNCE.prepare_data(
        exp, prediction=False if training else True, nanflag=False
    )

    # If len(exp['camnames']) divides evenly into n_views, duplicate here
    # This must come after loading in this experiment's data because there
    # is an assertion that len(exp['camnames']) == the number of cameras
    # in the label files (which will not be duplicated)
    exp = processing.dupe_params(exp, ["camnames"], n_views)

    # If there is "clean" data (full marker set), can take the
    # 3D COM from the labels
    if exp["com_fromlabels"] and training:
        print("For experiment {}, calculating 3D COM from labels".format(e))
        com3d_dict_ = deepcopy(datadict_3d_)
        for key in com3d_dict_.keys():
            com3d_dict_[key] = np.nanmean(
                datadict_3d_[key], axis=1, keepdims=True
            )
    elif "com_file" in expdict and expdict["com_file"] is not None:
        exp["com_file"] = expdict["com_file"]
        if ".mat" in exp["com_file"]:
            c3dfile = sio.loadmat(exp["com_file"])
            com3d_dict_ = check_COM_load(
                c3dfile, "com", datadict_, params["medfilt_window"]
            )
        elif ".pickle" in exp["com_file"]:
            datadict_, com3d_dict_ = serve_data_DANNCE.prepare_COM(
                exp["com_file"],
                datadict_,
                comthresh=params["comthresh"],
                weighted=params["weighted"],
                camera_mats=cameras_,
                method=params["com_method"],
            )
            if params["medfilt_window"] is not None:
                raise Exception(
                    "Sorry, median filtering a com pickle is not yet supported. Please use a com3d.mat or *dannce.mat file instead"
                )
        else:
            raise Exception("Not a valid com file format")
    else:
        # Then load COM from the label3d file
        exp["com_file"] = expdict["label3d_file"]
        c3dfile = io.load_com(exp["com_file"])
        com3d_dict_ = check_COM_load(
            c3dfile, "com3d", datadict_, params["medfilt_window"]
        )

    print("Experiment {} using com3d: {}".format(e, exp["com_file"]))

    if params["medfilt_window"] is not None:
        print(
            "Median filtering COM trace with window size {}".format(
                params["medfilt_window"]
            )
        )

    # Remove any 3D COMs that are beyond the confines off the 3D arena
    do_cthresh = True if exp["cthresh"] is not None else False

    pre = len(samples_)
    samples_ = serve_data_DANNCE.remove_samples_com(
        samples_,
        com3d_dict_,
        rmc=do_cthresh,
        cthresh=exp["cthresh"],
    )
    msg = "Removed {} samples from the dataset because they either had COM positions over cthresh, or did not have matching sampleIDs in the COM file"
    print(msg.format(pre - len(samples_)))

    return exp, samples_, datadict_, datadict_3d_, cameras_, com3d_dict_


def check_COM_load(c3dfile: Dict, kkey: Text, datadict_: Dict, wsize: int):
    """Check that the COM file is of the appropriate format, and filter it.

    Args:
        c3dfile (Dict): Loaded com3d dictionary.
        kkey (Text): Key to use for extracting com.
        datadict_ (Dict): Dictionary containing data.
        wsize (int): Window size.

    Returns:
        Dict: Dictionary containing com data.
    """
    c3d = c3dfile[kkey]

    # do a median filter on the COM traces if indicated
    if wsize is not None:
        if wsize % 2 == 0:
            wsize += 1
            print("medfilt_window was not odd, changing to: {}".format(wsize))

        from scipy.signal import medfilt

        c3d = medfilt(c3d, (wsize, 1))

    c3dsi = np.squeeze(c3dfile["sampleID"])
    com3d_dict_ = {}
    for (i, s) in enumerate(c3dsi):
        com3d_dict_[s] = c3d[i]

    return com3d_dict_
