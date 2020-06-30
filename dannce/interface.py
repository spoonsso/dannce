import sys
import numpy as np
import os
from copy import deepcopy
import scipy.io as sio
import imageio
import time

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.losses as keras_losses
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard

import dannce.engine.serve_data_DANNCE as serve_data_DANNCE

from dannce.engine.generator import DataGenerator_3Dconv
from dannce.engine.generator import DataGenerator_3Dconv_frommem
from dannce.engine.generator import DataGenerator_3Dconv_torch
from dannce.engine.generator import DataGenerator_3Dconv_tf
from dannce.engine.generator_aux import DataGenerator_downsample
import dannce.engine.processing as processing
from dannce.engine.processing import savedata_tomat, savedata_expval
from dannce.engine import nets
from dannce.engine import losses
from dannce.engine import ops

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DEFAULT_VIDDIR = 'videos'
_DEFAULT_COMSTRING = 'COM'
_DEFAULT_COMFILENAME = 'com3d.mat'

def com_predict(base_config_path):
    # Load in the params
    base_params = processing.read_config(base_config_path)
    base_params = processing.make_paths_safe(base_params)

    params = processing.read_config(base_params["io_config"])
    params = processing.make_paths_safe(params)
    params = processing.inherit_config(params, base_params, list(base_params.keys()))
    processing.check_config(params)
    # Load the appropriate loss function and network
    try:
        params["loss"] = getattr(losses, params["loss"])
    except AttributeError:
        params["loss"] = getattr(keras_losses, params["loss"])
    params["net"] = getattr(nets, params["net"])

    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpuID"]

    # If params['N_CHANNELS_OUT'] is greater than one, we enter a mode in
    # which we predict all available labels + the COM
    MULTI_MODE = params["N_CHANNELS_OUT"] > 1
    params["N_CHANNELS_OUT"] = params["N_CHANNELS_OUT"] + int(MULTI_MODE)

    # Grab the input file for prediction
    params["label3d_file"] = processing.grab_predict_label3d_file()

    # Also add parent params under the 'experiment' key for compatibility
    # with DANNCE's video loading function
    params["experiment"] = {}
    params["experiment"][0] = params

    # Build net
    print("Initializing Network...")
    model = params["net"](
        params["loss"],
        float(params["lr"]),
        params["N_CHANNELS_IN"],
        params["N_CHANNELS_OUT"],
        params["metric"],
        multigpu=False,
    )

    if "com_predict_weights" not in params.keys():
        wdir = params["com_train_dir"]
        weights = os.listdir(wdir)
        weights = [f for f in weights if ".hdf5" in f]
        weights = sorted(weights, key=lambda x: int(x.split(".")[1].split("-")[0]))
        weights = weights[-1]
        params["com_predict_weights"] = os.path.join(wdir, weights)
    
    print("Loading weights from " + params["com_predict_weights"])
    model.load_weights(params["com_predict_weights"])

    print("COMPLETE\n")

    def evaluate_ondemand(start_ind, end_ind, valid_gen):
        """Perform COM detection over a set of frames.

        :param start_ind: Starting frame index
        :param end_ind: Ending frame index
        :param steps: Subsample every steps frames
        """
        end_time = time.time()
        sample_save = 100
        for i in range(start_ind, end_ind):
            print("Predicting on sample {}".format(i), flush=True)
            if (i - start_ind) % sample_save == 0 and i != start_ind:
                print(i)
                print(
                    "{} samples took {} seconds".format(
                        sample_save, time.time() - end_time
                    )
                )
                end_time = time.time()

            if (i - start_ind) % 1000 == 0 and i != start_ind:
                print("Saving checkpoint at {}th sample".format(i))
                processing.save_COM_checkpoint(
                    save_data, com_predict_dir, datadict_, cameras, params
                )

            pred_ = model.predict(valid_gen.__getitem__(i)[0])

            pred_ = np.reshape(
                pred_,
                [
                    -1,
                    len(params["CAMNAMES"]),
                    pred_.shape[1],
                    pred_.shape[2],
                    pred_.shape[3],
                ],
            )

            for m in range(pred_.shape[0]):

                # By selecting -1 for the last axis, we get the COM index for a
                # normal COM network, and also the COM index for a multi_mode COM network,
                # as in multimode the COM label is put at the end
                pred = pred_[m, :, :, :, -1]
                sampleID_ = partition["valid_sampleIDs"][i * pred_.shape[0] + m]
                save_data[sampleID_] = {}
                save_data[sampleID_]["triangulation"] = {}

                for j in range(pred.shape[0]):  # this loops over all cameras
                    # get coords for each map. This assumes that image are coming
                    # out in pred in the same order as CONFIG_PARAMS['CAMNAMES']
                    pred_max = np.max(np.squeeze(pred[j]))
                    ind = (
                        np.array(processing.get_peak_inds(np.squeeze(pred[j])))
                        * params["DOWNFAC"]
                    )
                    ind[0] += params["CROP_HEIGHT"][0]
                    ind[1] += params["CROP_WIDTH"][0]
                    ind = ind[::-1]
                    # now, the center of mass is (x,y) instead of (i,j)
                    # now, we need to use camera calibration to triangulate
                    # from 2D to 3D

                    if "COMdebug" in params.keys() and j == cnum:
                        # Write preds
                        plt.figure(0)
                        plt.cla()
                        plt.imshow(np.squeeze(pred[j]))
                        plt.savefig(
                            os.path.join(
                                cmapdir, params["COMdebug"] + str(i + m) + ".png"
                            )
                        )

                        plt.figure(1)
                        plt.cla()
                        im = valid_gen.__getitem__(i * pred_.shape[0] + m)
                        plt.imshow(processing.norm_im(im[0][j]))
                        plt.plot(
                            (ind[0] - params["CROP_WIDTH"][0]) / params["DOWNFAC"],
                            (ind[1] - params["CROP_HEIGHT"][0]) / params["DOWNFAC"],
                            "or",
                        )
                        plt.savefig(
                            os.path.join(
                                overlaydir, params["COMdebug"] + str(i + m) + ".png"
                            )
                        )

                    save_data[sampleID_][params["CAMNAMES"][j]] = {
                        "pred_max": pred_max,
                        "COM": ind,
                    }

                    # Undistort this COM here.
                    pts1 = save_data[sampleID_][params["CAMNAMES"][j]]["COM"]
                    pts1 = pts1[np.newaxis, :]
                    pts1 = ops.unDistortPoints(
                        pts1,
                        cameras[params["CAMNAMES"][j]]["K"],
                        cameras[params["CAMNAMES"][j]]["RDistort"],
                        cameras[params["CAMNAMES"][j]]["TDistort"],
                        cameras[params["CAMNAMES"][j]]["R"],
                        cameras[params["CAMNAMES"][j]]["t"],
                    )
                    save_data[sampleID_][params["CAMNAMES"][j]]["COM"] = np.squeeze(
                        pts1
                    )

                # Triangulate for all unique pairs
                for j in range(pred.shape[0]):
                    for k in range(j + 1, pred.shape[0]):
                        pts1 = save_data[sampleID_][params["CAMNAMES"][j]]["COM"]
                        pts2 = save_data[sampleID_][params["CAMNAMES"][k]]["COM"]
                        pts1 = pts1[np.newaxis, :]
                        pts2 = pts2[np.newaxis, :]

                        test3d = ops.triangulate(
                            pts1,
                            pts2,
                            camera_mats[params["CAMNAMES"][j]],
                            camera_mats[params["CAMNAMES"][k]],
                        ).squeeze()

                        save_data[sampleID_]["triangulation"][
                            "{}_{}".format(params["CAMNAMES"][j], params["CAMNAMES"][k])
                        ] = test3d

    com_predict_dir = os.path.join(params["com_predict_dir"])
    print(com_predict_dir)

    # Copy the configs for reproducibility
    processing.copy_config(
        com_predict_dir,
        sys.argv[1],
        base_params["io_config"],
    )

    if "COMdebug" in params.keys():
        cmapdir = os.path.join(com_predict_dir, "cmap")
        overlaydir = os.path.join(com_predict_dir, "overlay")
        if not os.path.exists(cmapdir):
            os.makedirs(cmapdir)
        if not os.path.exists(overlaydir):
            os.makedirs(overlaydir)
        cnum = params["CAMNAMES"].index(params["COMdebug"])
        print("Writing " + params["COMdebug"] + " confidence maps to " + cmapdir)
        print("Writing " + params["COMdebug"] + "COM-image overlays to " + overlaydir)

    samples, datadict, datadict_3d, cameras, camera_mats = serve_data_DANNCE.prepare_data(
        params, multimode=MULTI_MODE, prediction=True, return_cammat=True, nanflag=False,
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
    vids = processing.initialize_vids(params, 
                                      datadict, 
                                      0,
                                      vids,
                                      pathonly=True)

    # Parameters
    valid_params = {
        "dim_in": (
            params["CROP_HEIGHT"][1] - params["CROP_HEIGHT"][0],
            params["CROP_WIDTH"][1] - params["CROP_WIDTH"][0],
        ),
        "n_channels_in": params["N_CHANNELS_IN"],
        "batch_size": 1,
        "n_channels_out": params["N_CHANNELS_OUT"],
        "out_scale": params["SIGMA"],
        "camnames": {0: params["CAMNAMES"]},
        "crop_width": params["CROP_WIDTH"],
        "crop_height": params["CROP_HEIGHT"],
        "downsample": params["DOWNFAC"],
        "labelmode": "coord",
        "chunks": params["chunks"],
        "shuffle": False,
        "dsmode": params["dsmode"],
        "preload": False,
    }

    partition = {}
    partition["valid_sampleIDs"] = samples
    labels = datadict

    save_data = {}

    valid_generator = DataGenerator_downsample(
        partition["valid_sampleIDs"], labels, vids, **valid_params
    )

    # If we just want to analyze a chunk of video...
    st_ind = (
        params["start_sample_index"] if "start_sample_index" in params.keys() else 0
    )
    if params["max_num_samples"] == "max":
        evaluate_ondemand(st_ind, len(valid_generator), valid_generator)
    else:
        endIdx = np.min([st_ind + params["max_num_samples"], len(valid_generator)])
        evaluate_ondemand(st_ind, endIdx, valid_generator)

    processing.save_COM_checkpoint(
        save_data, com_predict_dir, datadict_, cameras, params
    )

    print("done!")


def com_train(base_config_path):
    # Set up parameters
    base_params = processing.read_config(base_config_path)
    base_params = processing.make_paths_safe(base_params)

    params = processing.read_config(base_params["io_config"])
    params = processing.make_paths_safe(params)
    params = processing.inherit_config(params, base_params, list(base_params.keys()))
    processing.check_config(params)

    params["loss"] = getattr(losses, params["loss"])
    params["net"] = getattr(nets, params["net"])

    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpuID"]

    # MULTI_MODE is where the full set of markers is trained on, rather than
    # the COM only. In some cases, this can help improve COMfinder performance.
    MULTI_MODE = params["N_CHANNELS_OUT"] > 1
    params["N_CHANNELS_OUT"] = params["N_CHANNELS_OUT"] + int(MULTI_MODE)

    samples = []
    datadict = {}
    datadict_3d = {}
    cameras = {}
    camnames = {}

    # Use the same label files and experiment settings as DANNCE unless
    # indicated otherwise by using a 'com_exp' block in io.yaml.
    #
    # This can be useful for introducing additional COM-only label files.
    if "com_exp" in params.keys():
        exps = params["com_exp"]
    else:
        exps = params["exp"]
    num_experiments = len(exps)
    params["experiment"] = {}
    for e, expdict in enumerate(exps):
        exp = params.copy()
        exp = processing.make_paths_safe(exp)
        exp["label3d_file"] = expdict["label3d_file"]
        exp["base_exp_folder"] = os.path.dirname(exp["label3d_file"])
        if "viddir" not in expdict.keys():
            # if the videos are not at the _DEFAULT_VIDDIR, then it must
            # be specified in the io.yaml experiment block
            exp["viddir"] = os.path.join(exp["base_exp_folder"],
                                         _DEFAULT_VIDDIR)
        else:
            exp["viddir"] = expdict["viddir"]
        print("Experiment {} using videos in {}".format(e, exp["viddir"]))

        if "CAMNAMES" in expdict.keys():
            exp["CAMNAMES"] = expdict["CAMNAMES"]
        print("Experiment {} using CAMNAMES: {}".format(e, exp["CAMNAMES"]))

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
        camnames[e] = params["experiment"][e]["CAMNAMES"]

    com_train_dir = params["com_train_dir"]

    # Copy the configs into the for reproducibility
    processing.copy_config(
        com_train_dir,
        sys.argv[1],
        base_params["io_config"],
    )

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

    print(
        "Using {} downsampling".format(
            params["dsmode"]
        )
    )

    train_params = {
        "dim_in": (
            params["CROP_HEIGHT"][1] - params["CROP_HEIGHT"][0],
            params["CROP_WIDTH"][1] - params["CROP_WIDTH"][0],
        ),
        "n_channels_in": params["N_CHANNELS_IN"],
        "batch_size": 1,
        "n_channels_out": params["N_CHANNELS_OUT"],
        "out_scale": params["SIGMA"],
        "camnames": camnames,
        "crop_width": params["CROP_WIDTH"],
        "crop_height": params["CROP_HEIGHT"],
        "downsample": params["DOWNFAC"],
        "shuffle": False,
        "chunks": params["chunks"],
        "dsmode": params["dsmode"],
        "preload": False,
    }

    valid_params = deepcopy(train_params)
    valid_params["shuffle"] = False

    partition = processing.make_data_splits(samples,
                                            params,
                                            com_train_dir,
                                            num_experiments)

    labels = datadict
    # Build net
    print("Initializing Network...")

    model = params["net"](
        params["loss"],
        float(params["lr"]),
        params["N_CHANNELS_IN"],
        params["N_CHANNELS_OUT"],
        params["metric"],
        multigpu=False,
    )
    print("COMPLETE\n")

    if params["com_finetune_weights"] is not None:
        weights = os.listdir(params["com_finetune_weights"])
        weights = [f for f in weights if ".hdf5" in f]
        weights = weights[0]

        try:
            model.load_weights(os.path.join(params["com_finetune_weights"], weights))
        except:
            print(
                "Note: model weights could not be loaded due to a mismatch in dimensions.\
                   Assuming that this is a fine-tune with a different number of outputs and removing \
                  the top of the net accordingly"
            )
            model.layers[-1].name = "top_conv"
            model.load_weights(os.path.join(params["com_finetune_weights"], weights), by_name=True)

    if "lockfirst" in params.keys() and params["lockfirst"]:
        for layer in model.layers[:2]:
            layer.trainable = False

    model.compile(
        optimizer=Adam(lr=float(params["lr"])), loss=params["loss"], metrics=["mse"],
    )

    # Create checkpoint and logging callbacks
    if params["num_validation_per_exp"] > 0:
        kkey = "weights.{epoch:02d}-{val_loss:.5f}.hdf5"
        mon = "val_loss"
    else:
        kkey = "weights.{epoch:02d}-{loss:.5f}.hdf5"
        mon = "loss"

    # Create checkpoint and logging callbacks
    model_checkpoint = ModelCheckpoint(
        os.path.join(com_train_dir, kkey),
        monitor=mon,
        save_best_only=True,
        save_weights_only=True,
    )
    csvlog = CSVLogger(os.path.join(com_train_dir, "training.csv"))
    tboard = TensorBoard(
        log_dir=com_train_dir + "logs", write_graph=False, update_freq=100
    )

    # Initialize data structures
    ncams = len(camnames[0])
    dh = (params["CROP_HEIGHT"][1] - params["CROP_HEIGHT"][0]) // params["DOWNFAC"]
    dw = (params["CROP_WIDTH"][1] - params["CROP_WIDTH"][0]) // params["DOWNFAC"]
    ims_train = np.zeros((ncams * len(partition["train_sampleIDs"]), dh, dw, 3), dtype="float32")
    y_train = np.zeros(
        (ncams * len(partition["train_sampleIDs"]), dh, dw, params["N_CHANNELS_OUT"]),
        dtype="float32",
    )
    ims_valid = np.zeros((ncams * len(partition["valid_sampleIDs"]), dh, dw, 3), dtype="float32")
    y_valid = np.zeros(
        (ncams * len(partition["valid_sampleIDs"]), dh, dw, params["N_CHANNELS_OUT"]),
        dtype="float32",
    )

    # Set up generators
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

    def write_debug(trainData=True):
        """
        Factoring re-used debug output code.

        Writes training or validation images to an output directory, together
        with the ground truth COM labels and predicted COM labels, respectively.
        """

        if params["debug"] and not MULTI_MODE:

            if trainData:
                outdir = "debug_im_out"
                ims_out = ims_train
                label_out = y_train
            else:
                outdir = "debug_im_out_valid"
                ims_out = ims_valid
                label_out = model.predict(ims_valid,
                                          batch_size=1)
            # Plot all training images and save
            # create new directory for images if necessary
            debugdir = os.path.join(params["com_train_dir"], outdir)
            print("Saving debug images to: " + debugdir)
            if not os.path.exists(debugdir):
                os.makedirs(debugdir)

            plt.figure()
            for i in range(ims_out.shape[0]):
                plt.cla()
                processing.plot_markers_2d(
                    processing.norm_im(ims_out[i]), label_out[i], newfig=False
                )
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())

                imname = str(i) + ".png"
                plt.savefig(
                    os.path.join(debugdir, imname), bbox_inches="tight", pad_inches=0
                )
        elif params["debug"] and MULTI_MODE:
            print("Note: Cannot output debug information in COM multi-mode")

    write_debug(trainData=True)

    model.fit(
        ims_train,
        y_train,
        validation_data=(ims_valid, y_valid),
        batch_size=params["BATCH_SIZE"] * ncams,
        epochs=params["EPOCHS"],
        callbacks=[csvlog, model_checkpoint, tboard],
        shuffle=True,
    )

    write_debug(trainData=False)

    print("Saving full model at end of training")
    sdir = os.path.join(params["com_train_dir"], "fullmodel_weights")
    if not os.path.exists(sdir):
        os.makedirs(sdir)
    model.save(os.path.join(sdir, "fullmodel_end.hdf5"))


def dannce_train(base_config_path):
    """Entrypoint for dannce training."""
    # Set up parameters
    base_params = processing.read_config(base_config_path)
    base_params = processing.make_paths_safe(base_params)

    params = processing.read_config(base_params["io_config"])
    params = processing.make_paths_safe(params)
    params = processing.inherit_config(params, base_params, list(base_params.keys()))
    processing.check_config(params)
    
    params["loss"] = getattr(losses, params["loss"])
    params["net"] = getattr(nets, params["net"])

    # Default to 6 views but a smaller number of views can be specified in the DANNCE config.
    # If the legnth of the camera files list is smaller than _N_VIEWS, relevant lists will be
    # duplicated in order to match _N_VIEWS, if possible.
    _N_VIEWS = int(params["_N_VIEWS"] if "_N_VIEWS" in params.keys() else 6)

    # Convert all metric strings to objects
    metrics = []
    for m in params["metric"]:
        try:
            m_obj = getattr(losses, m)
        except AttributeError:
            m_obj = getattr(keras.losses, m)
        metrics.append(m_obj)

    # set GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpuID"]

    # find the weights given config path
    if params["dannce_finetune_weights"] != "None":
        weights = os.listdir(params["dannce_finetune_weights"])
        weights = [f for f in weights if ".hdf5" in f]
        weights = weights[0]

        params["dannce_finetune_weights"] = \
            os.path.join(params["dannce_finetune_weights"], weights)

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
    for e, expdict in enumerate(exps):
        exp = params.copy()
        exp = processing.make_paths_safe(exp)
        exp["label3d_file"] = expdict["label3d_file"]
        exp["base_exp_folder"] = os.path.dirname(exp["label3d_file"])
        if "viddir" not in expdict.keys():
            # if the videos are not at the _DEFAULT_VIDDIR, then it must
            # be specified in the io.yaml experiment portion
            exp["viddir"] = os.path.join(exp["base_exp_folder"],
                                         _DEFAULT_VIDDIR)
        else:
            exp["viddir"] = expdict["viddir"]
        print("Experiment {} using videos in {}".format(e, exp["viddir"]))

        if "CAMNAMES" in expdict.keys():
            exp["CAMNAMES"] = expdict["CAMNAMES"]
        print("Experiment {} using CAMNAMES: {}".format(e, exp["CAMNAMES"]))

        (
            exp,
            samples_,
            datadict_,
            datadict_3d_,
            cameras_,
            com3d_dict_,
        ) = do_COM_load(exp, expdict, _N_VIEWS, e, params)

        print("Using {} samples total.".format(len(samples_)))

        samples, datadict, datadict_3d, com3d_dict = serve_data_DANNCE.add_experiment(
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
        camnames[e] = exp["CAMNAMES"]
        print("Using the following cameras: {}".format(camnames[e]))
        params["experiment"][e] = exp

    dannce_train_dir = params["dannce_train_dir"]

    # Copy the configs for reproducibility
    processing.copy_config(
        dannce_train_dir,
        sys.argv[1],
        base_params["io_config"],
    )

    # Additionally, to keep videos unique across experiments, need to add
    # experiment labels in other places. E.g. experiment 0 CameraE's "camname"
    # Becomes 0_CameraE. *NOTE* This function modified camnames in place
    # to add the appropriate experiment ID
    cameras, datadict, params = serve_data_DANNCE.prepend_experiment(
        params, datadict, num_experiments, camnames, cameras
    )

    samples = np.array(samples)

    # Initialize video objects
    vids = {}
    for e in range(num_experiments):
        if params["IMMODE"] == "vid":
            vids = processing.initialize_vids(
                params, datadict, e, vids, pathonly=True
            )

    # Parameters
    if params["EXPVAL"]:
        outmode = "coordinates"
    else:
        outmode = "3dprob"

    gridsize = tuple([params["NVOX"]]*3)

    # When this true, the data generator will shuffle the cameras and then select the first 3,
    # to feed to a native 3 camera model
    if "cam3_train" in params.keys() and params["cam3_train"]:
        cam3_train = True
    else:
        cam3_train = False

    valid_params = {
        "dim_in": (
            params["CROP_HEIGHT"][1] - params["CROP_HEIGHT"][0],
            params["CROP_WIDTH"][1] - params["CROP_WIDTH"][0],
        ),
        "n_channels_in": params["N_CHANNELS_IN"],
        "batch_size": 1,
        "n_channels_out": params["NEW_N_CHANNELS_OUT"],
        "out_scale": params["SIGMA"],
        "crop_width": params["CROP_WIDTH"],
        "crop_height": params["CROP_HEIGHT"],
        "vmin": params["VMIN"],
        "vmax": params["VMAX"],
        "nvox": params["NVOX"],
        "interp": params["INTERP"],
        "depth": params["DEPTH"],
        "channel_combo": params["CHANNEL_COMBO"],
        "mode": outmode,
        "camnames": camnames,
        "immode": params["IMMODE"],
        "shuffle": False,  # We will shuffle later
        "rotation": False,  # We will rotate later if desired
        "vidreaders": vids,
        "distort": True,
        "expval": params["EXPVAL"],
        "crop_im": False,
        "chunks": params["chunks"],
        "preload": False,
    }

    # Setup a generator that will read videos and labels
    tifdirs = []  # Training from single images not yet supported in this demo

    partition = processing.make_data_splits(samples,
                                            params,
                                            dannce_train_dir,
                                            num_experiments)

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
    gridsize = tuple([params["NVOX"]]*3)
    X_train = np.zeros(
        (
            len(partition["train_sampleIDs"]),
            *gridsize,
            params["N_CHANNELS_IN"] * len(camnames[0]),
        ),
        dtype="float32",
    )

    X_valid = np.zeros(
        (
            len(partition["valid_sampleIDs"]),
            *gridsize,
            params["N_CHANNELS_IN"] * len(camnames[0]),
        ),
        dtype="float32",
    )

    X_train_grid = None
    X_valid_grid = None
    if params["EXPVAL"]:
        y_train = np.zeros(
            (len(partition["train_sampleIDs"]), 3, params["NEW_N_CHANNELS_OUT"],),
            dtype="float32",
        )
        X_train_grid = np.zeros(
            (len(partition["train_sampleIDs"]), params["NVOX"] ** 3, 3),
            dtype="float32",
        )

        y_valid = np.zeros(
            (len(partition["valid_sampleIDs"]), 3, params["NEW_N_CHANNELS_OUT"],),
            dtype="float32",
        )
        X_valid_grid = np.zeros(
            (len(partition["valid_sampleIDs"]), params["NVOX"] ** 3, 3),
            dtype="float32",
        )
    else:
        y_train = np.zeros(
            (
                len(partition["train_sampleIDs"]),
                *gridsize,
                params["NEW_N_CHANNELS_OUT"],
            ),
            dtype="float32",
        )

        y_valid = np.zeros(
            (
                len(partition["valid_sampleIDs"]),
                *gridsize,
                params["NEW_N_CHANNELS_OUT"],
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
        if params["EXPVAL"]:
            X_train[i] = rr[0][0]
            X_train_grid[i] = rr[0][1]
        else:
            X_train[i] = rr[0]
        y_train[i] = rr[1]

    if 'debug_volume_tifdir' in params.keys():
        # When this option is toggled in the config, rather than
        # training, the image volumes are dumped to tif stacks.
        # This can be used for debugging problems with calibration or
        # COM estimation
        tifdir = params['debug_volume_tifdir']
        print("Dump training volumes to {}".format(tifdir))
        for i in range(X_train.shape[0]):
            for j in range(len(camnames[0])):
                im = X_train[i, :, :, :, j*3:(j+1)*3]
                im = processing.norm_im(im)*255
                im = im.astype('uint8')
                of = os.path.join(tifdir,
                    partition['train_sampleIDs'][i]+'_cam' + str(j) + '.tif')
                imageio.mimwrite(of, np.transpose(im, [2, 0, 1, 3]))
        print("Done! Exiting.")
        sys.exit()

    print("Loading validation data into memory")
    for i in range(len(partition["valid_sampleIDs"])):
        print(i, end="\r")
        rr = valid_generator.__getitem__(i)
        if params["EXPVAL"]:
            X_valid[i] = rr[0][0]
            X_valid_grid[i] = rr[0][1]
        else:
            X_valid[i] = rr[0]
        y_valid[i] = rr[1]

    # Now we can generate from memory with shuffling, rotation, etc.
    if params["CHANNEL_COMBO"] == "random":
        randflag = True
    else:
        randflag = False

    train_generator = DataGenerator_3Dconv_frommem(
        np.arange(len(partition["train_sampleIDs"])),
        X_train,
        y_train,
        batch_size=params["BATCH_SIZE"],
        random=randflag,
        rotation=params["ROTATE"],
        expval=params["EXPVAL"],
        xgrid=X_train_grid,
        nvox=params["NVOX"],
        cam3_train=cam3_train,
    )
    valid_generator = DataGenerator_3Dconv_frommem(
        np.arange(len(partition["valid_sampleIDs"])),
        X_valid,
        y_valid,
        batch_size=1,
        random=randflag,
        rotation=False,
        expval=params["EXPVAL"],
        xgrid=X_valid_grid,
        nvox=params["NVOX"],
        shuffle=False,
        cam3_train=cam3_train,
    )

    # Build net
    print("Initializing Network...")

    # Currently, we expect four modes of use:
    # 1) Training a new network from scratch
    # 2) Fine-tuning a network trained on a diff. dataset (transfer learning)
    # 3) Continuing to train 1) or 2) from a full model checkpoint (including optimizer state)

    print("NUM CAMERAS: {}".format(len(camnames[0])))

    if params["train_mode"] == "new":
        model = params["net"](
            params["loss"],
            float(params["lr"]),
            params["N_CHANNELS_IN"] + params["DEPTH"],
            params["N_CHANNELS_OUT"],
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
            params["N_CHANNELS_IN"] + params["DEPTH"],
            params["N_CHANNELS_OUT"],
            len(camnames[0]),
            params["NEW_LAST_KERNEL_SIZE"],
            params["NEW_N_CHANNELS_OUT"],
            params["dannce_finetune_weights"],
            params["N_LAYERS_LOCKED"],
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
            params["N_CHANNELS_IN"] + params["DEPTH"],
            params["N_CHANNELS_OUT"],
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
        optimizer=Adam(lr=float(params["lr"])), loss=params["loss"], metrics=metrics,
    )

    print("COMPLETE\n")

    # Create checkpoint and logging callbacks
    if params["num_validation_per_exp"] > 0:
        kkey = "weights.{epoch:02d}-{val_loss:.5f}.hdf5"
        mon = "val_loss"
    else:
        kkey = "weights.{epoch:02d}-{loss:.5f}.hdf5"
        mon = "loss"

    model_checkpoint = ModelCheckpoint(
        os.path.join(dannce_train_dir, kkey),
        monitor=mon,
        save_best_only=True,
        save_weights_only=True,
    )
    csvlog = CSVLogger(os.path.join(dannce_train_dir, "training.csv"))
    tboard = TensorBoard(
        log_dir=dannce_train_dir + "logs", write_graph=False, update_freq=100
    )

    model.fit(
        x=train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        verbose=params["VERBOSE"],
        epochs=params["EPOCHS"],
        callbacks=[csvlog, model_checkpoint, tboard],
    )

    print("Saving full model at end of training")
    sdir = os.path.join(params["dannce_train_dir"], "fullmodel_weights")
    if not os.path.exists(sdir):
        os.makedirs(sdir)
    model.save(os.path.join(sdir, "fullmodel_end.hdf5"))

    print("done!")


def dannce_predict(base_config_path):
    # Set up parameters
    base_params = processing.read_config(base_config_path)
    base_params = processing.make_paths_safe(base_params)
    params = processing.read_config(base_params["io_config"])
    params = processing.make_paths_safe(params)
    params = processing.inherit_config(params, base_params, list(base_params.keys()))
    processing.check_config(params)
    # Load the appropriate loss function and network
    try:
        params["loss"] = getattr(losses, params["loss"])
    except AttributeError:
        params["loss"] = getattr(keras_losses, params["loss"])
    netname = params["net"]
    params["net"] = getattr(nets, params["net"])

    # Default to 6 views but a smaller number of views can be specified in the DANNCE config.
    # If the legnth of the camera files list is smaller than _N_VIEWS, relevant lists will be
    # duplicated in order to match _N_VIEWS, if possible.
    _N_VIEWS = int(params["_N_VIEWS"] if "_N_VIEWS" in params.keys() else 6)

    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpuID"]
    gpuID = params["gpuID"]

    # While we can use experiment files for DANNCE training,
    # for prediction we use the base data files present in the main config
    # Grab the input file for prediction
    params["label3d_file"] = processing.grab_predict_label3d_file()
    params["base_exp_folder"] = os.path.dirname(params["label3d_file"])

    dannce_predict_dir = params["dannce_predict_dir"]

    # default to slow numpy backend if there is no predict_mode in config file. I.e. legacy support
    predict_mode = (
        params["predict_mode"] if "predict_mode" in params.keys() else "numpy"
    )
    print("Using {} predict mode".format(predict_mode))

    # Copy the configs into the dannce_predict_dir, for reproducibility
    processing.copy_config(
        dannce_predict_dir,
        sys.argv[1],
        base_params["io_config"],
    )

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
    ) = do_COM_load(params["experiment"][0],
                    params["experiment"][0],
                    _N_VIEWS,
                    0,
                    params,
                    training=False)

    # Write 3D COM to file. This might be different from the input com3d file
    # if arena thresholding was applied.
    cfilename = os.path.join(dannce_predict_dir, "COM3D_undistorted.mat")
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
    samples, datadict, datadict_3d, com3d_dict = serve_data_DANNCE.add_experiment(
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
    camnames[0] = params["experiment"][0]["CAMNAMES"]

    # Need a '0' experiment ID to work with processing functions.
    # *NOTE* This function modified camnames in place
    # to add the appropriate experiment ID
    cameras, datadict, params = serve_data_DANNCE.prepend_experiment(
        params, datadict, 1, camnames, cameras
    )

    samples = np.array(samples)

    # Initialize video dictionary. paths to videos only.
    # TODO: Remove this IMMODE option if we decide not
    # to support tifs
    if params["IMMODE"] == "vid":
        vids = {}
        vids = processing.initialize_vids(params, 
                                          datadict,
                                          0,
                                          vids, 
                                          pathonly=True)

    # Parameters
    valid_params = {
        "dim_in": (
            params["CROP_HEIGHT"][1] - params["CROP_HEIGHT"][0],
            params["CROP_WIDTH"][1] - params["CROP_WIDTH"][0],
        ),
        "n_channels_in": params["N_CHANNELS_IN"],
        "batch_size": params["BATCH_SIZE"],
        "n_channels_out": params["N_CHANNELS_OUT"],
        "out_scale": params["SIGMA"],
        "crop_width": params["CROP_WIDTH"],
        "crop_height": params["CROP_HEIGHT"],
        "vmin": params["VMIN"],
        "vmax": params["VMAX"],
        "nvox": params["NVOX"],
        "interp": params["INTERP"],
        "depth": params["DEPTH"],
        "channel_combo": params["CHANNEL_COMBO"],
        "mode": "coordinates",
        "camnames": camnames,
        "immode": params["IMMODE"],
        "shuffle": False,
        "rotation": False,
        "vidreaders": vids,
        "distort": True,
        "expval": params["EXPVAL"],
        "crop_im": False,
        "chunks": params["chunks"],
        "preload": False,
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

        device = "cuda:" + gpuID
        genfunc = DataGenerator_3Dconv_torch
    elif predict_mode == "tf":
        device = "/GPU:" + gpuID
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

    if "dannce_predict_model" in params.keys():
        mdl_file = params["dannce_predict_model"]
    else:
        wdir = params["dannce_train_dir"]
        weights = os.listdir(wdir)
        weights = [f for f in weights if ".hdf5" in f]
        weights = sorted(weights, key=lambda x: int(x.split(".")[1].split("-")[0]))
        weights = weights[-1]

        mdl_file = os.path.join(wdir, weights)

    print("Loading model from " + mdl_file)

    if (
        netname == "unet3d_big_tiedfirstlayer_expectedvalue"
        or "FROM_WEIGHTS" in params.keys()
    ):
        # This network is too "custom" to be loaded in as a full model, until I
        # figure out how to unroll the first tied weights layer
        gridsize = tuple([params["NVOX"]]*3)
        model = params["net"](
            params["loss"],
            float(params["lr"]),
            params["N_CHANNELS_IN"] + params["DEPTH"],
            params["N_CHANNELS_OUT"],
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

    # To speed up EXPVAL prediction, rather than doing two forward passes: one for the 3d coordinate
    # and one for the probability map, here we splice on a new output layer after
    # the softmax on the last convolutional layer
    if params["EXPVAL"]:
        from tensorflow.keras.layers import GlobalMaxPooling3D

        o2 = GlobalMaxPooling3D()(model.layers[-3].output)
        model = Model(
            inputs=[model.layers[0].input, model.layers[-2].input],
            outputs=[model.layers[-1].output, o2],
        )

    save_data = {}

    def evaluate_ondemand(start_ind, end_ind, valid_gen):
        """Evaluate experiment.
        :param start_ind: Starting frame
        :param end_ind: Ending frame
        :param valid_gen: Generator
        """
        end_time = time.time()
        for i in range(start_ind, end_ind):
            print("Predicting on batch {}".format(i), flush=True)
            if (i - start_ind) % 10 == 0 and i != start_ind:
                print(i)
                print("10 batches took {} seconds".format(time.time() - end_time))
                end_time = time.time()

            if (i - start_ind) % 1000 == 0 and i != start_ind:
                print("Saving checkpoint at {}th batch".format(i))
                if params["EXPVAL"]:
                    p_n = savedata_expval(
                        dannce_predict_dir + "save_data_AVG.mat",
                        write=True,
                        data=save_data,
                        tcoord=False,
                        num_markers=nchn,
                        pmax=True,
                    )
                else:
                    p_n = savedata_tomat(
                        dannce_predict_dir + "save_data_MAX.mat",
                        params["VMIN"],
                        params["VMAX"],
                        params["NVOX"],
                        write=True,
                        data=save_data,
                        num_markers=nchn,
                        tcoord=False,
                    )

            ims = valid_gen.__getitem__(i)
            pred = model.predict(ims[0])

            if params["EXPVAL"]:
                probmap = pred[1]
                pred = pred[0]
                for j in range(pred.shape[0]):
                    pred_max = probmap[j]
                    sampleID = partition["valid_sampleIDs"][i * pred.shape[0] + j]
                    save_data[i * pred.shape[0] + j] = {
                        "pred_max": pred_max,
                        "pred_coord": pred[j],
                        "sampleID": sampleID,
                    }
            else:
                if predict_mode == "torch":
                    for j in range(pred.shape[0]):
                        preds = torch.as_tensor(
                            pred[j], dtype=torch.float32, device=device
                        )
                        pred_max = preds.max(0).values.max(0).values.max(0).values
                        pred_total = preds.sum((0, 1, 2))
                        xcoord, ycoord, zcoord = processing.plot_markers_3d_torch(preds)
                        coord = torch.stack([xcoord, ycoord, zcoord])
                        pred_log = pred_max.log() - pred_total.log()
                        sampleID = partition["valid_sampleIDs"][i * pred.shape[0] + j]

                        save_data[i * pred.shape[0] + j] = {
                            "pred_max": pred_max.cpu().numpy(),
                            "pred_coord": coord.cpu().numpy(),
                            "true_coord_nogrid": ims[1][j],
                            "logmax": pred_log.cpu().numpy(),
                            "sampleID": sampleID,
                        }

                elif predict_mode == "tf":
                    # get coords for each map
                    with tf.device(device):
                        for j in range(pred.shape[0]):
                            preds = tf.constant(pred[j], dtype="float32")
                            pred_max = tf.math.reduce_max(
                                tf.math.reduce_max(tf.math.reduce_max(preds))
                            )
                            pred_total = tf.math.reduce_sum(
                                tf.math.reduce_sum(tf.math.reduce_sum(preds))
                            )
                            xcoord, ycoord, zcoord = processing.plot_markers_3d_tf(
                                preds
                            )
                            coord = tf.stack([xcoord, ycoord, zcoord], axis=0)
                            pred_log = tf.math.log(pred_max) - tf.math.log(pred_total)
                            sampleID = partition["valid_sampleIDs"][
                                i * pred.shape[0] + j
                            ]

                            save_data[i * pred.shape[0] + j] = {
                                "pred_max": pred_max.numpy(),
                                "pred_coord": coord.numpy(),
                                "true_coord_nogrid": ims[1][j],
                                "logmax": pred_log.numpy(),
                                "sampleID": sampleID,
                            }

                else:
                    # get coords for each map
                    for j in range(pred.shape[0]):
                        pred_max = np.max(pred[j], axis=(0, 1, 2))
                        pred_total = np.sum(pred[j], axis=(0, 1, 2))
                        xcoord, ycoord, zcoord = processing.plot_markers_3d(
                            pred[j, :, :, :, :]
                        )
                        coord = np.stack([xcoord, ycoord, zcoord])
                        pred_log = np.log(pred_max) - np.log(pred_total)
                        sampleID = partition["valid_sampleIDs"][i * pred.shape[0] + j]

                        save_data[i * pred.shape[0] + j] = {
                            "pred_max": pred_max,
                            "pred_coord": coord,
                            "true_coord_nogrid": ims[1][j],
                            "logmax": pred_log,
                            "sampleID": sampleID,
                        }

    max_eval_batch = params["maxbatch"]
    print(max_eval_batch)
    if max_eval_batch == "max":
        max_eval_batch = len(valid_generator)

    if "NEW_N_CHANNELS_OUT" in params.keys():
        nchn = params["NEW_N_CHANNELS_OUT"]
    else:
        nchn = params["N_CHANNELS_OUT"]

    evaluate_ondemand(0, max_eval_batch, valid_generator)

    if params["EXPVAL"]:
        p_n = savedata_expval(
            dannce_predict_dir + "save_data_AVG.mat",
            write=True,
            data=save_data,
            tcoord=False,
            num_markers=nchn,
            pmax=True,
        )
    else:
        p_n = savedata_tomat(
            dannce_predict_dir + "save_data_MAX.mat",
            params["VMIN"],
            params["VMAX"],
            params["NVOX"],
            write=True,
            data=save_data,
            num_markers=nchn,
            tcoord=False,
        )

    print("done!")

def do_COM_load(exp, expdict, _N_VIEWS, e, params, training=True):
    """
    Factors COM loading and processing code, which is shared by
    dannce_train() and dannce_predict()
    """
    if "com_file" in expdict.keys():
        exp["com_file"] = expdict["com_file"]
    else:
        # If one wants to use default pathing, then the COM directory
        # is extracted from the com_predict_dir by searching for the 
        # path downstream of _DEFAULT_COMSTRING. If the comfilenames
        # is at a location that does not match this pattern, then it must
        # be specified
        if _DEFAULT_COMSTRING not in exp["com_predict_dir"]:
            raise Exception("Default COM directory not found",
                ",please add an absolute path in your experiment defintions.")
        compath = _DEFAULT_COMSTRING + \
                exp["com_predict_dir"].split(_DEFAULT_COMSTRING)[-1]
        exp["com_file"] = os.path.join(exp["base_exp_folder"],
                                       compath,
                                       _DEFAULT_COMFILENAME)

    print("Experiment {} using com3d: {}".format(e, exp["com_file"]))
    (
        samples_,
        datadict_,
        datadict_3d_,
        cameras_,
    ) = serve_data_DANNCE.prepare_data(exp, 
                                       prediction = False if training else True,
                                       nanflag=False)

    # If len(exp['CAMNAMES']) divides evenly into _N_VIEWS, duplicate here
    # This must come after loading in this excperiment's data because there
    # is an assertion that len(exp['CAMNAMES']) == the number of cameras
    # in the label files (which will not be duplicated)
    exp = processing.dupe_params(exp, ["CAMNAMES"], _N_VIEWS)

    # New option: if there is "clean" data (full marker set), can take the
    # 3D COM from the labels
    if "COM_fromlabels" in exp.keys() and exp["COM_fromlabels"] and training:
        print("For experiment {}, calculating 3D COM from labels".format(e))
        com3d_dict_ = deepcopy(datadict_3d_)
        for key in com3d_dict_.keys():
            com3d_dict_[key] = np.nanmean(datadict_3d_[key], axis=1, keepdims=True)
    else:  # then use the com files
        print(
            "Loading 3D COM and samples from file: {}".format(exp["com_file"])
        )
        if '.mat' in exp["com_file"]:
            c3dfile = sio.loadmat(exp["com_file"])
            c3d = c3dfile["com"]
            c3dsi = np.squeeze(c3dfile["sampleID"])
            com3d_dict_ = {}
            for (i, s) in enumerate(c3dsi):
                com3d_dict_[s] = c3d[i]

            # verify all of the datadict_ keys are in this sample set
            assert (set(c3dsi) & set(list(datadict_.keys()))) == set(
                list(datadict_.keys())
            )
        elif '.pickle' in exp["com_file"]:
            datadict_, com3d_dict_ = serve_data_DANNCE.prepare_COM(
                exp["com_file"],
                datadict_,
                comthresh=params["comthresh"],
                weighted=params["weighted"],
                camera_mats=cameras_,
                method=params["com_method"],
            )
        else:
            raise Exception("com3d file but be .pickle or .mat")
        # Remove any 3D COMs that are beyond the confines off the 3D arena
        pre = len(samples_)
        samples_ = serve_data_DANNCE.remove_samples_com(
            samples_,
            com3d_dict_,
            rmc=True,
            cthresh=exp["cthresh"],
        )
        msg = "Detected {} bad COMs and removed the associated frames from the dataset"
        print(msg.format(pre - len(samples_)))
    return exp, samples_, datadict_, datadict_3d_, cameras_, com3d_dict_