"""Handle training and prediction for DANNCE and COM networks."""
from pkg_resources import packaging
import sys
import numpy as np
import os
from copy import deepcopy
import scipy.io as sio
import imageio
import time
import gc
from collections import OrderedDict
import torch
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.losses as keras_losses
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
import dannce.engine.serve_data_DANNCE as serve_data_DANNCE
from dannce.engine.generator_cxf import DataGenerator_3Dconv
from dannce.engine.generator_cxf import DataGenerator_3Dconv_torch_video_canvas_faster as DataGenerator_3Dconv_torch_video
from dannce.engine.generator_cxf import DataGenerator_3Dconv_torch_video_canvas_faster_single as DataGenerator_3Dconv_torch_video_single
from dannce.engine.generator_cxf import DataGenerator_3Dconv_frommem
from dannce.engine.generator import DataGenerator_3Dconv_npy
from dannce.engine.generator_cxf import DataGenerator_3Dconv_torch
from dannce.engine.generator import DataGenerator_3Dconv_tf

import dannce.engine.processing_cxf as processing
from dannce.engine.processing_cxf import savedata_tomat, savedata_expval
from dannce.engine import nets, losses, ops, io
from dannce import (
    _param_defaults_dannce,
    _param_defaults_shared,
    _param_defaults_com,
)
import dannce.engine.inference_cxf as inference
from dannce.utils_cxf.cameraIntrinsics_OpenCV import cv2_pose_to_matlab_pose
import matplotlib
import tqdm
import pickle

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


def load_label3d_dataset(label3d_file):
    import pickle
    label3d_pkl = os.path.splitext(label3d_file)[0] + ".pkl"
    if os.path.exists(label3d_pkl):
        pkldata = pickle.load(open(label3d_pkl, "rb"))
        data_3D = pkldata["data_3D"]
        camParams = pkldata["camParams"]
        com3d = pkldata["com3d"]
        imageNames = pkldata["imageNames"]
        vol_size = pkldata["vol_size"]
    else:
        matdata = sio.loadmat(label3d_file)
        data_3D = matdata["data_3D"]
        com3d = matdata['com3d'] if 'com3d' in matdata else None
        imageNamesOrig = np.squeeze(matdata['imageNames'])
        imageNames = [str(imageNamesOrig[i][0][0][0]) 
                        for i in range(imageNamesOrig.shape[0])]
        vol_size = matdata["vol_size"] if "vol_size" in matdata else None
        camParamsOrig = matdata["camParams"] if "camParams" in matdata else matdata["params"]
        camParams = list()
        for icam in range(camParamsOrig.shape[0]):
            camParam = {key: camParamsOrig[icam][0][key][0] 
                            for key in ['K', 'RDistort', 'TDistort', 't', 'r']}
            camParam['R'] = camParam['r']
            camParams.append(camParam)
    pts3d = data_3D.reshape((data_3D.shape[0], -1, 3)) # (N, K, 3)
    pts3d_T = pts3d.transpose((0,2,1)) # (N, 3, K)
    com3d = (np.nanmin(pts3d, axis=1) + np.nanmax(pts3d, axis=1)) / 2 if com3d is None else com3d
    return pts3d_T, com3d, camParams, imageNames, vol_size


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
    metrics = nets.get_metrics(params)

    # set GPU ID
    if not params["multi_gpu_train"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]
    
    if not params["gpu_id"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]
    
    for gpu_device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu_device, True)

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # find the weights given config path
    if params["dannce_finetune_weights"] is not None:
        params["dannce_finetune_weights"] = processing.get_ft_wt(params)
        print("Fine-tuning from {}".format(params["dannce_finetune_weights"]))

    samples = []
    datadict = {}      #label:= filenames
    datadict_3d = {}   #3d xyz
    com3d_dict = {}
    exp_voxel_size = {}
    cameras = {}
    camnames = {}
    exps = params["exp"]

    num_experiments = len(exps)
    params["experiment"] = {}
    total_chunks = {}

    for e, expdict in enumerate(exps):
        pts3d_T, com3d, camParams, imageNames, vol_size = load_label3d_dataset(expdict["label3d_file"])
        ntime = pts3d_T.shape[0]
        exp_id = str(e)
        frame_id = [exp_id+'_'+str(i) for i in range(ntime)]
        com3d_dict.update(dict(zip(frame_id, com3d)))
        datadict_3d.update(dict(zip(frame_id, pts3d_T)))
        datadict.update(dict(zip(frame_id, imageNames)))
        
        samples += frame_id
            
        ncamera = len(camParams)
        camnames[int(exp_id)] = [f'Camera{i+1}' for i in range(ncamera)]
        cameras[int(exp_id)] = OrderedDict(zip(camnames[int(exp_id)], camParams))
        exp_voxel_size[int(exp_id)]=vol_size

    dannce_train_dir = params["dannce_train_dir"]

    # Dump the params into file for reproducibility
    processing.save_params(dannce_train_dir, params)

    samples = np.array(samples)

    if params["use_npy"]:
        # Add all npy volume directories to list, to be used by generator
        npydir = {}
        for e in range(num_experiments):
            npydir[e] = params["experiment"][e]["npy_vol_dir"]

        samples = processing.remove_samples_npy(npydir, samples, params)

    # Parameters
    if params["expval"]:
        outmode = "coordinates"
    else:
        outmode = "3dprob"

    gridsize = tuple([params["nvox"]] * 3)
    vids = None
    # When this true, the data generator will shuffle the cameras and then select the first 3,
    # to feed to a native 3 camera model
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
            "mono": params["mono"],
            "mirror": params["mirror"],
            "exp_voxel_size": exp_voxel_size
        }

        # Setup a generator that will read videos and labels
        tifdirs = []  # Training from single images not yet supported in this demo
        DataGen = DataGenerator_3Dconv if False else DataGenerator_3Dconv_torch
        train_generator = DataGen(
            partition["train_sampleIDs"],
            datadict,
            datadict_3d,
            cameras,
            partition["train_sampleIDs"],
            com3d_dict,
            tifdirs,
            **valid_params
        )
        valid_generator = DataGen(
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
        for i in tqdm.tqdm(range(len(partition["train_sampleIDs"]))):
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
        for i in tqdm.tqdm(range(len(partition["valid_sampleIDs"]))):
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

    if params["n_rand_views"] == 0:
        print("Using default n_rand_views augmentation with {} views and with replacement".format(n_views))
        print("To disable n_rand_views augmentation, set it to None in the config.")
        params["n_rand_views"] = n_views
        params["rand_view_replace"] = True


    shared_args = {'chan_num': params["chan_num"],
                   'expval': params["expval"],
                   'nvox': params["nvox"],
                   'heatmap_reg': params["heatmap_reg"],
                   'heatmap_reg_coeff': params["heatmap_reg_coeff"]}
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

    Xtemp, Ytemp = train_generator[0]
    # Build net
    print("Initializing Network...")

    # Currently, we expect four modes of use:
    # 1) Training a new network from scratch
    # 2) Fine-tuning a network trained on a diff. dataset (transfer learning)
    # 3) Continuing to train 1) or 2) from a full model checkpoint (including optimizer state)

    # if params["multi_gpu_train"]:
    # strategy = tf.distribute.MirroredStrategy()
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # scoping = strategy.scope()
    # else:
    import contextlib
    scoping = contextlib.suppress()

    print("NUM CAMERAS: {}".format(len(camnames[0])))

    # with scoping:
    with strategy.scope():
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
            fargs = [params["loss"],
                     float(params["lr"]),
                     params["chan_num"] + params["depth"],
                     params["n_channels_out"],
                     len(camnames[0]),
                     params["new_last_kernel_size"],
                     params["new_n_channels_out"],
                     params["dannce_finetune_weights"],
                     params["n_layers_locked"],
                     False,
                     True,
                     gridsize]
            try:
                model = params["net"](
                        *fargs
                )
            except:
                if params["expval"]:
                    print("Could not load weights for finetune (likely because you are finetuning a previously finetuned network). Attempting to finetune from a full finetune model file.")
                    model = nets.finetune_fullmodel_AVG(
                            *fargs
                    )
                else:
                    raise Exception("Finetuning from a previously finetuned model is currently possible only for AVG models")
        elif params["train_mode"] == "continued":
            model = load_model(
                params["dannce_finetune_weights"],
                custom_objects={
                    "ops": ops,
                    "slice_input": nets.slice_input,
                    "mask_nan_keep_loss": losses.mask_nan_keep_loss,
                    "mask_nan_l1_loss": losses.mask_nan_l1_loss,
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

        if params["heatmap_reg"]:
            model = nets.add_heatmap_output(model)

        if params["heatmap_reg"] or params["train_mode"] != "continued":
            # recompiling a full model will reset the optimizer state
            model.compile(
                optimizer=Adam(learning_rate=float(params["lr"])),
                loss=params["loss"] if not params["heatmap_reg"] else [params["loss"], losses.heatmap_max_regularizer],
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
            self.saveE = np.arange(50, total_epochs, 50)
        def on_epoch_end(self, epoch, logs=None):
            lkey = 'val_loss' if 'val_loss' in logs else 'loss'
            val_loss = logs[lkey]
            if epoch in self.saveE:
                # Do a garbage collect to combat keras memory leak
                gc.collect()
                print("Saving checkpoint weights at epoch {}".format(epoch))
                src = os.path.join(self.odir, f'epoch_{epoch}.hdf5')
                dst = os.path.join(self.odir, 'latest.hdf5')
                self.model.save(src)
                if os.path.exists(dst): os.remove(dst)
                os.symlink(os.path.abspath(src), dst)
                
    callbacks = [csvlog, model_checkpoint, tboard, saveCheckPoint(params['dannce_train_dir'], params["epochs"])]

    if params['expval'] and not params["use_npy"] and not params["heatmap_reg"] and params["save_pred_targets"]:
        save_callback = savePredTargets(params['epochs'],
            X_train,
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
    os.makedirs(sdir, exist_ok=True)

    model = nets.remove_heatmap_output(model, params)
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
    device = f"/GPU:{gpu_id}"
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(devices=[physical_devices[int(gpu_id)]], device_type='GPU')
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

    samples = []
    datadict = {}      #label:= filenames
    datadict_3d = {}   #3d xyz
    com3d_dict = {}
    cameras = {}
    camnames = {}
    exps = params["predict_exp"]
    params["experiment"] = {}
    exp_voxel_size = {}
    for e, expdict in enumerate(exps):
        pts3d_T, com3d, camParams, imageNames, vol_size = load_label3d_dataset(expdict['predict3d_file'])
        ntime = pts3d_T.shape[0]
        pts3d_T *= 0
        exp_id = str(e)
        frame_id = [exp_id+'_'+str(i) for i in range(ntime)]
        com3d_dict.update(dict(zip(frame_id, com3d)))
        datadict_3d.update(dict(zip(frame_id, pts3d_T)))
        datadict.update(dict(zip(frame_id, imageNames)))
        
        samples += frame_id
            
        ncamera = len(camParams)
        camnames[int(exp_id)] = [f'Camera{i+1}' for i in range(ncamera)]
        cameras[int(exp_id)] = OrderedDict(zip(camnames[int(exp_id)], camParams))
        exp_voxel_size[int(exp_id)]=vol_size

    samples = np.array(samples)
    vids = None
    params['camParams'] = camParams
    
    # For real mono prediction
    params["chan_num"] = 1 if params["mono"] else params["n_channels_in"]


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
        "mono": params["mono"],
        "mirror": params["mirror"],
        "predict_flag": True,
        "exp_voxel_size": exp_voxel_size
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
        weights = [f for f in weights if ".hdf5" in f and "checkpoint" not in f]
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
                "mask_nan_l1_loss": losses.mask_nan_l1_loss,
                "euclidean_distance_3D": losses.euclidean_distance_3D,
                "centered_euclidean_distance_3D": losses.centered_euclidean_distance_3D,
            },
        )

    # If there is a heatmap regularization i/o, remove it
    model = nets.remove_heatmap_output(model, params)

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

    device = "/GPU:0"
    save_data={}
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
    imageNames = [datadict[v] for v in partition["valid_sampleIDs"]]
    otherParams = {'imageNames': imageNames, 
                    'com3d': com3d,
                    'vol_size': vol_size,
                    'camParams': camParams,
                    'camnames': camnames}
    if params["expval"]:
        path = os.path.join(params["dannce_predict_dir"], 
                    os.path.splitext(os.path.basename(expdict['predict3d_file']))[0] + "_AVG.mat")
        p_n = savedata_expval(
            path,
            params,
            write=True,
            data=save_data,
            tcoord=False,
            num_markers=n_chn,
            pmax=True,
            otherParams=otherParams
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
            tcoord=False
        )

def dannce_predict_video(params: Dict, video_file: str):
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
    gpu_id = int(params["gpu_id"])
    device = f'cuda:{gpu_id}'
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(devices=[physical_devices[int(gpu_id)]], device_type='GPU')
    gpu_id_0=0
    # While we can use experiment files for DANNCE training,
    # for prediction we use the base data files present in the main config
    # Grab the input file for prediction
    params["label3d_file"] = params['predict_exp'][0]['predict3d_file']
    print('Using the following dannce prediction file as Camera System: ', params["label3d_file"])
    params["base_exp_folder"] = os.path.dirname(params["label3d_file"])

    # default to slow numpy backend if there is no predict_mode in config file. I.e. legacy support
    datadict = {}      #label:= filenames
    datadict_3d = {}   #3d xyz
    com3d_dict = {}
    cameras = {}
    camnames = {}
    exps = params["predict_exp"]
    params["experiment"] = {}

    if isinstance(exps, str):
        exps = [exps]
    else:
        assert isinstance(exps, list) and len(exps) == 1, "Only 1 video is supported"
    
    for e, expdict in enumerate(exps):
        print('Load camera params from ', expdict['predict3d_file'])
        _, _, camParams, _, vol_size = load_label3d_dataset(expdict['predict3d_file'])
        
        ncamera = len(camParams)
        camnames[e] = [f'Camera{i+1}' for i in range(ncamera)]
        cameras[e] = OrderedDict(zip(camnames[e], camParams))

    vids = None
    params['camParams'] = camParams

    # For real mono prediction
    params["chan_num"] = 1 if params["mono"] else params["n_channels_in"]


    # Parameters
    valid_params = {
        "dim_in": (
            params["crop_height"][1] - params["crop_height"][0],
            params["crop_width"][1] - params["crop_width"][0],
        ),
        "n_channels_in": params["n_channels_in"],
        "batch_size": 1,
        "n_channels_out": params["n_channels_out"],
        "out_scale": params["sigma"],
        "crop_width": params["crop_width"],
        "crop_height": params["crop_height"],
        "vmin": params["vmin"],
        "vmax": params["vmax"],
        "nvox": params["nvox"],
        "interp": params["interp"],
        # "depth": params["depth"],
        # "channel_combo": params["channel_combo"],
        # "mode": "coordinates",
        "mode": "coordinates" if params["expval"] else "3dprob",
        "camnames": camnames,
        # "immode": params["immode"],
        "shuffle": False,
        "rotation": False,
        "vidreaders": vids,
        "distort": True,
        "expval": params["expval"],
        "crop_im": False,
        # "chunks": params["chunks"],
        "mono": params["mono"],
        "mirror": False,
        "predict_flag": True,
        "gpu_id": str(gpu_id_0),
    }

    # Datasets
    partition = {"valid_sampleIDs": [f'0_{i}' for i in range(180000)]}

    # TODO: Remove tifdirs arguments, which are deprecated
    tifdirs = []

    # Generate the dataset
    flag = params['predict_video_single_rat'] # use single animal and no mask
    genfunc = DataGenerator_3Dconv_torch_video if flag else DataGenerator_3Dconv_torch_video_single
    
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
    valid_generator.set_video(video_file, gpu_id_0)

    # Build net
    print("Initializing Network...")

    # This requires that the network be saved as a full model, not just weights.
    # As a precaution, we import all possible custom objects that could be used
    # by a model and thus need declarations

    assert params["dannce_predict_model"] is not None
    mdl_file = params["dannce_predict_model"]

    print("Loading model from " + mdl_file)

    with tf.device(f"/GPU:{gpu_id}"):
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
                    "mask_nan_l1_loss": losses.mask_nan_l1_loss,
                    "euclidean_distance_3D": losses.euclidean_distance_3D,
                    "centered_euclidean_distance_3D": losses.centered_euclidean_distance_3D,
                    'mask_nan_heatmap_max_regularizer': losses.mask_nan_heatmap_max_regularizer,
                },
            )

    # If there is a heatmap regularization i/o, remove it
    model = nets.remove_heatmap_output(model, params)

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

    start_batch = 0
    if params['max_eval_batch'] is not None:
        max_eval_batch = min([len(valid_generator), params['max_eval_batch']])
    else:
        max_eval_batch = len(valid_generator)
    
    # max_eval_batch = 30*60*5 #first 10 minutes of the video
    if params["new_n_channels_out"] is not None:
        n_chn = params["new_n_channels_out"]
    else:
        n_chn = params["n_channels_out"]


    save_data={}
    infer_dannce = inference.infer_dannce if params["expval"] else inference.infer_dannce_max
    save_data = infer_dannce(
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

    path = os.path.splitext(video_file)[0] + '_dannce_predict.mat'
    otherParams = {'imageNames': [], 
                    'com3d': [],
                    'vol_size': params['vol_size'],
                    'camParams': camParams,
                    'camnames': camnames}
    p_n = savedata_expval(
        path,
        params,
        write=True,
        data=save_data,
        tcoord=False,
        num_markers=n_chn,
        pmax=True,
        otherParams=otherParams,
    )


def dannce_predict_video_trt(params: Dict, video_file: str):
    """Predict with dannce network

    Args:
        params (Dict): Paremeters dictionary.
    """
    # Depth disabled until next release.
    params["depth"] = False
    n_views = int(params["n_views"])
    gpu_id = int(params["gpu_id"])
    device = f'cuda:{gpu_id}'
    params["base_exp_folder"] = '.'
    datadict = {}      #label:= filenames
    datadict_3d = {}   #3d xyz
    com3d_dict = {}
    cameras = {}
    camnames = {}

    params["experiment"] = {}

    print('Load camera params from segpkl')
    pklfile = os.path.splitext(video_file)[0] + '.segpkl'
    pkldata = pickle.load(open(pklfile, 'rb'))
    camParams = cv2_pose_to_matlab_pose(pkldata['ba_poses'])
    ncamera = len(camParams)
    assert ncamera == n_views, 'ncamera != n_views'
    e=0
    camnames[e] = [f'Camera{i+1}' for i in range(ncamera)]
    cameras[e] = OrderedDict(zip(camnames[e], camParams))

    vids = None
    params['camParams'] = camParams

    # For real mono prediction
    params["chan_num"] = 1 if params["mono"] else params["n_channels_in"]

    # Parameters
    valid_params = {
        "dim_in": (
            params["crop_height"][1] - params["crop_height"][0],
            params["crop_width"][1] - params["crop_width"][0],
        ),
        "n_channels_in": params["n_channels_in"],
        "batch_size": 1,
        "n_channels_out": params["n_channels_out"],
        "out_scale": params["sigma"],
        "crop_width": params["crop_width"],
        "crop_height": params["crop_height"],
        "vmin": params["vmin"],
        "vmax": params["vmax"],
        "nvox": params["nvox"],
        "interp": params["interp"],
        # "depth": params["depth"],
        # "channel_combo": params["channel_combo"],
        # "mode": "coordinates",
        "mode": "coordinates" if params["expval"] else "3dprob",
        "camnames": camnames,
        # "immode": params["immode"],
        "shuffle": False,
        "rotation": False,
        "vidreaders": vids,
        "distort": True,
        "expval": params["expval"],
        "crop_im": False,
        # "chunks": params["chunks"],
        "mono": params["mono"],
        "mirror": False,
        "predict_flag": True,
        "gpu_id": str(gpu_id),
    }

    # Datasets
    partition = {"valid_sampleIDs": [f'0_{i}' for i in range(180000)]}

    # TODO: Remove tifdirs arguments, which are deprecated
    tifdirs = []

    # Generate the dataset
    flag = params['predict_video_single_rat'] # use single animal and no mask
    genfunc = DataGenerator_3Dconv_torch_video_single if flag else DataGenerator_3Dconv_torch_video_single 
    
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
    valid_generator.set_video(video_file, gpu_id, pkldata)

    # Load model from tensorrt
    assert params["dannce_predict_model"] is not None
    mdl_file = params["dannce_predict_model"]
    mdl_file = mdl_file.replace('.hdf5', '.engine')
    print("Loading model from " + mdl_file)
    assert os.path.exists(mdl_file), f"Model file {mdl_file} not found"

    from torch2trt import TRTModule
    with torch.cuda.device(device):
        trt_model = TRTModule()
        trt_model.load_from_engine(mdl_file)

    # To speed up expval prediction, rather than doing two forward passes: one for the 3d coordinate
    # and one for the probability map, here we splice on a new output layer after
    # the softmax on the last convolutional layer
    assert not params["expval"]
    start_batch = 0
    max_eval_batch = params['max_eval_batch']
    if max_eval_batch is not None:
        if isinstance(max_eval_batch, str): max_eval_batch=int(eval(max_eval_batch))
        max_eval_batch = min([len(valid_generator), max_eval_batch])
        print(f'Use the first frames of {max_eval_batch}/{len(valid_generator)}')
    else:
        max_eval_batch = len(valid_generator)
    
    if params["new_n_channels_out"] is not None:
        n_chn = params["new_n_channels_out"]
    else:
        n_chn = params["n_channels_out"]

    save_data={}
    infer_dannce = inference.infer_dannce if params["expval"] else inference.infer_dannce_max_trt
    save_data = infer_dannce(
        start_batch,
        max_eval_batch,
        valid_generator,
        params,
        trt_model,
        partition,
        save_data,
        device,
        n_chn,
    )

    path = os.path.splitext(video_file)[0] + '_dannce_predict.mat'
    otherParams = {'imageNames': [], 
                    'com3d': [],
                    'nclass': valid_generator.nclass,
                    'vol_size': params['vol_size'],
                    'camParams': camParams,
                    'camnames': camnames}
    p_n = savedata_expval(
        path,
        params,
        write=True,
        data=save_data,
        tcoord=False,
        num_markers=n_chn,
        pmax=True,
        otherParams=otherParams,
    )
