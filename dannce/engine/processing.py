"""Processing functions for dannce."""
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import downscale_local_mean as dsm
import imageio
import os
import dannce.engine.serve_data_DANNCE as serve_data_DANNCE
import PIL
from six.moves import cPickle
import scipy.io as sio

from dannce.engine import io
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml
import shutil
import time


def initialize_vids(CONFIG_PARAMS, datadict, e, vids, pathonly=True):
    """
    Initializes video path dictionaries for a training session. This is different
        than a predict session because it operates over a single animal ("experiment")
        at a time
    """
    for i in range(len(CONFIG_PARAMS["experiment"][e]["camnames"])):
        # Rather than opening all vids, only open what is needed based on the
        # maximum frame ID for this experiment and Camera
        flist = []
        for key in datadict.keys():
            if int(key.split("_")[0]) == e:
                flist.append(
                    datadict[key]["frames"][
                        CONFIG_PARAMS["experiment"][e]["camnames"][i]
                    ]
                )

        flist = max(flist)

        # For COM prediction, we don't prepend experiment IDs
        # So detect this case and act accordingly.
        basecam = CONFIG_PARAMS["experiment"][e]["camnames"][i]
        if "_" in basecam:
            basecam = basecam.split("_")[1]

        if CONFIG_PARAMS["vid_dir_flag"]:
            addl = ""
        else:
            addl = os.listdir(
                os.path.join(
                    CONFIG_PARAMS["experiment"][e]["viddir"],
                    basecam,
                )
            )[0]
        r = generate_readers(
            CONFIG_PARAMS["experiment"][e]["viddir"],
            os.path.join(
                basecam, addl
            ),
            maxopt=flist,  # Large enough to encompass all videos in directory.
            extension=CONFIG_PARAMS["experiment"][e]["extension"],
            pathonly=pathonly,
        )

        if "_" in CONFIG_PARAMS["experiment"][e]["camnames"][i]:
            vids[CONFIG_PARAMS["experiment"][e]["camnames"][i]] = {}
            for key in r:
                vids[CONFIG_PARAMS["experiment"][e]["camnames"][i]][str(e) + "_" + key] = r[
                    key
                ]
        else:
            vids[CONFIG_PARAMS["experiment"][e]["camnames"][i]] = r

    return vids

def infer_params(params, dannce_net, prediction):
    """
    Some parameters that were previously specified in configs can just be inferred
        from others, thus relieving config bloat
    """
    # Grab the camnames from *dannce.mat if not in config
    if 'camnames' not in params:
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

    if any([".mp4" in file for file in video_files]) or \
        any([".avi" in file for file in video_files]):

        print_and_set(params, "vid_dir_flag", True)
    else:
        print_and_set(params, "vid_dir_flag", False)
        viddir = os.path.join(viddir,
                              video_files[0])
        video_files = os.listdir(viddir)

    extension = ".mp4" if any([".mp4" in file for file in video_files]) else ".avi"
    print_and_set(params, "extension", extension)
    video_files = [file for file in video_files if extension in file]
    if len(video_files) > 1:
        video_files = sorted(video_files, key=lambda x: int(x.split(".")[0]))
        chunks = int(video_files[1].split(".")[0]) - int(video_files[0].split(".")[0])
    else:
        chunks = 1e10
    camf = os.path.join(viddir, video_files[0])

    print_and_set(params, "chunks", chunks)

    # Infer N_CHANNELS_IN from the video info
    v = imageio.get_reader(camf)
    im = v.get_data(0)
    v.close()
    print_and_set(params, "n_channels_in", im.shape[-1])

    if dannce_net and 'net' not in params and 'expval' not in params:
        # Here we assume that if the network and expval are specified by the user
        # then there is no reason to infer anything. net + expval compatibility
        # are subsequently verified during check_config()
        #
        # If both the net and expval are unspecified, then we use the simpler
        # 'net_type' + 'train_mode' to select the correct network and set expval.
        # During prediction, the train_mode might be missing, and in any case only the
        # expval needs to be set.
        if 'net_type' not in params:
            raise Exception("Without a net name and expval params, net_type must be specified")
        
        if not prediction and 'train_mode' not in params:
            raise Exception("Need to specific train_mode for DANNCE training")

        if params['net_type'] == 'AVG':
            print_and_set(params, "expval", True)
        elif params['net_type'] == 'MAX':
            print_and_set(params, "expval", False)
        else:
            raise Exception("{} not a valid net_type".format(params['net_type']))

        if not prediction:
            if params['net_type'] == 'AVG' and params['train_mode'] == 'finetune':
                print_and_set(params, "net", "finetune_AVG")
            elif params['net_type'] == 'AVG':
                # This is the network for training from scratch.
                # This will also set the network for "continued", but that network
                # will be ignored, as for continued training the full model file
                # is loaded in without a call to construct the network. However, a value
                # for params['net'] is still required for initialization
                print_and_set(params, "net", "unet3d_big_expectedvalue")
            elif params['net_type'] == 'MAX' and params['train_mode'] == "finetune":
                print_and_set(params, "net", "finetune_MAX")
            elif params['net_type'] == 'MAX':
                print_and_set(params, "net", "unet3d_big")

    elif dannce_net and 'expval' not in params:
        if 'AVG' in params["net"] or 'expected' in params["net"]:
            print_and_set(params, "expval", True)
        else:
            print_and_set(params, "expval", False)

    if dannce_net:
        print_and_set(params,
                      "maxbatch",
                      int(params["max_num_samples"]//params["batch_size"]))

        if "vol_size" in params:
            print_and_set(params, "vmin", -1*params["vol_size"]//2)
            print_and_set(params, "vmax", params["vol_size"]//2)

    return params

    
def print_and_set(params, varname, value):
    # Should add new values to params in place, no need to return
    params[varname] = value
    print("Setting {} to {}.".format(varname, params[varname]))

def check_config(params, dannce_net):
    """
    Add parameter checks and restrictions here.
    """
    check_camnames(params)

    if 'exp' in params.keys():
        for expdict in params['exp']:
            check_camnames(expdict)

    if dannce_net:
        check_net_expval(params)
        check_vmin_vmax(params)

def check_vmin_vmax(params):
    for v in ["vmin", "vmax", "nvox"]:
        if v not in params:
            raise Exception("{} not in parameters. Please add it, or use vol_size instead of vmin and vmax".format(v))

def check_camnames(camp):
    """
    Raises an exception if camera names contain '_'
    """
    if 'camnames' in camp:
        for cam in camp['camnames']:
            if '_' in cam:
                raise Exception("Camera names cannot contain '_' ")

def check_net_expval(params):
    """
    Raise an exception if the network and expval (i.e. AVG/MAX) are incompatible
    """
    if params['expval'] and 'AVG' not in params['net'] and 'expected' not in params['net']:
        raise Exception("Config is set to AVG but you are using a MAX network")
    if not params['expval'] and 'MAX' not in params['net'] and 'expected' in params['net']:
        raise Exception("Config is set to MAX but you are using an AVG network")

def copy_config(RESULTSDIR, main_config, io_config):
    """
    Copies config files into the results directory, and creates results
        directory if necessary
    """
    print("Saving results to: {}".format(RESULTSDIR))

    if not os.path.exists(RESULTSDIR):
        os.makedirs(RESULTSDIR)

    mconfig = os.path.join(
        RESULTSDIR, "copy_main_config_" + main_config.split(os.sep)[-1]
    )
    dconfig = os.path.join(
        RESULTSDIR, "copy_io_config_" + io_config.split(os.sep)[-1]
    )

    shutil.copyfile(main_config, mconfig)
    shutil.copyfile(io_config, dconfig)

def make_data_splits(samples, params, RESULTSDIR, num_experiments):
    """
    Make train/validation splits from list of samples, or load in a specific
        list of sampleIDs if desired.
    """
    # TODO: Switch to .mat from .pickle so that these lists are easier to read
    # and change.

    partition = {}
    if "load_valid" not in params.keys():
        all_inds = np.arange(len(samples))

        # extract random inds from each set for validation
        v = params["num_validation_per_exp"]
        valid_inds = []

        if params["num_validation_per_exp"] > 0:  # if 0, do not perform validation
            for e in range(num_experiments):
                tinds = [
                    i for i in range(len(samples)) if int(samples[i].split("_")[0]) == e
                ]
                valid_inds = valid_inds + list(
                    np.random.choice(tinds, (v,), replace=False)
                )
                valid_inds = list(np.sort(valid_inds))
                
        train_inds = [i for i in all_inds if i not in valid_inds]

        assert (set(valid_inds) & set(train_inds)) == set()

        partition["valid_sampleIDs"] = samples[valid_inds]
        partition["train_sampleIDs"] = samples[train_inds]

        # Save train/val inds
        with open(RESULTSDIR + "val_samples.pickle", "wb") as f:
            cPickle.dump(partition["valid_sampleIDs"], f)

        with open(RESULTSDIR + "train_samples.pickle", "wb") as f:
            cPickle.dump(partition["train_sampleIDs"], f)
    else:
        # Load validation samples from elsewhere
        with open(os.path.join(params["load_valid"], "val_samples.pickle"), "rb",) as f:
            partition["valid_sampleIDs"] = cPickle.load(f)
        partition["train_sampleIDs"] = [
            f for f in samples if f not in partition["valid_sampleIDs"]
        ]

    return partition

def rename_weights(traindir, kkey, mon):
    """
    At the end of DANNCe or COM training, rename the best weights file with the epoch #
        and value of the monitored quantity
    """
    #First load in the training.csv
    r = np.genfromtxt(os.path.join(traindir,'training.csv'),
                      delimiter=',',
                      names=True)
    e = r['epoch']
    q = r[mon]
    minq = np.min(q)
    beste = e[np.argmin(q)]

    newname = 'weights.' + str(int(beste)) + '-' + '{:.5f}'.format(minq) + '.hdf5'

    os.rename(os.path.join(traindir,kkey), os.path.join(traindir,newname))

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

def prepare_save_metadata(params):
    """
    To save metadata, i.e. the prediction param values associated with COM or DANNCE
        output, we need to convert loss and metrics and net into names, and remove
        the 'experiment' field
    """
    meta = params.copy()
    if 'experiment' in meta:
        del meta['experiment']
    if 'loss' in meta:
        meta['loss'] = meta['loss'].__name__
    if 'net' in meta:
        meta['net'] = meta['net'].__name__
    if 'metric' in meta:
        meta['metric'] = [f.__name__ if not isinstance(f,str) else f for f in meta['metric'] ]

    # Need to convert None to string but still want to conserve the metadat structure
    # format, so we don't want to convert the whole dict to a string
    for key in meta.keys():
        if meta[key] is None:
            meta[key] = 'None'

    return meta

def save_COM_dannce_mat(params, com3d, sampleID):
    """
    Instead of saving 3D COM to com3d.mat, save it into the dannce.mat file, which
    streamlines subsequent dannce access.
    """
    com = {}
    com['com3d'] = com3d
    com['sampleID'] = sampleID
    com['metadata'] = prepare_save_metadata(params)

    # Open dannce.mat file, add com and re-save
    print("Saving COM predictions to " + params['label3d_file'])
    rr = sio.loadmat(params['label3d_file'])
    rr['com'] = com
    sio.savemat(params['label3d_file'], rr)

def save_COM_checkpoint(save_data, RESULTSDIR, datadict_, cameras, params):
    """
    Saves COM pickle and matfiles

    """
    # Save undistorted 2D COMs and their 3D triangulations
    f = open(os.path.join(RESULTSDIR, "com3d.pickle"), "wb")
    cPickle.dump(save_data, f)
    f.close()

    # We need to remove the eID in front of all the keys in datadict
    # for prepare_COM to run properly
    datadict_save = {}
    for key in datadict_:
        datadict_save[int(float(key.split("_")[-1]))] = datadict_[key]

    _, com3d_dict = serve_data_DANNCE.prepare_COM(
        os.path.join(RESULTSDIR, "com3d.pickle"),
        datadict_save,
        comthresh=0,
        weighted=False,
        camera_mats=cameras,
        method="median"
    )

    cfilename = os.path.join(RESULTSDIR, "com3d.mat")
    print("Saving 3D COM to {}".format(cfilename))
    samples_keys = list(com3d_dict.keys())

    c3d = np.zeros((len(samples_keys), 3))
    for i in range(len(samples_keys)):
        c3d[i] = com3d_dict[samples_keys[i]]

    sio.savemat(cfilename, {"sampleID": samples_keys, 
                            "com": c3d,
                            "metadata": prepare_save_metadata(params)})

    # Also save a copy into the label3d file
    save_COM_dannce_mat(params, c3d, samples_keys)


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


def grab_exp_file(CONFIG_PARAMS, defaultdir=""):
    """
    Finds the paths to the training experiment yaml files.
    """
    if "exp_path" not in CONFIG_PARAMS:
        raise Exception("exp_path must be defined in DANNCE_CONFIG.")
    else:
        exps = CONFIG_PARAMS["exp_path"]
    print("Using the following exp.yaml files: {}".format(exps))

    return exps


def grab_predict_exp_file(defaultdir=""):
    """
    Finds the paths to the training experiment yaml files.
    """
    def_ep = os.path.join(".", defaultdir)
    exps = os.listdir(def_ep)
    exps = [os.path.join(def_ep, f) for f in exps if "exp.yaml" == f]

    if len(exps) == 0:
        raise Exception("Did not find any exp.yaml file in {}".format(def_ep))
    if len(exps) > 1:
        raise Exception("Multiple files named exp.yaml in {}".format(def_ep))
    print("Using the following exp.yaml files: {}".format(exps))
    return exps[0]


def grab_predict_label3d_file(defaultdir=""):
    """
    Finds the paths to the training experiment yaml files.
    """
    def_ep = os.path.join(".", defaultdir)
    label3d_files = os.listdir(def_ep)
    label3d_files = [
        os.path.join(def_ep, f) for f in label3d_files if "dannce.mat" in f
    ]

    if len(label3d_files) == 0:
        raise Exception("Did not find any *dannce.mat file in {}".format(def_ep))
    print("Using the following *dannce.mat files: {}".format(label3d_files[0]))
    return label3d_files[0]

def load_expdict(params, e, expdict, _DEFAULT_VIDDIR):
    """
    Load in camnames and video directories and label3d files for a single experiment
        during training.
    """
    exp = params.copy()
    exp = make_paths_safe(exp)
    exp["label3d_file"] = expdict["label3d_file"]
    exp["base_exp_folder"] = os.path.dirname(exp["label3d_file"])
    if "viddir" not in expdict.keys():
        # if the videos are not at the _DEFAULT_VIDDIR, then it must
        # be specified in the io.yaml experiment portion
        exp["viddir"] = os.path.join(exp["base_exp_folder"], _DEFAULT_VIDDIR)
    else:
        exp["viddir"] = expdict["viddir"]
    print("Experiment {} using videos in {}".format(e, exp["viddir"]))

    l3d_camnames = io.load_camnames(expdict["label3d_file"])
    if "camnames" in expdict.keys():
        exp["camnames"] = expdict["camnames"]
    elif l3d_camnames is not None:
        exp["camnames"] = l3d_camnames

    print("Experiment {} using camnames: {}".format(e, exp["camnames"]))

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
    print("NOTE: Ignoring mp4 files numbered above {}".format(maxopt))
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


def write_config(resultsdir, configdict, message, filename="modelconfig.cfg"):
    """Write a dictionary of k-v pairs to file.

    A much more customizable configuration writer. Accepts a dictionary of
    key-value pairs and just writes them all to file,
    together with a custom message
    """
    f = open(resultsdir + filename, "w")
    for key in configdict:
        f.write("{}: {}\n".format(key, configdict[key]))
    f.write("message:" + message)


def read_config(filename):
    """Read configuration file.

    :param filename: Path to configuration file.
    """
    with open(filename) as f:
        CONFIG_PARAMS = yaml.safe_load(f)

    return CONFIG_PARAMS


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


def plot_markers_3d(stack, nonan=True):
    """Return the 3d coordinates for each of the peaks in probability maps."""
    x = []
    y = []
    z = []
    for mark in range(stack.shape[-1]):
        ind = np.unravel_index(
            np.argmax(stack[:, :, :, mark], axis=None), stack[:, :, :, mark].shape
        )
        if ~np.isnan(stack[0, 0, 0, mark]) and nonan:
            x.append(ind[1])
            y.append(ind[0])
            z.append(ind[2])
        elif ~np.isnan(stack[0, 0, 0, mark]) and not nonan:
            x.append(ind[1])
            y.append(ind[0])
            z.append(ind[2])
        elif not nonan:
            x.append(np.nan)
            y.append(np.nan)
            z.append(np.nan)
    return x, y, z


def plot_markers_3d_tf(stack, nonan=True):
    """Return the 3d coordinates for each of the peaks in probability maps."""
    import tensorflow as tf
    
    with tf.device(stack.device):
        n_mark = stack.shape[-1]
        indices = tf.math.argmax(tf.reshape(stack, [-1, n_mark]), output_type="int32")
        inds = unravel_index(indices, stack.shape[:-1])

        if ~tf.math.reduce_any(tf.math.is_nan(stack[0, 0, 0, :])) and (
            nonan or not nonan
        ):
            x = inds[1]
            y = inds[0]
            z = inds[2]
        elif not nonan:
            x = tf.Variable(tf.cast(inds[1], "float32"))
            y = tf.Variable(tf.cast(inds[0], "float32"))
            z = tf.Variable(tf.cast(inds[3], "float32"))
            nans = tf.math.is_nan(stack[0, 0, 0, :])
            for mark in range(0, n_mark):
                if nans[mark]:
                    x[mark].assign(np.nan)
                    y[mark].assign(np.nan)
                    z[mark].assign(np.nan)
        return x, y, z


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

        sdict = {"pred": d_coords, 
                 "data": t_coords, 
                 "p_max": p_max, 
                 "sampleID": sID,
                 "metadata": prepare_save_metadata(params)}
    if write and data is None:
        sio.savemat(
            fname.split(".pickle")[0] + ".mat",
            sdict,
        )
    elif write and data is not None:
        sio.savemat(
            fname, sdict
        )

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
                "metadata": prepare_save_metadata(params)
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

def dupe_params(exp, dupes, _N_VIEWS):
    """
    When The number of views (_N_VIEWS) required
        as input to the network is greater than the
        number of actual cameras (e.g. when trying to
        fine-tune a 6-camera network on data from a 
        2-camera system), automatically duplicate necessary
        parameters to match the required _N_VIEWS.
    """

    for d in dupes:
        val = exp[d]
        if _N_VIEWS % len(val) == 0:
            num_reps = _N_VIEWS // len(val)
            exp[d] = val * num_reps
        else:
            raise Exception(
                "The length of the {} list must divide evenly into {}.".format(
                    d, _N_VIEWS
                )
            )

    return exp
