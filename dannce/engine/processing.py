"""Processing functions for dannce."""
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import downscale_local_mean as dsm
import imageio
import os
import ast
import PIL
from six.moves import cPickle
import scipy.io as sio

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import yaml
import shutil
import time
import tensorflow as tf

def initialize_vids(CONFIG_PARAMS, datadict, pathonly=True):
    """
    Modularizes video dict initialization
    """
    flist = []

    for i in range(len(CONFIG_PARAMS['experiment']['CAMNAMES'])):
        # Rather than opening all vids, only open what is needed based on the 
        # maximum frame ID for this experiment and Camera
        for key in datadict.keys():
            flist.append(datadict[key]['frames']
                         [CONFIG_PARAMS['experiment']['CAMNAMES'][i]])

    flist = max(flist)

    vids = {}

    for i in range(len(CONFIG_PARAMS['experiment']['CAMNAMES'])):
        if CONFIG_PARAMS['vid_dir_flag']:
            addl = ''
        else:
            addl = os.listdir(
                os.path.join(
                    CONFIG_PARAMS['experiment']['viddir'],
                    CONFIG_PARAMS['experiment']['CAMNAMES'][i]))[0]
        vids[CONFIG_PARAMS['experiment']['CAMNAMES'][i]] = \
            generate_readers(
                CONFIG_PARAMS['experiment']['viddir'],
                os.path.join(CONFIG_PARAMS['experiment']['CAMNAMES'][i], addl),
                minopt=0,
                maxopt=flist,
                extension=CONFIG_PARAMS['experiment']['extension'],
                pathonly=pathonly)

    return vids

def initialize_vids_train(CONFIG_PARAMS, datadict, e, vids, pathonly=True):
    """
    Initializes video path dictionaries for a training session. This is different
        than a predict session because it operates over a single animal ("experiment")
        oat a time
    """
    for i in range(len(CONFIG_PARAMS['experiment'][e]['CAMNAMES'])):
        # Rather than opening all vids, only open what is needed based on the 
        # maximum frame ID for this experiment and Camera
        flist = []
        for key in datadict.keys():
            if int(key.split('_')[0]) == e:
                flist.append(datadict[key]['frames']
                             [CONFIG_PARAMS['experiment'][e]['CAMNAMES'][i]])

        flist = max(flist)

        if CONFIG_PARAMS['vid_dir_flag']:
            addl = ''
        else:
            addl = os.listdir(os.path.join(
                CONFIG_PARAMS['experiment'][e]['viddir'],
                CONFIG_PARAMS['experiment'][e]['CAMNAMES'][i].split('_')[1]))[0]
        r = \
            generate_readers(
                CONFIG_PARAMS['experiment'][e]['viddir'],
                os.path.join(CONFIG_PARAMS['experiment'][e]
                             ['CAMNAMES'][i].split('_')[1], addl),
                maxopt=flist,  # Large enough to encompass all videos in directory.
                extension=CONFIG_PARAMS['experiment'][e]['extension'],
                pathonly=pathonly)

        # Add e to key
        vids[CONFIG_PARAMS['experiment'][e]['CAMNAMES'][i]] = {}
        for key in r:
            vids[CONFIG_PARAMS['experiment'][e]['CAMNAMES'][i]][str(e) +
                                                                '_' + key]\
                                                                = r[key]

    return vids

def copy_config(RESULTSDIR,main_config,dannce_config,com_config):
    """
    Copies config files into the results directory
    """
    mconfig = os.path.join(RESULTSDIR, 
                           'copy_main_config_' +
                           main_config.split(os.sep)[-1])
    dconfig = os.path.join(RESULTSDIR, 
                           'copy_dannce_config_' +
                           dannce_config.split(os.sep)[-1])
    cconfig = os.path.join(RESULTSDIR, 
                           'copy_com_config_' +
                           com_config.split(os.sep)[-1])

    shutil.copyfile(main_config, mconfig)
    shutil.copyfile(dannce_config, dconfig)
    shutil.copyfile(com_config, cconfig)

def make_paths_safe(params):
	"""Given a parameter dictionary, loops through the keys and replaces any \\ or / with os.sep
	to promote OS agnosticism
	"""
	for key in params.keys():
		if isinstance(params[key], str):
			params[key] = params[key].replace('/', os.sep)
			params[key] = params[key].replace('\\', os.sep)

	return params

def trim_COM_pickle(
    fpath, start_sample, end_sample, opath=None):
    """Trim dictionary entries to the range [start_sample, end_sample].

    spath is the output path for saving the trimmed COM dictionary, if desired
    """
    with open(fpath, 'rb') as f:
        save_data = cPickle.load(f)
    sd = {}

    for key in save_data:
        if key >= start_sample and key <= end_sample:
            sd[key] = save_data[key]

    with open(opath, 'wb') as f:
        cPickle.dump(sd, f)
    return sd

def save_COM_checkpoint(save_data, camnames):
    """Saves a COM dict to mat file

    """

    n = len(list(save_data.keys()))
    num_cams = len(camnames)
    allCOMs = np.zeros((num_cams, n, 2))
    for (i, key) in enumerate(save_data.keys()):
        for c in range(num_cams):
            allCOMs[c, i] = save_data[key][camnames[c]]['COM']

    # Save Coms to a mat file
    comfile = 'allCOMs_distorted.mat'
    sio.savemat(comfile, {'allCOMs': allCOMs})

    # Also save save_data

    sdfile = 'save_data.pickle'

    with open(sdfile, 'wb') as f:
        cPickle.dump(save_data,f)

    return comfile

def inherit_config(child, parent, keys):
    """
    If a key in keys does not exist in child, assigns the key-value in parent to
        child.
    """
    for key in keys:
        if key not in child.keys():
            child[key] = parent[key]

    return child

def batch_rgb2gray(imstack):
    """Convert to gray image-wise.

    batch dimension is first.
    """
    grayim = np.zeros(
        (imstack.shape[0], imstack.shape[1], imstack.shape[2]), 'float32')
    for i in range(grayim.shape[0]):
        grayim[i] = rgb2gray(imstack[i].astype('uint8'))
    return grayim


def return_tile(imstack, fac=2):
    """Crop a larger image into smaller tiles without any overlap."""
    height = imstack.shape[1] // fac
    width = imstack.shape[2] // fac
    out = np.zeros(
        (imstack.shape[0] * fac * fac, height, width, imstack.shape[3]), 'float32')
    cnt = 0
    for i in range(imstack.shape[0]):
        for j in np.arange(0, imstack.shape[1], height):
            for k in np.arange(0, imstack.shape[2], width):
                out[cnt, :, :, :] = imstack[i, j:j + height, k:k + width, :]
                cnt = cnt + 1
    return out


def tile2im(imstack, fac=2):
    """Reconstruct lagrer image from tiled data."""
    height = imstack.shape[1]
    width = imstack.shape[2]
    out = np.zeros(
        (imstack.shape[0] // (fac * fac), height * fac,
            width * fac, imstack.shape[3]),
        'float32')
    cnt = 0
    for i in range(out.shape[0]):
        for j in np.arange(0, out.shape[1], height):
            for k in np.arange(0, out.shape[2], width):
                out[i, j:j + height, k:k + width, :] = imstack[cnt]
                cnt += 1
    return out


def downsample_batch(imstack, fac=2, method='PIL'):
    """Downsample each image in a batch."""
    out = np.zeros(
        (imstack.shape[0], imstack.shape[1] // fac,
            imstack.shape[2] // fac, imstack.shape[3]),
        'float32')
    if method == 'PIL':
        if out.shape[-1] == 3:
            # this is just an RGB image, so no need to loop over channels with PIL
            for i in range(imstack.shape[0]):
                out[i] = np.array(
                    PIL.Image.fromarray(imstack[i].astype('uint8')).resize(
                        (out.shape[2], out.shape[1]), resample=PIL.Image.LANCZOS))
        else:
            for i in range(imstack.shape[0]):
                for j in range(imstack.shape[3]):
                    out[i, :, :, j] = np.array(
                        PIL.Image.fromarray(imstack[i, :, :, j]).resize(
                            (out.shape[2], out.shape[1]), resample=PIL.Image.LANCZOS))

    elif method == 'dsm':
        for i in range(imstack.shape[0]):
            for j in range(imstack.shape[3]):
                out[i, :, :, j] = dsm(imstack[i, :, :, j], (fac, fac))

    elif method == 'nn':
        # do simple, faster nearest neighbors
        for i in range(imstack.shape[0]):
            for j in range(imstack.shape[3]):
                out[i, :, :, j] = imstack[i, ::fac, ::fac, j]
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
                np.squeeze(imstack[i]).shape)
            maxpos[i, 0] = ind[1]
            maxpos[i, 1] = ind[0]
    return maxpos


def generate_readers(
    viddir, camname, minopt=0, maxopt=300000, pathonly=False, extension='.mp4'):
    """Open all mp4 objects with imageio, and return them in a dictionary."""
    print('NOTE: Ignoring mp4 files numbered above {}'.format(maxopt))
    out = {}
    mp4files = \
        [os.path.join(camname, f) for f in os.listdir(os.path.join(viddir, camname))
         if extension in f and int(
            f.rsplit(extension)[0]) <= maxopt and int(
            f.rsplit(extension)[0]) >= minopt]

    # This is a trick (that should work) for getting rid of
    # awkward sub-directory folder names when they are being used
    mp4files_scrub = \
        [os.path.join(
            os.path.normpath(f).split(os.sep)[0],
            os.path.normpath(f).split(os.sep)[-1])
         for f in mp4files]

    pixelformat = "yuv420p"
    input_params = []
    output_params = []

    for i in range(len(mp4files)):
        if pathonly:
            out[mp4files_scrub[i]] = os.path.join(viddir, mp4files[i])
        else:
            out[mp4files_scrub[i]] = \
                    imageio.get_reader(os.path.join(viddir, mp4files[i]), 
                        pixelformat=pixelformat, 
                        input_params=input_params, 
                        output_params=output_params)

    return out

def cropcom(im, com, size=512):
    """Crops single input image around the coordinates com."""
    minlim_r = int(np.round(com[1])) - size // 2
    maxlim_r = int(np.round(com[1])) + size // 2
    minlim_c = int(np.round(com[0])) - size // 2
    maxlim_c = int(np.round(com[0])) + size // 2

    out = im[np.max([minlim_r, 0]):maxlim_r, np.max([minlim_c, 0]):maxlim_c, :]

    dim = out.shape[2]

    # pad with zeros if region ended up outside the bounds of the original image
    if minlim_r < 0:
        out = np.concatenate(
            (np.zeros((abs(minlim_r), out.shape[1], dim)), out),
            axis=0)
    if maxlim_r > im.shape[0]:
        out = np.concatenate(
            (out, np.zeros((maxlim_r - im.shape[0], out.shape[1], dim))),
            axis=0)
    if minlim_c < 0:
        out = np.concatenate(
            (np.zeros((out.shape[0], abs(minlim_c), dim)), out),
            axis=1)
    if maxlim_c > im.shape[1]:
        out = np.concatenate(
            (out, np.zeros((out.shape[0], maxlim_c - im.shape[1], dim))),
            axis=1)

    return out

def write_config(resultsdir, configdict, message, filename='modelconfig.cfg'):
    """Write a dictionary of k-v pairs to file.

    A much more customizable configuration writer. Accepts a dictionary of
    key-value pairs and just writes them all to file,
    together with a custom message
    """
    f = open(resultsdir + filename, 'w')
    for key in configdict:
        f.write('{}: {}\n'.format(key, configdict[key]))
    f.write('message:' + message)


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
            np.argmax(markers[:, :, mark], axis=None), markers[:, :, mark].shape)
        plt.plot(ind[1], ind[0], '.r')


def preprocess_3d(im_stack):
    """Easy inception-v3 style image normalization across a set of images."""
    im_stack /= 127.5
    im_stack -= 1.
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
            np.argmax(stack[:, :, :, mark], axis=None), stack[:, :, :, mark].shape)
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
    with tf.device(stack.device):
        n_mark = stack.shape[-1]
        indices = tf.math.argmax(tf.reshape(stack, [-1,n_mark]), output_type='int32')
        inds = unravel_index(indices, stack.shape[:-1])

        if ~tf.math.reduce_any(tf.math.is_nan(stack[0, 0, 0, :])) and (nonan or not nonan):
            x = inds[1]
            y = inds[0]
            z = inds[2]
        elif not nonan:
            x = tf.Variable(tf.cast(inds[1], 'float32'))
            y = tf.Variable(tf.cast(inds[0], 'float32'))
            z = tf.Variable(tf.cast(inds[3], 'float32'))
            nans = tf.math.is_nan(stack[0, 0, 0, :])
            for mark in range(0,n_mark):
                if nans[mark]:
                    x[mark].assign(np.nan)
                    y[mark].assign(np.nan)
                    z[mark].assign(np.nan)
        return x, y, z


def plot_markers_3d_torch(stack, nonan=True):
    """Return the 3d coordinates for each of the peaks in probability maps."""
    import torch
    n_mark = stack.shape[-1]
    index = stack.flatten(0,2).argmax(dim=0).to(torch.int32)
    inds = unravel_index(index, stack.shape[:-1])
    if ~torch.any(torch.isnan(stack[0, 0, 0, :])) and (nonan or not nonan):
        x = inds[1]
        y = inds[0]
        z = inds[2]
    elif not nonan:
        x = inds[1]
        y = inds[0]
        z = inds[2]
        for mark in range(0,n_mark):
            if torch.isnan(stack[:,:,:,mark]):
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
        im_norm = \
            (im[:, :, :, mark] * (im[:, :, :, mark] >= thresh)) / np.sum(
                im[:, :, :, mark] * (im[:, :, :, mark] >= thresh))
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
    fname, write=True, data=None, num_markers=20, tcoord=True, pmax=False):
    """Save the expected values."""
    if data is None:
        f = open(fname, 'rb')
        data = cPickle.load(f)
        f.close()

    d_coords = np.zeros((len(list(data.keys())), 3, num_markers))
    t_coords = np.zeros((len(list(data.keys())), 3, num_markers))
    sID = np.zeros((len(list(data.keys())),))
    p_max = np.zeros((len(list(data.keys())), num_markers))

    for (i, key) in enumerate(data.keys()):
        d_coords[i] = data[key]['pred_coord']
        if tcoord:
            t_coords[i] = np.reshape(data[key]['true_coord_nogrid'], (3, num_markers))
        if pmax:
            p_max[i] = data[key]['pred_max']
        sID[i] = data[key]['sampleID']

    if write and data is None:
        sio.savemat(
            fname.split('.pickle')[0] + '.mat',
            {'pred': d_coords, 'data': t_coords, 'p_max': p_max, 'sampleID': sID})
    elif write and data is not None:
        sio.savemat(
            fname,
            {'pred': d_coords, 'data': t_coords, 'p_max': p_max, 'sampleID': sID})

    return d_coords, t_coords, p_max, sID


def savedata_tomat(
    fname, vmin, vmax, nvox, write=True, data=None, num_markers=20,
    tcoord=True, tcoord_scale=True, addCOM=None):
    """Save pickled data to a mat file.

    From a save_data structure saved to a *.pickle file, save a matfile
        with useful variables for easier manipulation in matlab.
    Also return pred_out_world and other variables for plotting within jupyter
    """
    if data is None:
        f = open(fname, 'rb')
        data = cPickle.load(f)
        f.close()

    d_coords = np.zeros((list(data.keys())[-1] + 1, 3, num_markers))
    t_coords = np.zeros((list(data.keys())[-1] + 1, 3, num_markers))
    p_max = np.zeros((list(data.keys())[-1] + 1, num_markers))
    log_p_max = np.zeros((list(data.keys())[-1] + 1, num_markers))
    sID = np.zeros((list(data.keys())[-1] + 1,))
    for (i, key) in enumerate(data.keys()):
        d_coords[i] = data[key]['pred_coord']
        if tcoord:
            t_coords[i] = np.reshape(data[key]['true_coord_nogrid'], (3, num_markers))
        p_max[i] = data[key]['pred_max']
        log_p_max[i] = data[key]['logmax']
        sID[i] = data[key]['sampleID']

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

    if write and data is None:
        sio.savemat(
            fname.split('.pickle')[0] + '.mat',
            {'pred': pred_out_world, 'data': t_coords,
             'p_max': p_max, 'sampleID': sID, 'log_pmax': log_p_max})
    elif write and data is not None:
        sio.savemat(
            fname,
            {'pred': pred_out_world, 'data': t_coords,
             'p_max': p_max, 'sampleID': sID, 'log_pmax': log_p_max})
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

    return np.sum(map_ * ((x - expx)**2 + (y - expy)**2))


def spatial_entropy(map_):
    """Calculate the spatial entropy of the input."""
    map_ = map_ / np.sum(map_)
    return -1 * np.sum(map_ * np.log(map_))
