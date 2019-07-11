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



def close_open_vids(
    lastvid, lastvid_, currvid, currvid_, framecnt, cnames, vids,
    vid_dir_flag, viddir, currentframes, maxframes):
    """Track which videos are required for each batch and open/close them.

    When long recordings are split over many video files,
    ffmpeg cannot spawn enough processes at one time.
    Thus, we need to keep track of which videos are required
    for each batch and open/close them as necessary.
    """
    ext = '.' + list(vids.keys())[0].split('.')[-1]
    # Only trigger closing if there is a "new" lastvid
    if lastvid != lastvid_:
        print('attempting to close video')
        for n in range(len(cnames)):
            for key in list(vids[cnames[n]].keys()):
                vikey = key.split(os.sep)[1]
                if lastvid == vikey:
                    print("Closing video: {}".format(key))
                    vids[cnames[n]][key].close()
        lastvid_ = lastvid

    # Open new vids for this interval
    if currvid != currvid_:
        currvid_ = currvid
        for j in range(len(cnames)):
            if vid_dir_flag:
                addl = ''
            else:
                # TODO(undefined): camnames
                addl = os.listdir(os.path.join(viddir, cnames[j]))[0]
            vids[cnames[j]] = \
                generate_readers(
                    viddir,
                    os.path.join(cnames[j], addl),
                    minopt=currentframes // framecnt * framecnt - framecnt,
                    maxopt=maxframes,
                    extension=ext)
    return vids, lastvid_, currvid_


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

    for i in range(len(mp4files)):
        if pathonly:
            out[mp4files_scrub[i]] = os.path.join(viddir, mp4files[i])
        else:
            out[mp4files_scrub[i]] = \
                imageio.get_reader(os.path.join(viddir, mp4files[i]))
    return out


def generate_readers_wrap(CONFIG_PARAMS, vid_dir_flag=False):
    """TODO(description).

    This is a new version of generate_readers that calls the old version.
    This higher level wrapper accounts for the use case
    wherein the numbered subdirectories have not been removed from the
    video folders
    """
    vids = {}
    for i in range(len(CONFIG_PARAMS['CAMNAMES'])):
        if vid_dir_flag:
            addl = ''
        else:
            addl = os.listdir(os.path.join(
                CONFIG_PARAMS['viddir'], CONFIG_PARAMS['CAMNAMES'][i]))[0]
        vids[CONFIG_PARAMS['CAMNAMES'][i]] = \
            generate_readers(
                CONFIG_PARAMS['viddir'],
                os.path.join(CONFIG_PARAMS['CAMNAMES'][i], addl),
                maxopt=70000, extension='.mp4')
    return vids


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


def collect_predictions_2D(model, valid_generator):
    """Return prediction for each model of images generated with valid_generator.

    Returns the locations of the peaks for each keypoint, as predicted by the
    model for a set of input images, peak positions for the ground truth
    are also returned
    """
    num_ims = valid_generator.__len__()

    # Get first object, to be used to initialize other variables
    # with the correct size
    firstim = valid_generator.__getitem__(0)
    num_views = len(valid_generator.camnames)
    num_markers = valid_generator.n_channels_out
    batch_size = valid_generator.batch_size

    valid = np.zeros((batch_size * num_ims, num_views, num_markers, 2))
    truth = np.zeros((batch_size * num_ims, num_views, num_markers, 2))
    valid_peak = np.zeros((batch_size * num_ims, num_views, num_markers))

    cnt = 0
    for i in range(num_ims):
        ims = valid_generator.__getitem__(i)
        if i % 100 == 0:
            print(i)
        for j in range(batch_size):
            im = ims[0][j * num_views:(j + 1) * num_views]
            pred_single = model.predict(im)
            truth_single = ims[1][j * num_views:(j + 1) * num_views]
            for k in range(pred_single.shape[-1]):
                valid[cnt, :, k, :] = batch_maximum(pred_single[:, :, :, k])
                truth[cnt, :, k, :] = batch_maximum(truth_single[:, :, :, k])
                valid_peak[cnt, :, k] = \
                    np.max(np.max(pred_single[:, :, :, k], axis=1), axis=1)
            cnt = cnt + 1
    return valid, truth, valid_peak


def err_vs_kept(err, pmax, bins=100):
    """Plot mean dist error as a func of proportion of frames kept and thresh."""
    thresh_range = np.linspace(0, 1, bins)
    kept = np.zeros((bins,))
    error = np.zeros((bins,))
    for i in range(bins):
        this_thresh = thresh_range[i]
        inds = np.where(pmax > this_thresh)[0]
        error[i] = np.mean(err[inds])
        kept[i] = len(inds) / len(err)
    return error, kept


def err_vs_kept_percentile(err, pmax, bins=100, pct=50):
    """Calculate distance error at a set percentile.

    as a function of the proportion of frames kept, as the peak threshold varies
    """
    thresh_range = np.linspace(0, 1, bins)
    kept = np.zeros((bins,))
    error = np.zeros((bins,))
    for i in range(bins):
        this_thresh = thresh_range[i]
        inds = np.where(pmax > this_thresh)[0]
        if err[inds].size == 0:
            error[i] = np.nan
        else:
            error[i] = np.percentile(err[inds], pct)
        kept[i] = len(inds) / len(err)
    return error, kept


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


def write_unprojected_grids(imstack, fn):
    """Write grids to tif files for visualization.

    input--
        imstack: shape (nvox, nvox, nvox, 9). Last index corresponds to 3
        cameras w/ 3 channels (i.e. RGB)
        fn: list of paths + filenames for each saved grid volume
    """
    imstack[:, :, :, :3] = norm_im(imstack[:, :, :, :3])
    imstack[:, :, :, 3:6] = norm_im(imstack[:, :, :, 3:6])
    imstack[:, :, :, 6:] = norm_im(imstack[:, :, :, 6:])

    imageio.mimwrite(
        fn[0],
        [((imstack[:, :, i, :3]) * 255).astype('uint8') for i in range(128)])

    imageio.mimwrite(
        fn[1],
        [((imstack[:, :, i, 3:6]) * 255).astype('uint8') for i in range(128)])

    imageio.mimwrite(
        fn[2],
        [((imstack[:, :, i, 6:]) * 255).astype('uint8') for i in range(128)])


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


def animate_predictions(preds):
    """Animate predictions."""
    # As a global import, this produces seg faults
    import matplotlib.pyplot as plt

    truth_out = np.zeros((preds.shape[0], 20, 3))
    pred_out = np.zeros((preds.shape[0], 20, 3))

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    aa = ax1.imshow(np.max(np.mean(preds[0, :, :, :, :], axis=2), axis=2))
    ax1.axis('off')
    ax1.set_title('XY')
    bb = ax2.imshow(
        np.max(np.mean(preds[0, :, :, :, :], axis=0), axis=2)[:, ::-1].T)
    ax2.axis('off')
    ax2.set_title('XZ')
    cc = ax3.imshow(
        np.max(np.mean(preds[0, :, :, :, :], axis=1), axis=2)[:, ::-1].T)
    ax3.axis('off')
    ax3.set_title('YZ')

    # plot just the peak values now
    dd_1, = ax4.plot([0], [0], 'or', mfc='none')
    dd_2, = ax4.plot([0], [0], 'ob', mfc='none')
    ax4.set_facecolor('k')
    ax4.set_xlim([0, 96])
    ax4.set_ylim([0, 96])
    ax4.axis('off')
    ax4.set_title('XY')

    ee_1, = ax5.plot([0], [0], 'or', mfc='none')
    ee_2, = ax5.plot([0], [0], 'ob', mfc='none')
    ax5.set_facecolor('k')
    ax5.set_xlim([0, 96])
    ax5.set_ylim([0, 96])
    ax5.axis('off')
    ax5.set_title('XZ')

    ff_1, = ax6.plot([0], [0], 'or', mfc='none')
    ff_2, = ax6.plot([0], [0], 'ob', mfc='none')
    ax6.set_xlim([0, 96])
    ax6.set_ylim([0, 96])
    ax5.set_facecolor('black')
    ax6.axis('off')
    ax6.set_title('YZ')

    for i in range(preds.shape[0]):
        aa.set_data(np.max(np.mean(preds[i, :, :, :, :], axis=2), axis=2))
        bb.set_data(np.max(np.mean(preds[i, :, :, :, :], axis=0), axis=2)[:, ::-1].T)
        cc.set_data(np.max(np.mean(preds[i, :, :, :, :], axis=1), axis=2)[:, ::-1].T)

        # Todo(undefined): ims
        x, y, z = plot_markers_3d(ims[1][i, :, :, :, :], nonan=False)
        dd_1.set_xdata(x)
        dd_1.set_ydata(y)
        ee_1.set_xdata(x)
        ee_1.set_ydata(z)
        ff_1.set_xdata(y)
        ff_1.set_ydata(z)
        truth_out[i, :, 0] = x
        truth_out[i, :, 1] = y
        truth_out[i, :, 2] = z
        x, y, z = plot_markers_3d(preds[i, :, :, :, :])
        dd_2.set_xdata(x)
        dd_2.set_ydata(y)
        ee_2.set_xdata(x)
        ee_2.set_ydata(z)
        ff_2.set_xdata(y)
        ff_2.set_ydata(z)
        pred_out[i, :, 0] = x
        pred_out[i, :, 1] = y
        pred_out[i, :, 2] = z
        fig.canvas.draw()
    return truth_out, pred_out


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


def align_data(datamat, rotate=True):
    """Align mocap data.

    Takes in a raw mocap data matrix and aligns the data via mean subtraction
    of SpineM positions and x-y rotation relative to the orientation of
    the SpineF->SpineM segment

    inputs--
        datmat: n-d numpy array of 20-point 3-D mocap data to be aligned.
        The assumption here is that the
            x-y-z coordinates for individual points have been linearized and
            follow the standard order
            set by the mocapstruct variables in Jesse's data repositories. So
            the standard size is
            (num_frames, 60)

    outputs--
        n-d numpy array, aligned data.
    """
    # SpineM is 5th row (index 4)
    # SpineF is 4th row (index 3)
    # Reshape so that each marker is a row
    tens = np.reshape(datamat, (datamat.shape[0], 20, 3))

    # Subtract Spine M mean
    tens = tens - tens[:, 4, np.newaxis, :]

    if rotate:
        # get angle between SpineM and SpineF
        ang = np.arctan2(
            -(tens[:, 3, 1] - tens[:, 4, 1]), tens[:, 3, 0] - tens[:, 4, 0])

        # Construct rotation matrix
        global_rotmatrix = np.zeros((ang.shape[0], 2, 2))
        global_rotmatrix[:, 0, 0] = np.cos(ang)
        global_rotmatrix[:, 1, 0] = np.sin(ang)
        global_rotmatrix[:, 0, 1] = -np.sin(ang)
        global_rotmatrix[:, 1, 1] = np.cos(ang)

        # Rotate x-y
        rotated = np.zeros((tens.shape[0], 2, 20))
        for i in range(rotated.shape[0]):
            rotated[i] = global_rotmatrix[i] @ tens[i, :, :2].T

        # Add z component back
        rotated = np.concatenate(
            (rotated, np.transpose(tens[:, :, 2, np.newaxis], [0, 2, 1])), axis=1)
        rotated = np.transpose(rotated, [0, 2, 1])

        return np.reshape(rotated, (rotated.shape[0], datamat.shape[1]))

    else:
        return np.reshape(tens, datamat.shape)
