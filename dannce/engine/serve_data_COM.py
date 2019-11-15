"""Define routines for reading/structuring input data for COM."""
import numpy as np
import scipy.io as sio
from dannce.engine import processing as processing
from dannce.engine import ops as ops
import os
from six.moves import cPickle
_DEFAULT_CAM_NAMES = [
    'CameraR', 'CameraL', 'CameraE', 'CameraU', 'CameraS', 'CameraU2']


def prepare_data(CONFIG_PARAMS, vid_dir_flag=True, minopt=0, maxopt=70000, multimode=False):
    """Assemble necessary data structures given a set of config params.

    Given a set of config params, assemble necessary data structures and
    return them -- tailored to center of mass finding
        That is, we are refactoring to get rid of unneeded data structures
        (i.e. data 3d)

    vid_dir_flag: If True (default), the subdirectory (e.g. 60003847388) within
    each Camera directory has been removed.
        If False, it has not been removed, and the name of that directory
        must be appended to the Camera name.

    minopt, maxopt: the minimum and maximum video file labels to be loaded.
    """
    data = sio.loadmat(
        os.path.join(CONFIG_PARAMS['datadir'], CONFIG_PARAMS['datafile'][0]))

    samples = np.squeeze(data['data_sampleID'])

    # Collect data labels and matched frames info. We will keep the 2d labels
    # here just because we could in theory use this for training later.
    # No need to collect 3d data but it sueful for checking predictions
    if len(CONFIG_PARAMS['CAMNAMES']) != len(CONFIG_PARAMS['datafile']):
        raise Exception("need a datafile for every cameras")

    framedict = {}
    ddict = {}
    for i in range(len(CONFIG_PARAMS['datafile'])):
        fr = sio.loadmat(
            os.path.join(CONFIG_PARAMS['datadir'], CONFIG_PARAMS['datafile'][i]))
        framedict[CONFIG_PARAMS['CAMNAMES'][i]] = np.squeeze(fr['data_frame'])

        data = fr['data_2d']
        # reshape data_2d so that it is shape (time points, 2, 20)
        data = np.transpose(np.reshape(data, [data.shape[0], -1, 2]), [0, 2, 1])
        # Correct for Matlab "1" indexing
        data = data - 1

        # Convert to COM only
        if multimode:
            print("Entering multi-mode with {} + 1 targets".format(data.shape[-1]))
            dcom = np.nanmean(data, axis=2, keepdims=True)
            data = np.concatenate((data, dcom), axis=-1)
        else:
            data = np.nanmean(data, axis=2)
            data = data[:, :, np.newaxis]

        ddict[CONFIG_PARAMS['CAMNAMES'][i]] = data

    data_3d = fr['data_3d']
    data_3d = np.transpose(
        np.reshape(data_3d, [data_3d.shape[0], -1, 3]), [0, 2, 1])

    datadict = {}
    datadict_3d = {}
    for i in range(len(samples)):
        frames = {}
        data = {}
        for j in range(len(CONFIG_PARAMS['CAMNAMES'])):
            frames[CONFIG_PARAMS['CAMNAMES'][j]] = \
                framedict[CONFIG_PARAMS['CAMNAMES'][j]][i]
            data[CONFIG_PARAMS['CAMNAMES'][j]] = \
                ddict[CONFIG_PARAMS['CAMNAMES'][j]][i]
        datadict[samples[i]] = {'data': data, 'frames': frames}
        datadict_3d[samples[i]] = data_3d[i]

    # Set up cameras
    cameras = {}
    camera_mats = {}
    for i in range(len(CONFIG_PARAMS['CAMNAMES'])):
        test = sio.loadmat(
            os.path.join(CONFIG_PARAMS['CALIBDIR'], CONFIG_PARAMS['calib_file'][i]))
        cameras[CONFIG_PARAMS['CAMNAMES'][i]] = {
            'K': test['K'], 'R': test['r'], 't': test['t'],
            'RDistort': test['RDistort'], 'TDistort': test['TDistort']}
        camera_mats[CONFIG_PARAMS['CAMNAMES'][i]] = ops.camera_matrix(
            test['K'], test['r'], test['t'])

    # Generate video reader objects
    vids = {}

    for i in range(len(CONFIG_PARAMS['CAMNAMES'])):
        if vid_dir_flag:
            addl = ''
        else:
            addl = os.listdir(
                os.path.join(CONFIG_PARAMS['viddir'], CONFIG_PARAMS['CAMNAMES'][i]))[0]

        vids[CONFIG_PARAMS['CAMNAMES'][i]] = processing.generate_readers(
            CONFIG_PARAMS['viddir'],
            os.path.join(CONFIG_PARAMS['CAMNAMES'][i], addl),
            minopt=minopt,
            maxopt=maxopt)
    return samples, datadict, datadict_3d, cameras, camera_mats, vids


def COM_to_mat(
    comfile, pmax_thresh, camorder=_DEFAULT_CAM_NAMES, savedir=None):
    """Take in a COM pickle file and convert to separate matfiles per camera.

    This function takes in a COM file pickle saved by
    COM_finder_newdata/batch_evaluate.ipynb and converts it into
    separate matfiles for each cameras, with the COM rpedictions thresholded
    using pmax_thresh.

    This is useful for when one would like to refine the COM finder network
    using self-supervision the list camorder is passed so as to have a
    numbering for the camera names included as keys in the comfile
    Because camera numbers are used to save the output COMs
    """
    with open(comfile, 'rb') as f:
        compick = cPickle.load(f)

    # Extract all of the available camera names
    keys = list(compick.keys())
    cams = [cam for cam in compick[keys[0]].keys() if 'triangulation' not in cam]
    sID = np.zeros((len(keys),))
    com_im = np.zeros((len(cams), len(keys), 2)) * np.nan

    cnt = 0
    for key in keys:
        sID[cnt] = key
        for cam in cams:
            thisCOM = compick[key][cam]['COM']
            thispmax = compick[key][cam]['pred_max']
            num_cam = np.where(np.array(camorder) == cam)[0][0]
            if thispmax >= pmax_thresh:
                # then save this COM
                com_im[num_cam, cnt] = thisCOM
        cnt += 1

    if savedir is not None:
        for cam in cams:
            num_cam = np.where(np.array(camorder) == cam)[0][0]
            savedict = {'sID': sID, 'com_im': com_im[num_cam]}
            savename = 'cam{}_thresholded_pmax{}.mat'.format(num_cam + 1, pmax_thresh)
            savename = os.path.join(savedir, savename)
            sio.savemat(savename, savedict)
    else:
        return sID, com_im
