"""Scipt to run COM finding over a single experiment.

Usage: python ./predict_COMfinder path_to_config
"""

import numpy as np
import scipy.io as sio
from copy import deepcopy
import sys
import dannce.engine.processing as processing
import keras.losses
from dannce.engine import nets
from dannce.engine import losses
import dannce.engine.ops as ops
from dannce.engine.generator_aux import DataGenerator_downsample
import dannce.engine.serve_data_COM as serve_data
import dannce.engine.serve_data_DANNCE as serve_data_DANNCE
import os
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# Load in the params
PARENT_PARAMS = processing.read_config(sys.argv[1])
PARENT_PARAMS = processing.make_paths_safe(PARENT_PARAMS)

params = processing.read_config(PARENT_PARAMS['COM_CONFIG'])
params = processing.make_paths_safe(params)

# Load the appropriate loss function and network
try:
    params['loss'] = getattr(losses, params['loss'])
except AttributeError:
    params['loss'] = getattr(keras.losses, params['loss'])
params['net'] = getattr(nets, params['net'])

undistort = params['undistort']
vid_dir_flag = params['vid_dir_flag']
_N_VIDEO_FRAMES = params['chunks']

os.environ["CUDA_VISIBLE_DEVICES"] = params['gpuID']

# If params['N_CHANNELS_OUT'] is greater than one, we enter a mode in
# which we predict all available labels + the COM
MULTI_MODE = params['N_CHANNELS_OUT'] > 1
params['N_CHANNELS_OUT'] = params['N_CHANNELS_OUT'] + int(MULTI_MODE)

# Inherit required parameters from main config file

params = \
    processing.inherit_config(params,
                              PARENT_PARAMS,
                              ['CAMNAMES',
                               'CALIBDIR',
                               'calib_file',
                               'extension',
                               'datafile',
                               'datadir',
                               'viddir'])


# Build net
print("Initializing Network...")
model = params['net'](
    params['loss'],
    float(params['lr']),
    params['N_CHANNELS_IN'],
    params['N_CHANNELS_OUT'],
    params['metric'], multigpu=False)

if 'predict_weights' in params.keys():
    model.load_weights(params['predict_weights'])
else:
    wdir = params['RESULTSDIR']#os.path.join('.', 'COM', 'train_results')
    weights = os.listdir(wdir)
    weights = [f for f in weights if '.hdf5' in f]
    weights = sorted(weights,
                     key=lambda x: int(x.split('.')[1].split('-')[0]))
    weights = weights[-1]

    print("Loading weights from " + os.path.join(wdir, weights))
    model.load_weights(os.path.join(wdir, weights))

print("COMPLETE\n")


# TODO(Modularize): Script can likely be broken down into several utils
def evaluate_COM_steps(start_ind, end_ind, steps):
    """Perform COM detection over a set of frames.

    :param start_ind: Starting frame index
    :param end_ind: Ending frame index
    :param steps: Subsample every steps frames
    """
    for i in range(start_ind, end_ind, steps):
        # close videos from unneeded steps
        currentframes = labels[partition['valid'][i]]['frames']
        currentframes = min(list(currentframes.values()))

        m = min([i + steps, len(partition['valid']) - 1])
        maxframes = labels[partition['valid'][m]]['frames']
        maxframes = max(list(maxframes.values()))
        lastvid = str(
            (currentframes // _N_VIDEO_FRAMES) * _N_VIDEO_FRAMES -
            _N_VIDEO_FRAMES) + params['extension']

        # For each camera, cycle through videonames and close unneeded videos
        for n in range(len(params['CAMNAMES'])):
            for key in list(vids[params['CAMNAMES'][n]].keys()):
                vikey = os.path.split(key)[1]
                if lastvid == vikey:
                    print("Closing video: {}".format(key))
                    vids[params['CAMNAMES'][n]][key].close()

        # Open new vids for this interval
        for j in range(len(params['CAMNAMES'])):
            if vid_dir_flag:
                addl = ''
            else:
                addl = os.listdir(
                    os.path.join(params['viddir'], params['CAMNAMES'][j]))[0]

            vids[params['CAMNAMES'][j]] = \
                processing.generate_readers(
                    params['viddir'],
                    os.path.join(params['CAMNAMES'][j], addl),
                    minopt=currentframes // _N_VIDEO_FRAMES * _N_VIDEO_FRAMES,
                    maxopt=maxframes,
                    extension=params['extension'])

        e_ind = np.min([end_ind, i + steps])

        valid_generator = DataGenerator_downsample(
            partition['valid'][i:e_ind], labels, vids, **valid_params)

        pred_ = model.predict_generator(valid_generator, steps=e_ind-i, verbose=1)

        print(i)
        pred_ = np.reshape(
            pred_,
            [-1, len(params['CAMNAMES']), pred_.shape[1], pred_.shape[2], pred_.shape[3]])

        
        for m in range(len(partition['valid'][i:e_ind])):
            # odd loop condition, but it's because at the end of samples,
            # predict_generator will continue to make predictions in a way I
            # don't grasp yet, but also in a way we should ignore

            # By selecting -1 for the last axis, we get the COM index for a
            # normal COM network, and also the COM index for a multi_mode COM network,
            # as in multimode the COM label is put at the end
            pred = pred_[m, :, :, :, -1]
            sampleID_ = partition['valid'][i + m]
            save_data[sampleID_] = {}
            save_data[sampleID_]['triangulation'] = {}

            for j in range(pred.shape[0]):  # this loops over all cameras
                # get coords for each map. This assumes that image are coming
                # out in pred in the same order as CONFIG_PARAMS['CAMNAMES']
                pred_max = np.max(np.squeeze(pred[j]))
                ind = \
                    np.array(processing.get_peak_inds(np.squeeze(pred[j]))) * params['DOWNFAC']
                ind[0] += params['CROP_HEIGHT'][0]
                ind[1] += params['CROP_WIDTH'][0]
                ind = ind[::-1]
                # now, the center of mass is (x,y) instead of (i,j)
                # now, we need to use camera calibration to triangulate
                # from 2D to 3D

                if 'COMdebug' in params.keys() and j == cnum:
                    # Write preds
                    plt.figure(0)
                    plt.cla()
                    plt.imshow(np.squeeze(pred[j]))
                    plt.savefig(os.path.join(cmapdir,
                                             params['COMdebug'] + str(i+m) + '.png'))

                    plt.figure(1)
                    plt.cla()
                    im = valid_generator.__getitem__(i+m)
                    plt.imshow(processing.norm_im(im[0][j]))
                    plt.plot((ind[0]-params['CROP_WIDTH'][0])/params['DOWNFAC'],
                             (ind[1]-params['CROP_HEIGHT'][0])/params['DOWNFAC'],'or')
                    plt.savefig(os.path.join(overlaydir,
                                             params['COMdebug'] + str(i+m) + '.png'))

                save_data[sampleID_][params['CAMNAMES'][j]] = \
                    {'pred_max': pred_max, 'COM': ind}

                # Undistort this COM here.
                pts1 = save_data[sampleID_][params['CAMNAMES'][j]]['COM']
                pts1 = pts1[np.newaxis, :]
                pts1 = ops.unDistortPoints(
                    pts1, cameras[params['CAMNAMES'][j]]['K'],
                    cameras[params['CAMNAMES'][j]]['RDistort'],
                    cameras[params['CAMNAMES'][j]]['TDistort'],
                    cameras[params['CAMNAMES'][j]]['R'],
                    cameras[params['CAMNAMES'][j]]['t'])
                save_data[sampleID_][params['CAMNAMES'][j]]['COM'] = np.squeeze(pts1)

            # Triangulate for all unique pairs
            for j in range(pred.shape[0]):
                for k in range(j + 1, pred.shape[0]):
                    pts1 = save_data[sampleID_][params['CAMNAMES'][j]]['COM']
                    pts2 = save_data[sampleID_][params['CAMNAMES'][k]]['COM']
                    pts1 = pts1[np.newaxis, :]
                    pts2 = pts2[np.newaxis, :]
                    
                    test3d = ops.triangulate(
                        pts1, pts2, camera_mats[params['CAMNAMES'][j]],
                        camera_mats[params['CAMNAMES'][k]]).squeeze()

                    save_data[sampleID_]['triangulation']["{}_{}".format(
                        params['CAMNAMES'][j], params['CAMNAMES'][k])] = test3d

        # Save COM checkpoint in case of later crash. For instance, sometimes
        # there are more stated frames in a video than actually exist, which will crash
        # late into prediction
        processing.save_COM_checkpoint(save_data, params['CAMNAMES'])


RESULTSDIR = os.path.join(params['RESULTSDIR_PREDICT'])
print(RESULTSDIR)

if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)

if 'COMdebug' in params.keys():
    cmapdir = os.path.join(RESULTSDIR, 'cmap')
    overlaydir = os.path.join(RESULTSDIR, 'overlay')
    if not os.path.exists(cmapdir):
        os.makedirs(cmapdir)
    if not os.path.exists(overlaydir):
        os.makedirs(overlaydir)
    cnum = params['CAMNAMES'].index(params['COMdebug'])
    print("Writing " + params['COMdebug'] + " confidence maps to " + cmapdir)
    print("Writing " + params['COMdebug'] + "COM-image overlays to " + overlaydir)

samples, datadict, datadict_3d, cameras, camera_mats, vids = \
    serve_data.prepare_data(
        params, vid_dir_flag=params['vid_dir_flag'], minopt=0, maxopt=0, multimode=MULTI_MODE)

# Zero any negative frames
for key in datadict.keys():
    for key_ in datadict[key]['frames'].keys():
        if datadict[key]['frames'][key_] < 0:
            datadict[key]['frames'][key_] = 0

# The generator expects an experimentID in front of each sample key
samples = ['0_' + str(f) for f in samples]
datadict_ = {}
for key in datadict.keys():
    datadict_['0_' + str(key)] = datadict[key]

datadict = datadict_

# Parameters
valid_params = {
    'dim_in': (params['CROP_HEIGHT'][1]-params['CROP_HEIGHT'][0],
               params['CROP_WIDTH'][1]-params['CROP_WIDTH'][0]),
    'n_channels_in': params['N_CHANNELS_IN'],
    'batch_size': 1,
    'n_channels_out': params['N_CHANNELS_OUT'],
    'out_scale': params['SIGMA'],
    'camnames': {0: params['CAMNAMES']},
    'crop_width': params['CROP_WIDTH'],
    'crop_height': params['CROP_HEIGHT'],
    'downsample': params['DOWNFAC'],
    'labelmode': 'coord',
    'chunks': params['chunks'],
    'shuffle': False,
    'dsmode': params['dsmode'] if 'dsmode' in params.keys() else 'dsm'}

partition = {}
partition['valid'] = samples
labels = datadict
labels_3d = datadict_3d

save_data = {}

# If we just want to analyze a chunk of video...
st_ind = params['start_sample_index'] if 'start_sample_index' in params.keys() else 0
if params['max_num_samples'] == 'max':
    evaluate_COM_steps(st_ind, len(samples), 100)
else:
    evaluate_COM_steps(st_ind, st_ind + params['max_num_samples'], 100)

# Close video objects
for j in range(len(params['CAMNAMES'])):
    for key in vids[params['CAMNAMES'][j]]:
        vids[params['CAMNAMES'][j]][key].close()


# Save undistorted 2D COMs and their 3D triangulations
f = open(os.path.join(RESULTSDIR, 'COM_undistorted.pickle'), 'wb')
cPickle.dump(save_data, f)
f.close()

# Clean up checkpoints
os.remove('allCOMs_distorted.mat')
os.remove('save_data.pickle')

# Also save a COM3D_undistorted.mat file.
# We need to remove the eID in front of all the keys in datadict
# for prepare_COM to run properly
datadict = {}
for key in datadict_:
    datadict[int(key.split('_')[-1])] = datadict_[key]

_, com3d_dict = serve_data_DANNCE.prepare_COM(
    os.path.join(RESULTSDIR, 'COM_undistorted.pickle'),
    datadict,
    comthresh=0,
    weighted=False,
    retriangulate=False,
    camera_mats=cameras,
    method='median',
    allcams=False)

cfilename = os.path.join(RESULTSDIR, 'COM3D_undistorted.mat')
print("Saving 3D COM to {}".format(cfilename))
samples_keys = list(com3d_dict.keys())

c3d = np.zeros((len(samples_keys), 3))
for i in range(len(samples_keys)):
    c3d[i] = com3d_dict[samples_keys[i]]

# optionally, smooth with median filter
if 'MEDFILT_WINDOW' in params:
    #Make window size odd if not odd
    if params['MEDFILT_WINDOW'] % 2 == 0:
        params['MEDFILT_WINDOW'] += 1
        print("MEDFILT_WINDOW was not odd, changing to: {}".format(params['MEDFILT_WINDOW']))

    c3d_med = medfilt(c3d, (params['MEDFILT_WINDOW'], 1))

    sio.savemat(cfilename.split('.mat')[0] + '_medfilt.mat',
                {'sampleID': samples_keys, 'com': c3d_med})

sio.savemat(cfilename, {'sampleID': samples_keys, 'com': c3d})

print('done!')
