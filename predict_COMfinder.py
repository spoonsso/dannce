"""Scipt to run COM finding over a single experiment.

Usage: python ./predict_COMfinder path_to_config
"""

import numpy as np
import sys
import dannce.engine.processing as processing
import tensorflow.keras.losses as keras_losses
from dannce.engine import nets
from dannce.engine import losses
import dannce.engine.ops as ops
from dannce.engine.generator_aux import DataGenerator_downsample
import dannce.engine.serve_data_COM as serve_data
import os
import matplotlib
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load in the params
PARENT_PARAMS = processing.read_config(sys.argv[1])
PARENT_PARAMS = processing.make_paths_safe(PARENT_PARAMS)

params = processing.read_config(PARENT_PARAMS['COM_CONFIG'])
params = processing.make_paths_safe(params)
params = processing.inherit_config(params, PARENT_PARAMS, list(PARENT_PARAMS.keys()))


# Load the appropriate loss function and network
try:
    params['loss'] = getattr(losses, params['loss'])
except AttributeError:
    params['loss'] = getattr(keras_losses, params['loss'])
params['net'] = getattr(nets, params['net'])

vid_dir_flag = params['vid_dir_flag']
_N_VIDEO_FRAMES = params['chunks']

os.environ["CUDA_VISIBLE_DEVICES"] = params['gpuID']

# If params['N_CHANNELS_OUT'] is greater than one, we enter a mode in
# which we predict all available labels + the COM
MULTI_MODE = params['N_CHANNELS_OUT'] > 1
params['N_CHANNELS_OUT'] = params['N_CHANNELS_OUT'] + int(MULTI_MODE)

# Inherit required parameters from main config file

# Also add parent params under the 'experiment' key for compatibility
# with DANNCE's video loading function
exp_file = processing.grab_predict_exp_file()
exp = processing.read_config(exp_file)
exp = processing.inherit_config(exp, params, list(params.keys()))
for k in ['datadir', 'viddir', 'CALIBDIR']:
    exp[k] = os.path.join(exp['base_exp_folder'], exp[k])
exp['datadir'] = params['datadir']
exp['datafile'] = params['datafile']
params = exp
params['experiment'] = exp

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
            print("{} samples took {} seconds".format(sample_save,
                                                      time.time() - end_time))
            end_time = time.time()

        if (i - start_ind) % 1000 == 0 and i != start_ind:
            print('Saving checkpoint at {}th sample'.format(i))
            processing.save_COM_checkpoint(save_data,
                                           RESULTSDIR,
                                           datadict_,
                                           cameras,
                                           params)

        pred_ = model.predict(valid_gen.__getitem__(i)[0])

        pred_ = np.reshape(
            pred_,
            [-1, len(params['CAMNAMES']), pred_.shape[1], pred_.shape[2], pred_.shape[3]])

        for m in range(pred_.shape[0]):
            # odd loop condition, but it's because at the end of samples,
            # predict_generator will continue to make predictions in a way I
            # don't grasp yet, but also in a way we should ignore

            # By selecting -1 for the last axis, we get the COM index for a
            # normal COM network, and also the COM index for a multi_mode COM network,
            # as in multimode the COM label is put at the end
            pred = pred_[m, :, :, :, -1]
            sampleID_ = partition['valid'][i*pred_.shape[0] + m]
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
                    im = valid_gen.__getitem__(i*pred_.shape[0] + m)
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

samples, datadict, datadict_3d, cameras, camera_mats = \
    serve_data.prepare_data(params, multimode=MULTI_MODE)

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

# Initialize video dictionary. paths to videos only.
vids = processing.initialize_vids(params, datadict, pathonly=True)

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
    'dsmode': params['dsmode'] if 'dsmode' in params.keys() else 'dsm',
    'preload': False}

partition = {}
partition['valid'] = samples
labels = datadict

save_data = {}

valid_generator = DataGenerator_downsample(
    partition['valid'], labels, vids, **valid_params)

# If we just want to analyze a chunk of video...
st_ind = params['start_sample_index'] if 'start_sample_index' in params.keys() else 0
if params['max_num_samples'] == 'max':
    evaluate_ondemand(st_ind, len(valid_generator), valid_generator)
else:
    endIdx = np.min([st_ind + params['max_num_samples'], len(valid_generator)])
    evaluate_ondemand(st_ind, endIdx, valid_generator)

processing.save_COM_checkpoint(save_data,
                               RESULTSDIR,
                               datadict_,
                               cameras,
                               params)

print('done!')
