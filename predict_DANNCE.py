"""Runs DANNCE over videos to predict keypoints.

Usage: python predict_DANNCE.py settings_config
"""
import os
import sys
import numpy as np
import time

import tensorflow as tf
import tensorflow.keras.losses as keras_losses

from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
from dannce.engine import losses
from dannce.engine import nets
import dannce.engine.serve_data_DANNCE as serve_data
import dannce.engine.processing as processing
import dannce.engine.ops as ops
from dannce.engine.processing import savedata_tomat, savedata_expval
from dannce.engine.generator import DataGenerator_3Dconv
from dannce.engine.generator import DataGenerator_3Dconv_torch
from dannce.engine.generator import DataGenerator_3Dconv_tf

import scipy.io as sio

# Set up parameters
base_params = processing.read_config(sys.argv[1])
base_params = processing.make_paths_safe(base_params)
dannce_params = processing.read_config(base_params['DANNCE_CONFIG'])
dannce_params = processing.make_paths_safe(dannce_params)
dannce_params = processing.inherit_config(dannce_params, base_params, list(base_params.keys()))

# Load the appropriate loss function and network
try:
    dannce_params['loss'] = getattr(losses, dannce_params['loss'])
except AttributeError:
    dannce_params['loss'] = getattr(keras_losses, dannce_params['loss'])
netname = dannce_params['net']
dannce_params['net'] = getattr(nets, dannce_params['net'])

# While we can use experiment files for DANNCE training, 
# for prediction we use the base data files present in the main config
exp_file = processing.grab_predict_exp_file()
exp = processing.read_config(exp_file)
exp = processing.inherit_config(exp, dannce_params, list(dannce_params.keys()))
for k in ['COM3D_DICT','COMfilename', 'datadir', 'viddir', 'CALIBDIR']:
    if k in exp:
        exp[k] = os.path.join(exp['base_exp_folder'], exp[k])
exp['datadir'] = dannce_params['datadir']
exp['datafile'] = dannce_params['datafile']
dannce_params = exp
dannce_params['experiment'] = exp


RESULTSDIR = dannce_params['RESULTSDIR_PREDICT']
print(RESULTSDIR)

if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)

# default to slow numpy backend if there is no rpedict_mode in config file. I.e. legacy support
predict_mode = dannce_params['predict_mode'] if 'predict_mode' in dannce_params.keys() \
                    else 'numpy'
print("Using {} predict mode".format(predict_mode))

# Copy the configs into the RESULTSDIR, for reproducibility
processing.copy_config(RESULTSDIR, sys.argv[1],
                        base_params['DANNCE_CONFIG'],
                        base_params['COM_CONFIG'])

# Default to 6 views but a smaller number of views can be specified in the DANNCE config.
# If the legnth of the camera files list is smaller than _N_VIEWS, relevant lists will be
# duplicated in order to match _N_VIEWS, if possible.
_N_VIEWS = int(dannce_params['_N_VIEWS'] if '_N_VIEWS' in dannce_params.keys() else 6)


os.environ["CUDA_VISIBLE_DEVICES"] =  dannce_params['gpuID']
gpuID = dannce_params['gpuID']

# If len(dannce_params['experiment']['CAMNAMES']) divides evenly into 6, duplicate here,
# Unless the network was "hard" trained to use less than 6 cameras
if 'hard_train' in base_params.keys() and base_params['hard_train']:
    print("Not duplicating camnames, datafiles, and calib files")
else:
    dupes = ['CAMNAMES', 'datafile', 'calib_file']
    for d in dupes:
        val = dannce_params['experiment'][d]
        if _N_VIEWS % len(val) == 0:
            num_reps = _N_VIEWS // len(val)
            dannce_params['experiment'][d] = val * num_reps
        else:
            raise Exception("The length of the {} list must divide evenly into {}.".format(d, _N_VIEWS))

samples_, datadict_, datadict_3d_, data_3d_, cameras_ = \
    serve_data.prepare_data(dannce_params['experiment'])
if 'allcams' in dannce_params.keys() and dannce_params['allcams']:
    # This code allows a COM estimate from e.g. 6 cameras to be used when running 
    # DANNCE with 3 cameras
    # Primarily used for debugging
    # Make sure all cameras in debug fields are added, so that COM predictions
    # can be more stable
    dcameras_ = {}
    for i in range(len(dannce_params['dCAMNAMES'])):
        test = sio.loadmat(
            os.path.join(dannce_params['dCALIBDIR'], dannce_params['dcalib_file'][i]))
        dcameras_[dannce_params['dCAMNAMES'][i]] = {
            'K': test['K'], 'R': test['r'], 't': test['t']}
        if 'RDistort' in list(test.keys()):
            # Added Distortion params on Dec. 19 2018
            dcameras_[dannce_params['dCAMNAMES'][i]]['RDistort'] = test['RDistort']
            dcameras_[dannce_params['dCAMNAMES'][i]]['TDistort'] = test['TDistort']


if 'COM3D_DICT' not in dannce_params.keys():

    # Load in the COM file at the default location, or use one in the config file if provided
    if 'COMfilename' in dannce_params.keys():
        comfn = dannce_params['COMfilename']
    else:
        comfn = os.path.join('.', 'COM', 'predict_results')
        comfn = os.listdir(comfn)
        comfn = [f for f in comfn if 'COM_undistorted.pickle' in f]
        comfn = os.path.join('.', 'COM', 'predict_results', comfn[0])

    datadict_, com3d_dict_ = serve_data.prepare_COM(
        comfn,
        datadict_,
        comthresh=dannce_params['comthresh'],
        weighted=dannce_params['weighted'],
        retriangulate=dannce_params['retriangulate'] if 'retriangulate' in dannce_params.keys() else True,
        camera_mats=dcameras_ if 'allcams' in dannce_params.keys() and dannce_params['allcams'] else cameras_,
        method=dannce_params['com_method'],
        allcams=dannce_params['allcams'] if 'allcams' in dannce_params.keys() and dannce_params['allcams'] else False)

    # Need to cap this at the number of samples included in our
    # COM finding estimates
    tff = list(com3d_dict_.keys())
    samples_ = samples_[:len(tff)]
    data_3d_ = data_3d_[:len(tff)]
    pre = len(samples_)
    samples_, data_3d_ = \
        serve_data.remove_samples_com(samples_, data_3d_, com3d_dict_, rmc=True, cthresh=dannce_params['cthresh'])
    msg = "Detected {} bad COMs and removed the associated frames from the dataset"
    print(msg.format(pre - len(samples_)))

else:

    print("Loading 3D COM and samples from file: {}".format(exp['COM3D_DICT']))
    c3dfile = sio.loadmat(exp['COM3D_DICT'])
    c3d = c3dfile['com']
    c3dsi = np.squeeze(c3dfile['sampleID'])
    com3d_dict_ = {}
    for (i, s) in enumerate(c3dsi):
        com3d_dict_[s] = c3d[i]

    #verify all of these samples are in datadict_, which we require in order to get the frames IDs
    # for the videos
    assert (set(c3dsi) & set(list(datadict_.keys()))) == set(list(datadict_.keys()))

# Write 3D COM to file
cfilename = os.path.join(RESULTSDIR,'COM3D_undistorted.mat')
print("Saving 3D COM to {}".format(cfilename))
c3d = np.zeros((len(samples_), 3))
for i in range(len(samples_)):
    c3d[i] = com3d_dict_[samples_[i]]
sio.savemat(cfilename, {'sampleID': samples_, 'com': c3d})

# The library is configured to be able to train over multiple animals ("experiments")
# at once. Because supporting code expects to see an experiment ID# prepended to
# each of these data keys, we need to add a token experiment ID here.
samples = []
datadict = {}
datadict_3d = {}
com3d_dict = {}
samples, datadict, datadict_3d, com3d_dict = serve_data.add_experiment(
    0, samples, datadict, datadict_3d, com3d_dict,
    samples_, datadict_, datadict_3d_, com3d_dict_)
cameras = {}
cameras[0] = cameras_
camnames = {}
camnames[0] = dannce_params['experiment']['CAMNAMES']
samples = np.array(samples)
# Initialize video dictionary. paths to videos only.
if dannce_params['IMMODE'] == 'vid':
    vids = processing.initialize_vids(dannce_params, datadict, pathonly=True)

# Parameters
valid_params = {
    'dim_in': (dannce_params['CROP_HEIGHT'][1]-dannce_params['CROP_HEIGHT'][0],
               dannce_params['CROP_WIDTH'][1]-dannce_params['CROP_WIDTH'][0]),
    'n_channels_in': dannce_params['N_CHANNELS_IN'],
    'batch_size': dannce_params['BATCH_SIZE'],
    'n_channels_out': dannce_params['N_CHANNELS_OUT'],
    'out_scale': dannce_params['SIGMA'],
    'crop_width': dannce_params['CROP_WIDTH'],
    'crop_height': dannce_params['CROP_HEIGHT'],
    'vmin': dannce_params['VMIN'],
    'vmax': dannce_params['VMAX'],
    'nvox': dannce_params['NVOX'],
    'interp': dannce_params['INTERP'],
    'depth': dannce_params['DEPTH'],
    'channel_combo': dannce_params['CHANNEL_COMBO'],
    'mode': 'coordinates',
    'camnames': camnames,
    'immode': dannce_params['IMMODE'],
    'shuffle': False,
    'rotation': False,
    'vidreaders': vids,
    'distort': dannce_params['DISTORT'],
    'expval': dannce_params['EXPVAL'],
    'crop_im': False,
    'chunks': dannce_params['chunks'],
    'preload': False}

# Datasets
partition = {}
valid_inds = np.arange(len(samples))
partition['valid_sampleIDs'] = samples[valid_inds]
tifdirs = []

# Generators
if predict_mode == 'torch':
    import torch
    device = 'cuda:' + gpuID
    valid_generator = DataGenerator_3Dconv_torch(
        partition['valid_sampleIDs'], datadict, datadict_3d, cameras,
        partition['valid_sampleIDs'], com3d_dict, tifdirs, **valid_params)
elif predict_mode == 'tf':
    device = '/GPU:' + gpuID
    valid_generator = DataGenerator_3Dconv_tf(
        partition['valid_sampleIDs'], datadict, datadict_3d, cameras,
        partition['valid_sampleIDs'], com3d_dict, tifdirs, **valid_params)
else:
    valid_generator = DataGenerator_3Dconv(
        partition['valid_sampleIDs'], datadict, datadict_3d, cameras,
        partition['valid_sampleIDs'], com3d_dict, tifdirs, **valid_params)


# Build net
print("Initializing Network...")

# This requires that the network be saved as a full model, not just weights.
# As a precaution, we import all possible custom objects that could be used
# by a model and thus need declarations

if 'predict_model' in dannce_params.keys():
    mdl_file = dannce_params['predict_model']
else:
    wdir = dannce_params['RESULTSDIR']
    weights = os.listdir(wdir)
    weights = [f for f in weights if '.hdf5' in f]
    weights = sorted(weights,
                     key=lambda x: int(x.split('.')[1].split('-')[0]))
    weights = weights[-1]

    mdl_file = os.path.join(wdir, weights)
    print("Loading model from " + mdl_file)

if netname == 'unet3d_big_tiedfirstlayer_expectedvalue' or \
     'FROM_WEIGHTS' in dannce_params.keys():
    # This network is too "custom" to be loaded in as a full model, until I
    # figure out how to unroll the first tied weights layer
    gridsize = (dannce_params['NVOX'], dannce_params['NVOX'], dannce_params['NVOX'])
    model = dannce_params['net'](dannce_params['loss'],
                                 float(dannce_params['lr']),
                                 dannce_params['N_CHANNELS_IN'] + dannce_params['DEPTH'],
                                 dannce_params['N_CHANNELS_OUT'],
                                 len(camnames[0]),
                                 batch_norm=dannce_params['batch_norm'],
                                 instance_norm=dannce_params['instance_norm'],
                                 include_top=True,
                                 gridsize=gridsize)
    model.load_weights(mdl_file)
else:
    model = load_model(mdl_file, 
                       custom_objects={'ops': ops,
                                       'slice_input': nets.slice_input,
                                       'mask_nan_keep_loss': losses.mask_nan_keep_loss,
                                       'euclidean_distance_3D': losses.euclidean_distance_3D,
                                       'centered_euclidean_distance_3D': losses.centered_euclidean_distance_3D})

# To speed up EXPVAL prediction, rather than doing two forward passes: one for the 3d coordinate
# and one for the probability map, here we splice on a new output layer after
# the softmax on the last convolutional layer
if dannce_params['EXPVAL']:
    from tensorflow.keras.layers import GlobalMaxPooling3D
    o2 = GlobalMaxPooling3D()(model.layers[-3].output)
    model = Model(inputs=[model.layers[0].input, model.layers[-2].input],
        outputs=[model.layers[-1].output, o2])

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
            print('Saving checkpoint at {}th batch'.format(i))
            if dannce_params['EXPVAL']:
                p_n = savedata_expval(
                    RESULTSDIR + 'save_data_AVG.mat',
                    write=True,
                    data=save_data,
                    tcoord=False,
                    num_markers=nchn,
                    pmax=True)
            else:
                p_n = savedata_tomat(
                    RESULTSDIR + 'save_data_MAX.mat',
                    dannce_params['VMIN'],
                    dannce_params['VMAX'],
                    dannce_params['NVOX'],
                    write=True,
                    data=save_data,
                    num_markers=nchn,
                    tcoord=False)

        ims = valid_gen.__getitem__(i)
        pred = model.predict(ims[0])
        
        if dannce_params['EXPVAL']:
            probmap = pred[1]
            pred = pred[0]
            for j in range(pred.shape[0]):
                pred_max = probmap[j]
                sampleID = partition['valid_sampleIDs'][i * pred.shape[0] + j]
                save_data[i * pred.shape[0] + j] = {
                    'pred_max': pred_max,
                    'pred_coord': pred[j],
                    'sampleID': sampleID}
        else:
            if predict_mode == 'torch':
                for j in range(pred.shape[0]):
                    preds = torch.as_tensor(pred[j], dtype = torch.float32, device = device)
                    pred_max = preds.max(0).values.max(0).values.max(0).values
                    pred_total = preds.sum((0,1,2))
                    xcoord, ycoord, zcoord = processing.plot_markers_3d_torch(preds)
                    coord = torch.stack([xcoord,ycoord,zcoord])
                    pred_log = pred_max.log() - pred_total.log()
                    sampleID = partition['valid_sampleIDs'][i*pred.shape[0] + j]

                    save_data[i*pred.shape[0] + j] = {
                        'pred_max': pred_max.cpu().numpy(),
                        'pred_coord': coord.cpu().numpy(),
                        'true_coord_nogrid': ims[1][j],
                        'logmax': pred_log.cpu().numpy(),
                        'sampleID': sampleID}

            elif predict_mode == 'tf':
                # get coords for each map
                with tf.device(device):
                    for j in range(pred.shape[0]):
                        preds = tf.constant(pred[j], dtype = 'float32')
                        pred_max = tf.math.reduce_max(tf.math.reduce_max(tf.math.reduce_max(preds)))
                        pred_total = tf.math.reduce_sum(tf.math.reduce_sum(tf.math.reduce_sum(preds)))
                        xcoord, ycoord, zcoord = processing.plot_markers_3d_tf(preds)
                        coord = tf.stack([xcoord,ycoord,zcoord], axis=0)
                        pred_log = tf.math.log(pred_max) - tf.math.log(pred_total)
                        sampleID = partition['valid_sampleIDs'][i*pred.shape[0] + j]

                        save_data[i*pred.shape[0] + j] = {
                            'pred_max': pred_max.numpy(),
                            'pred_coord': coord.numpy(),
                            'true_coord_nogrid': ims[1][j],
                            'logmax': pred_log.numpy(),
                            'sampleID': sampleID}

            else:
                # get coords for each map
                for j in range(pred.shape[0]):
                    pred_max = np.max(pred[j], axis=(0,1,2))
                    pred_total = np.sum(pred[j], axis=(0,1,2))
                    xcoord, ycoord, zcoord = processing.plot_markers_3d(pred[j,:,:,:,:])
                    coord = np.stack([xcoord,ycoord,zcoord])
                    pred_log = np.log(pred_max) - np.log(pred_total)
                    sampleID = partition['valid_sampleIDs'][i*pred.shape[0] + j]

                    save_data[i*pred.shape[0] + j] = {
                        'pred_max': pred_max,
                        'pred_coord': coord,
                        'true_coord_nogrid': ims[1][j],
                        'logmax': pred_log,
                        'sampleID': sampleID}


max_eval_batch = dannce_params['maxbatch']
print(max_eval_batch)
if max_eval_batch == 'max':
    max_eval_batch = len(valid_generator)

if 'NEW_N_CHANNELS_OUT' in dannce_params.keys():
    nchn = dannce_params['NEW_N_CHANNELS_OUT']
else:
    nchn = dannce_params['N_CHANNELS_OUT']

evaluate_ondemand(0, max_eval_batch, valid_generator)

if dannce_params['EXPVAL']:
    p_n = savedata_expval(
        RESULTSDIR + 'save_data_AVG.mat',
        write=True,
        data=save_data,
        tcoord=False,
        num_markers=nchn,
        pmax=True)
else:
    p_n = savedata_tomat(
        RESULTSDIR + 'save_data_MAX.mat',
        dannce_params['VMIN'],
        dannce_params['VMAX'],
        dannce_params['NVOX'],
        write=True,
        data=save_data,
        num_markers=nchn,
        tcoord=False)

print("done!")
