"""Runs DANNCE over videos to predict keypoints.

Usage: python predict_DANNCE.py settings_config
"""
import sys
import numpy as np
import os
import time
import keras.backend as K
from dannce.engine import losses
from dannce.engine import nets
import keras.losses
import dannce.engine.serve_data_DANNCE as serve_data
import dannce.engine.processing as processing
import dannce.engine.ops as ops
from dannce.engine.processing import savedata_tomat, savedata_expval
from dannce.engine.generator_kmeans import DataGenerator_3Dconv_kmeans
from keras.layers import Conv3D, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import scipy.io as sio

_N_VIEWS = 6

# Set up parameters
PARENT_PARAMS = processing.read_config(sys.argv[1])
PARENT_PARAMS = processing.make_paths_safe(PARENT_PARAMS)
CONFIG_PARAMS = processing.read_config(PARENT_PARAMS['DANNCE_CONFIG'])
CONFIG_PARAMS = processing.make_paths_safe(CONFIG_PARAMS)

# Load the appropriate loss function and network
try:
    CONFIG_PARAMS['loss'] = getattr(losses, CONFIG_PARAMS['loss'])
except AttributeError:
    CONFIG_PARAMS['loss'] = getattr(keras.losses, CONFIG_PARAMS['loss'])
netname = CONFIG_PARAMS['net']
CONFIG_PARAMS['net'] = getattr(nets, CONFIG_PARAMS['net'])

# While we can use experiment files for DANNCE training, 
# for prediction we use the base data files present in the main config

CONFIG_PARAMS['experiment'] = PARENT_PARAMS
RESULTSDIR = CONFIG_PARAMS['RESULTSDIR_PREDICT']
print(RESULTSDIR)

if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)

# TODO(Devices): Is it necessary to set the device environment?
# This could mess with people's setups.
os.environ["CUDA_VISIBLE_DEVICES"] =  CONFIG_PARAMS['gpuID']

# If len(CONFIG_PARAMS['experiment']['CAMNAMES']) divides evenly into 6, duplicate here
dupes = ['CAMNAMES', 'datafile', 'calib_file']
for d in dupes:
    val = CONFIG_PARAMS['experiment'][d]
    if _N_VIEWS % len(val) == 0:
        num_reps = _N_VIEWS // len(val)
        CONFIG_PARAMS['experiment'][d] = val * num_reps
    else:
        raise Exception("The length of the {} list must divide evenly into {}.".format(d, _N_VIEWS))

samples_, datadict_, datadict_3d_, data_3d_, cameras_ = \
    serve_data.prepare_data(CONFIG_PARAMS['experiment'])

# Load in the COM file at the default location, or use one in the config file if provided
if 'COMfilename' in CONFIG_PARAMS.keys():
    comfn = CONFIG_PARAMS['COMfilename']
else:
    comfn = os.path.join('.', 'COM', 'predict_results')
    comfn = os.listdir(comfn)
    comfn = [f for f in comfn if 'COM_undistorted.pickle' in f]
    comfn = os.path.join('.', 'COM', 'predict_results', comfn[0])

if 'allcams' in CONFIG_PARAMS.keys() and CONFIG_PARAMS['allcams']: # Make sure all cameras in debug fields are added, so that COM predictions
# can be more stable
    dcameras_ = {}
    for i in range(len(CONFIG_PARAMS['dCAMNAMES'])):
        test = sio.loadmat(
            os.path.join(CONFIG_PARAMS['dCALIBDIR'], CONFIG_PARAMS['dcalib_file'][i]))
        dcameras_[CONFIG_PARAMS['dCAMNAMES'][i]] = {
            'K': test['K'], 'R': test['r'], 't': test['t']}
        if 'RDistort' in list(test.keys()):
            # Added Distortion params on Dec. 19 2018
            dcameras_[CONFIG_PARAMS['dCAMNAMES'][i]]['RDistort'] = test['RDistort']
            dcameras_[CONFIG_PARAMS['dCAMNAMES'][i]]['TDistort'] = test['TDistort']

datadict_, com3d_dict_ = serve_data.prepare_COM(
    comfn,
    datadict_,
    comthresh=CONFIG_PARAMS['comthresh'],
    weighted=CONFIG_PARAMS['weighted'],
    retriangulate=True,
    camera_mats=dcameras_ if 'allcams' in CONFIG_PARAMS.keys() and CONFIG_PARAMS['allcams'] else cameras_,
    method=CONFIG_PARAMS['com_method'],
    allcams=CONFIG_PARAMS['allcams'] if 'allcams' in CONFIG_PARAMS.keys() and CONFIG_PARAMS['allcams'] else False)

# Need to cap this at the number of samples included in our
# COM finding estimates
tf = list(com3d_dict_.keys())
samples_ = samples_[:len(tf)]
data_3d_ = data_3d_[:len(tf)]
pre = len(samples_)
samples_, data_3d_ = \
    serve_data.remove_samples_com(samples_, data_3d_, com3d_dict_, rmc=True, cthresh=CONFIG_PARAMS['cthresh'])
msg = "Detected {} bad COMs and removed the associated frames from the dataset"
print(msg.format(pre - len(samples_)))

# Write 3D COM to file
cfilename = os.path.join(RESULTSDIR,'COM3D_undistorted.mat')
print("Saving 3D COM to {}".format(cfilename))
c3d = np.zeros((len(samples_), 3))
for i in range(len(samples_)):
    c3d[i] = com3d_dict_[samples_[i]]
sio.savemat(cfilename, {'sampleID': samples_, 'com': c3d})

# TODO(Comment): Unclear what this section is doing.
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
camnames[0] = CONFIG_PARAMS['experiment']['CAMNAMES']
samples = np.array(samples)

# TODO(Repeated): This motif occurs in predict_COMfinder.
# Consider making a function out of it.
# Initialize video objects
vids = {}
if CONFIG_PARAMS['IMMODE'] == 'vid':
    for i in range(len(CONFIG_PARAMS['experiment']['CAMNAMES'])):
        if CONFIG_PARAMS['vid_dir_flag']:
            addl = ''
        else:
            addl = os.listdir(
                os.path.join(
                    CONFIG_PARAMS['experiment']['viddir'],
                    CONFIG_PARAMS['experiment']['CAMNAMES'][i]))[0]
        vids[CONFIG_PARAMS['experiment']['CAMNAMES'][i]] = \
            processing.generate_readers(
                CONFIG_PARAMS['experiment']['viddir'],
                os.path.join(CONFIG_PARAMS['experiment']['CAMNAMES'][i], addl),
                minopt=0,
                maxopt=1,
                extension=CONFIG_PARAMS['experiment']['extension'])

# Set framecnt according to the "chunks" config param
framecnt = CONFIG_PARAMS['chunks']

# Parameters
valid_params = {
    'dim_in': (CONFIG_PARAMS['CROP_HEIGHT'][1]-CONFIG_PARAMS['CROP_HEIGHT'][0],
               CONFIG_PARAMS['CROP_WIDTH'][1]-CONFIG_PARAMS['CROP_WIDTH'][0]),
    'n_channels_in': CONFIG_PARAMS['N_CHANNELS_IN'],
    'batch_size': CONFIG_PARAMS['BATCH_SIZE'],
    'n_channels_out': CONFIG_PARAMS['N_CHANNELS_OUT'],
    'out_scale': CONFIG_PARAMS['SIGMA'],
    'crop_width': CONFIG_PARAMS['CROP_WIDTH'],
    'crop_height': CONFIG_PARAMS['CROP_HEIGHT'],
    'vmin': CONFIG_PARAMS['VMIN'],
    'vmax': CONFIG_PARAMS['VMAX'],
    'nvox': CONFIG_PARAMS['NVOX'],
    'interp': CONFIG_PARAMS['INTERP'],
    'depth': CONFIG_PARAMS['DEPTH'],
    'channel_combo': CONFIG_PARAMS['CHANNEL_COMBO'],
    'mode': 'coordinates',
    'camnames': camnames,
    'immode': CONFIG_PARAMS['IMMODE'],
    'training': False,
    'shuffle': False,
    'rotation': False,
    'pregrid': None,
    'pre_projgrid': None,
    'stamp': False,
    'vidreaders': vids,
    'distort': CONFIG_PARAMS['DISTORT'],
    'expval': CONFIG_PARAMS['EXPVAL'],
    'crop_im': False,
    'chunks': CONFIG_PARAMS['chunks']}

# Datasets
partition = {}
valid_inds = np.arange(len(samples))
partition['valid_sampleIDs'] = samples[valid_inds]
tifdirs = []

# Generators
valid_generator = DataGenerator_3Dconv_kmeans(
    partition['valid_sampleIDs'], datadict, datadict_3d, cameras,
    partition['valid_sampleIDs'], com3d_dict, tifdirs, **valid_params)

# Build net
print("Initializing Network...")

# This requires that the network be saved as a full model, not just weights.
# As a precaution, we import all possible custom objects that could be used
# by a model and thus need declarations

if 'predict_model' in CONFIG_PARAMS.keys():
    mdl_file = CONFIG_PARAMS['predict_model']
else:
    wdir = os.path.join('.', 'DANNCE', 'train_results')
    weights = os.listdir(wdir)
    weights = [f for f in weights if '.hdf5' in f]
    weights = sorted(weights,
                     key=lambda x: int(x.split('.')[1].split('-')[0]))
    weights = weights[-1]

    mdl_file = os.path.join(wdir, weights)
    print("Loading model from " + mdl_file)

if netname == 'unet3d_big_tiedfirstlayer_expectedvalue' or \
     'FROM_WEIGHTS' in CONFIG_PARAMS.keys():
    # This network is too "custom" to be loaded in as a full model, until I
    # figure out how to unroll the first tied weights layer
    gridsize = (CONFIG_PARAMS['NVOX'], CONFIG_PARAMS['NVOX'], CONFIG_PARAMS['NVOX'])
    model = CONFIG_PARAMS['net'](CONFIG_PARAMS['loss'],
                                 float(CONFIG_PARAMS['lr']),
                                 CONFIG_PARAMS['N_CHANNELS_IN'] + CONFIG_PARAMS['DEPTH'],
                                 CONFIG_PARAMS['N_CHANNELS_OUT'],
                                 len(camnames[0]),
                                 batch_norm=CONFIG_PARAMS['batch_norm'],
                                 instance_norm=CONFIG_PARAMS['instance_norm'],
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

save_data = {}


def evaluate_ondemand(start_ind, end_ind, valid_gen, vids):
    """Evaluate experiment.

    :param start_ind: Starting frame
    :param end_ind: Ending frame
    :param valid_gen: Generator
    """
    end_time = time.time()
    lastvid_ = '0' + CONFIG_PARAMS['experiment']['extension']
    currvid_ = 0
    for i in range(start_ind, end_ind):
        print("Predicting on batch {}".format(i))
        if (i - start_ind) % 100 == 0 and i != start_ind:
            print(i)
            print("100 batches took {} seconds".format(time.time() - end_time))
            end_time = time.time()

        if (i - start_ind) % 1000 == 0 and i != start_ind:
            print('Saving checkpoint at {}th batch'.format(i))
            if CONFIG_PARAMS['EXPVAL']:
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
                    CONFIG_PARAMS['VMIN'],
                    CONFIG_PARAMS['VMAX'],
                    CONFIG_PARAMS['NVOX'],
                    write=True,
                    data=save_data,
                    num_markers=nchn,
                    tcoord=False)                
                pass


        # TODO(Repeated2): This motif is also repeated, consider modularizing
        if framecnt is not None:
            # We can't keep all these videos open, so close the ones
            # that are not needed
            # TODO(datadict class): Can we build classes that encapsulate these
            # values to make it easier to access without the
            # use of nested lists?
            currentframes = datadict[
                partition['valid_sampleIDs'][
                    i * CONFIG_PARAMS['BATCH_SIZE']]]['frames']
            m = min(
                [i * CONFIG_PARAMS['BATCH_SIZE'] + CONFIG_PARAMS['BATCH_SIZE'],
                 len(partition['valid_sampleIDs']) - 1]
            )

            curr_frames_max = \
                datadict[partition['valid_sampleIDs'][m]]['frames']
            maxframes = max(list(curr_frames_max.values()))
            currentframes = min(list(currentframes.values()))
            lastvid = str(currentframes // framecnt * framecnt - framecnt) \
                + CONFIG_PARAMS['experiment']['extension']
            currvid = maxframes // framecnt * framecnt

            vids_, lastvid_, currvid_ = processing.close_open_vids(
                lastvid, lastvid_,
                currvid,
                currvid_,
                framecnt,
                CONFIG_PARAMS['experiment']['CAMNAMES'],
                vids,
                CONFIG_PARAMS['vid_dir_flag'],
                CONFIG_PARAMS['experiment']['viddir'],
                currentframes,
                maxframes)
            valid_gen.vidreaders = vids_
            vids = vids_

        ims = valid_gen.__getitem__(i)
        pred = model.predict(ims[0])

        if CONFIG_PARAMS['EXPVAL']:
            probmap = get_output([ims[0][0], 0])[0]
            for j in range(pred.shape[0]):
                pred_max = np.max(np.max(np.max(
                    probmap[j], axis=0), axis=0), axis=0)
                sampleID = partition['valid_sampleIDs'][i * pred.shape[0] + j]
                save_data[i * pred.shape[0] + j] = {
                    'pred_max': pred_max,
                    'pred_coord': pred[j],
                    'sampleID': sampleID}
        else:
            # get coords for each map
            for j in range(pred.shape[0]):
                pred_max = np.max(np.max(np.max(
                    pred[j, :, :, :, :], axis=0), axis=0), axis=0)
                pred_total = np.sum(np.sum(np.sum(
                    pred[j, :, :, :, :], axis=0), axis=0), axis=0)
                coordx, coordy, coordz = processing.plot_markers_3d(pred[j])
                coord = np.stack((coordx, coordy, coordz))
                sampleID = partition['valid_sampleIDs'][i * pred.shape[0] + j]

                # TODO(pred_max_check): The saved value is pred_max_0,
                # which is undefined. changed to pred_max
                save_data[i * pred.shape[0] + j] = {
                    'pred_max': pred_max,
                    'pred_coord': coord,
                    'true_coord_nogrid': ims[1][j],
                    'logmax': np.log(pred_max) - np.log(pred_total),
                    'sampleID': sampleID}

max_eval_batch = CONFIG_PARAMS['maxbatch']
if max_eval_batch == 'max':
    max_eval_batch = len(valid_generator)

if 'NEW_N_CHANNELS_OUT' in CONFIG_PARAMS.keys():
    nchn = CONFIG_PARAMS['NEW_N_CHANNELS_OUT']
else:
    nchn = CONFIG_PARAMS['N_CHANNELS_OUT']

if CONFIG_PARAMS['EXPVAL']:
    get_output = K.function(
        [model.layers[0].input, K.learning_phase()], [model.layers[-3].output])
    evaluate_ondemand(0, max_eval_batch, valid_generator, vids)

    p_n = savedata_expval(
        RESULTSDIR + 'save_data_AVG.mat',
        write=True,
        data=save_data,
        tcoord=False,
        num_markers=nchn,
        pmax=True)
else:
    evaluate_ondemand(0, max_eval_batch, valid_generator, vids)

    p_n = savedata_tomat(
        RESULTSDIR + 'save_data_MAX.mat',
        CONFIG_PARAMS['VMIN'],
        CONFIG_PARAMS['VMAX'],
        CONFIG_PARAMS['NVOX'],
        write=True,
        data=save_data,
        num_markers=nchn,
        tcoord=False)

print("done!")
