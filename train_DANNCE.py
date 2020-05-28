"""
Trains DANNCE. Currently only supports fine-tuning with hand-labeled data.
That is, this code loads all volumes and labels into memory before training.
This memory-based training is only available when using a small number of
hand-labeled frames. Code for training over chronic recordings / motion
capture to come.

Usage: python train_DANNCE.py settings_config 

In contrast to predict_DANNCE.py, train_DANNCE.py can process multiple
experiment config files  to support training over multiple animals.
"""
import sys
import numpy as np
import os
from copy import deepcopy
import dannce.engine.serve_data_DANNCE as serve_data
import dannce.engine.processing as processing
from dannce.engine.generator import DataGenerator_3Dconv
from dannce.engine.generator import DataGenerator_3Dconv_frommem
from dannce.engine import nets
from dannce.engine import losses
from dannce.engine import ops
from six.moves import cPickle

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard

import scipy.io as sio
import tensorflow.keras as keras

# Set up parameters
PARENT_PARAMS = processing.read_config(sys.argv[1])
PARENT_PARAMS = processing.make_paths_safe(PARENT_PARAMS)

CONFIG_PARAMS = processing.read_config(PARENT_PARAMS['DANNCE_CONFIG'])
CONFIG_PARAMS = processing.make_paths_safe(CONFIG_PARAMS)

CONFIG_PARAMS['loss'] = getattr(losses, CONFIG_PARAMS['loss'])
CONFIG_PARAMS['net'] = getattr(nets, CONFIG_PARAMS['net'])

# Default to 6 views but a smaller number of views can be specified in the DANNCE config.
# If the legnth of the camera files list is smaller than _N_VIEWS, relevant lists will be
# duplicated in order to match _N_VIEWS, if possible.
_N_VIEWS = int(CONFIG_PARAMS['_N_VIEWS'] if '_N_VIEWS' in CONFIG_PARAMS.keys() else 6)

# Convert all metric strings to objects
metrics = []
for m in CONFIG_PARAMS['metric']:
    try:
        m_obj = getattr(losses, m)
    except AttributeError:
        m_obj = getattr(keras.losses, m)
    metrics.append(m_obj)

# set GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG_PARAMS['gpuID']

# find the weights given config path
if CONFIG_PARAMS['weights'] != 'None':
    weights = os.listdir(CONFIG_PARAMS['weights'])
    weights = [f for f in weights if '.hdf5' in f]
    weights = weights[0]

    CONFIG_PARAMS['weights'] = os.path.join(CONFIG_PARAMS['weights'],weights)

    print("Fine-tuning from {}".format(CONFIG_PARAMS['weights']))

samples = []
datadict = {}
datadict_3d = {}
com3d_dict = {}
cameras = {}
camnames = {}

exps = processing.grab_exp_file(CONFIG_PARAMS)

num_experiments = len(exps)
CONFIG_PARAMS['experiment'] = {}
for e in range(num_experiments):
    CONFIG_PARAMS['experiment'][e] = processing.read_config(exps[e])
    CONFIG_PARAMS['experiment'][e] = processing.make_paths_safe(CONFIG_PARAMS['experiment'][e])

    CONFIG_PARAMS['experiment'][e] = \
    processing.inherit_config(CONFIG_PARAMS['experiment'][e],
                              PARENT_PARAMS,
                              ['CAMNAMES',
                               'CALIBDIR',
                               'calib_file',
                               'extension',
                               'datafile'])

    if 'hard_train' in PARENT_PARAMS.keys() and PARENT_PARAMS['hard_train']:
        print("Not duplicating camnames, datafiles, and calib files")
    else:
        # If len(CONFIG_PARAMS['experiment'][e]['CAMNAMES']) divides evenly into 6, duplicate here
        dupes = ['CAMNAMES', 'datafile', 'calib_file']
        for d in dupes:
            val = CONFIG_PARAMS['experiment'][e][d]
            if _N_VIEWS % len(val) == 0:
                num_reps = _N_VIEWS // len(val)
                CONFIG_PARAMS['experiment'][e][d] = val * num_reps
            else:
                raise Exception("The length of the {} list must divide evenly into {}.".format(d, _N_VIEWS))

    samples_, datadict_, datadict_3d_, data_3d_, cameras_ = \
        serve_data.prepare_data(CONFIG_PARAMS['experiment'][e])
    
    # New option: if there is "clean" data (full marker set), can take the
    # 3D COM from the labels
    if 'COM_fromlabels' in CONFIG_PARAMS['experiment'][e].keys() \
      and CONFIG_PARAMS['experiment'][e]['COM_fromlabels']:
        print("For experiment {}, calculating 3D COM from labels".format(e))
        com3d_dict_ = deepcopy(datadict_3d_)
        for key in com3d_dict_.keys():
            com3d_dict_[key] = np.nanmean(datadict_3d_[key],axis=1,keepdims=True) 
    else: # then do traditional COM and sample alignment
        if 'COM3D_DICT' not in CONFIG_PARAMS['experiment'][e].keys():
            if 'COMfilename' not in CONFIG_PARAMS['experiment'][e].keys():
                raise Exception("The COMfilename or COM3D_DICT field must be populated in the",
                 "yaml for experiment {}".format(e))

            comfn = CONFIG_PARAMS['experiment'][e]['COMfilename']

            datadict_, com3d_dict_ = serve_data.prepare_COM(
                comfn,
                datadict_,
                comthresh=CONFIG_PARAMS['comthresh'],
                weighted=CONFIG_PARAMS['weighted'],
                retriangulate=CONFIG_PARAMS['retriangulate'] if 'retriangulate' in CONFIG_PARAMS.keys() else True,
                camera_mats=cameras_,
                method=CONFIG_PARAMS['com_method'])

            # Need to cap this at the number of samples included in our
            # COM finding estimates

            tff = list(com3d_dict_.keys())
            samples_ = samples_[:len(tff)]
            data_3d_ = data_3d_[:len(tff)]
            pre = len(samples_)
            samples_, data_3d_ = \
                serve_data.remove_samples_com(samples_, data_3d_, com3d_dict_, rmc=True, cthresh=CONFIG_PARAMS['cthresh'])
            msg = "Detected {} bad COMs and removed the associated frames from the dataset"
            print(msg.format(pre - len(samples_)))

        else:
            print("Loading 3D COM and samples from file: {}".
                format(CONFIG_PARAMS['experiment'][e]['COM3D_DICT']))
            c3dfile = sio.loadmat(CONFIG_PARAMS['experiment'][e]['COM3D_DICT'])
            c3d = c3dfile['com']
            c3dsi = np.squeeze(c3dfile['sampleID'])
            com3d_dict_ = {}
            for (i, s) in enumerate(c3dsi):
                com3d_dict_[s] = c3d[i]

            #verify all of the datadict_ keys are in this sample set
            assert (set(c3dsi) & set(list(datadict_.keys()))) == set(list(datadict_.keys()))

    print("Using {} samples total.".format(len(samples_)))
    
    samples, datadict, datadict_3d, com3d_dict = serve_data.add_experiment(
        e, samples, datadict, datadict_3d, com3d_dict,
        samples_, datadict_, datadict_3d_, com3d_dict_)

    cameras[e] = cameras_
    camnames[e] = CONFIG_PARAMS['experiment'][e]['CAMNAMES']
    print("Using the following cameras: {}".format(camnames[e]))

RESULTSDIR = CONFIG_PARAMS['RESULTSDIR']
print(RESULTSDIR)

if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)


# Additionally, to keep videos unique across experiments, need to add
# experiment labels in other places. E.g. experiment 0 CameraE's "camname"
# Becomes 0_CameraE.
cameras, datadict, CONFIG_PARAMS = serve_data.prepend_experiment(CONFIG_PARAMS, datadict,
                                                  num_experiments, camnames, cameras)

samples = np.array(samples)

# Initialize video objects
vids = {}
for e in range(num_experiments):
    if CONFIG_PARAMS['IMMODE'] == 'vid':
        vids = processing.initialize_vids_train(CONFIG_PARAMS, datadict, e,
                                                vids, pathonly=True)

# Parameters
if CONFIG_PARAMS['EXPVAL']:
    outmode = 'coordinates'
else:
    outmode = '3dprob'

gridsize = (CONFIG_PARAMS['NVOX'], CONFIG_PARAMS['NVOX'], CONFIG_PARAMS['NVOX'])

# When this true, the data generator will shuffle the cameras and then select the first 3,
# to feed to a native 3 camera model
if 'cam3_train' in CONFIG_PARAMS.keys() and CONFIG_PARAMS['cam3_train']:
    cam3_train = True
else:
    cam3_train = False

valid_params = {
    'dim_in': (CONFIG_PARAMS['CROP_HEIGHT'][1]-CONFIG_PARAMS['CROP_HEIGHT'][0],
               CONFIG_PARAMS['CROP_WIDTH'][1]-CONFIG_PARAMS['CROP_WIDTH'][0]),
    'n_channels_in': CONFIG_PARAMS['N_CHANNELS_IN'],
    'batch_size': 1,
    'n_channels_out': CONFIG_PARAMS['NEW_N_CHANNELS_OUT'],
    'out_scale': CONFIG_PARAMS['SIGMA'],
    'crop_width': CONFIG_PARAMS['CROP_WIDTH'],
    'crop_height': CONFIG_PARAMS['CROP_HEIGHT'],
    'vmin': CONFIG_PARAMS['VMIN'],
    'vmax': CONFIG_PARAMS['VMAX'],
    'nvox': CONFIG_PARAMS['NVOX'],
    'interp': CONFIG_PARAMS['INTERP'],
    'depth': CONFIG_PARAMS['DEPTH'],
    'channel_combo': CONFIG_PARAMS['CHANNEL_COMBO'],
    'mode': outmode,
    'camnames': camnames,
    'immode': CONFIG_PARAMS['IMMODE'],
    'shuffle': False,  # We will shuffle later
    'rotation': False,  # We will rotate later if desired
    'vidreaders': vids,
    'distort': CONFIG_PARAMS['DISTORT'],
    'expval': CONFIG_PARAMS['EXPVAL'],
    'crop_im': False,
    'chunks': CONFIG_PARAMS['chunks'],
    'preload': False}

# Setup a generator that will read videos and labels
tifdirs = []  # Training from single images not yet supported in this demo

partition = {}
if 'load_valid' not in CONFIG_PARAMS.keys():
    all_inds = np.arange(len(samples))

    # extract random inds from each set for validation
    v = CONFIG_PARAMS['num_validation_per_exp']
    valid_inds = []

    if CONFIG_PARAMS['num_validation_per_exp'] > 0: #if 0, do not perform validation
      for e in range(num_experiments):
          tinds = [i for i in range(len(samples))
                   if int(samples[i].split('_')[0]) == e]
          valid_inds = valid_inds + list(np.random.choice(tinds,
                                                          (v,), replace=False))

    train_inds = [i for i in all_inds if i not in valid_inds]

    assert (set(valid_inds) & set(train_inds)) == set()

    partition['valid_sampleIDs'] = samples[valid_inds]
    partition['train_sampleIDs'] = samples[train_inds]

    # Save train/val inds
    with open(RESULTSDIR + 'val_samples.pickle', 'wb') as f:
        cPickle.dump(partition['valid_sampleIDs'], f)

    with open(RESULTSDIR + 'train_samples.pickle', 'wb') as f:
        cPickle.dump(partition['train_sampleIDs'], f)
else:
    # Load validation samples from elsewhere
    with open(os.path.join(CONFIG_PARAMS['load_valid'], 'val_samples.pickle'),
              'rb') as f:
        partition['valid_sampleIDs'] = cPickle.load(f)
    partition['train_sampleIDs'] = [f for f in samples if f not in partition['valid_sampleIDs']]

train_generator = DataGenerator_3Dconv(partition['train_sampleIDs'],
                                              datadict,
                                              datadict_3d,
                                              cameras,
                                              partition['train_sampleIDs'],
                                              com3d_dict,
                                              tifdirs,
                                              **valid_params)
valid_generator = DataGenerator_3Dconv(partition['valid_sampleIDs'],
                                              datadict,
                                              datadict_3d,
                                              cameras,
                                              partition['valid_sampleIDs'],
                                              com3d_dict,
                                              tifdirs,
                                              **valid_params)

# We should be able to load everything into memory...
X_train = np.zeros((len(partition['train_sampleIDs']),
                    CONFIG_PARAMS['NVOX'],
                    CONFIG_PARAMS['NVOX'],
                    CONFIG_PARAMS['NVOX'],
                    CONFIG_PARAMS['N_CHANNELS_IN']*len(camnames[0])),
                   dtype='float32')

X_valid = np.zeros((len(partition['valid_sampleIDs']),
                    CONFIG_PARAMS['NVOX'],
                    CONFIG_PARAMS['NVOX'],
                    CONFIG_PARAMS['NVOX'],
                    CONFIG_PARAMS['N_CHANNELS_IN']*len(camnames[0])),
                   dtype='float32')

X_train_grid = None
X_valid_grid = None
if CONFIG_PARAMS['EXPVAL']:
    y_train = np.zeros((len(partition['train_sampleIDs']),
                        3,
                        CONFIG_PARAMS['NEW_N_CHANNELS_OUT']),
                       dtype='float32')
    X_train_grid = np.zeros((len(partition['train_sampleIDs']),
                             CONFIG_PARAMS['NVOX']**3, 3),
                            dtype='float32')

    y_valid = np.zeros((len(partition['valid_sampleIDs']),
                        3,
                        CONFIG_PARAMS['NEW_N_CHANNELS_OUT']),
                       dtype='float32')
    X_valid_grid = np.zeros((len(partition['valid_sampleIDs']),
                             CONFIG_PARAMS['NVOX']**3, 3),
                            dtype='float32')
else:
    y_train = np.zeros((len(partition['train_sampleIDs']),
                        CONFIG_PARAMS['NVOX'],
                        CONFIG_PARAMS['NVOX'],
                        CONFIG_PARAMS['NVOX'],
                        CONFIG_PARAMS['NEW_N_CHANNELS_OUT']),
                       dtype='float32')

    y_valid = np.zeros((len(partition['valid_sampleIDs']),
                        CONFIG_PARAMS['NVOX'],
                        CONFIG_PARAMS['NVOX'],
                        CONFIG_PARAMS['NVOX'],
                        CONFIG_PARAMS['NEW_N_CHANNELS_OUT']),
                       dtype='float32')


print("Loading training data into memory. This can take a while to seek through",
        "large sets of video. This process is much faster if the frame indices",
        "are sorted in ascending order in your label data file.")
for i in range(len(partition['train_sampleIDs'])):
    print(i, end='\r')
    rr = train_generator.__getitem__(i)
    if CONFIG_PARAMS['EXPVAL']:
        X_train[i] = rr[0][0]
        X_train_grid[i] = rr[0][1]
    else:
        X_train[i] = rr[0]
    y_train[i] = rr[1]

print("Loading validation data into memory")
for i in range(len(partition['valid_sampleIDs'])):
    print(i, end='\r')
    rr = valid_generator.__getitem__(i)
    if CONFIG_PARAMS['EXPVAL']:
        X_valid[i] = rr[0][0]
        X_valid_grid[i] = rr[0][1]
    else:
        X_valid[i] = rr[0]
    y_valid[i] = rr[1]

# Now we can generate from memory with shuffling, rotation, etc.
if CONFIG_PARAMS['CHANNEL_COMBO'] == 'random':
    randflag = True
else:
    randflag = False

train_generator = \
    DataGenerator_3Dconv_frommem(np.arange(len(partition['train_sampleIDs'])),
                                 X_train,
                                 y_train,
                                 batch_size=CONFIG_PARAMS['BATCH_SIZE'],
                                 random=randflag,
                                 rotation=CONFIG_PARAMS['ROTATE'],
                                 expval=CONFIG_PARAMS['EXPVAL'],
                                 xgrid=X_train_grid,
                                 nvox=CONFIG_PARAMS['NVOX'],
                                 cam3_train=cam3_train)
valid_generator = \
    DataGenerator_3Dconv_frommem(np.arange(len(partition['valid_sampleIDs'])),
                                 X_valid,
                                 y_valid,
                                 batch_size=1,
                                 random=randflag,
                                 rotation=False,
                                 expval=CONFIG_PARAMS['EXPVAL'],
                                 xgrid=X_valid_grid,
                                 nvox=CONFIG_PARAMS['NVOX'],
                                 shuffle=False,
                                 cam3_train=cam3_train)

# Build net
print("Initializing Network...")

assert not (CONFIG_PARAMS['batch_norm'] == True) & (CONFIG_PARAMS['instance_norm'] == True)

# Currently, we expect four modes of use:
# 1) Training a new network from scratch
# 2) Fine-tuning a network trained on a diff. dataset (transfer learning)
# 3) Continuing to train 1) or 2) from a full model checkpoint (including optimizer state)

print("NUM CAMERAS: {}".format(len(camnames[0])))

if CONFIG_PARAMS['train_mode'] == 'new':
    model = CONFIG_PARAMS['net'](CONFIG_PARAMS['loss'],
                                 float(CONFIG_PARAMS['lr']),
                                 CONFIG_PARAMS['N_CHANNELS_IN'] + CONFIG_PARAMS['DEPTH'],
                                 CONFIG_PARAMS['N_CHANNELS_OUT'],
                                 len(camnames[0]),
                                 batch_norm=CONFIG_PARAMS['batch_norm'],
                                 instance_norm=CONFIG_PARAMS['instance_norm'],
                                 include_top=True,
                                 gridsize=gridsize)
elif CONFIG_PARAMS['train_mode'] == 'finetune':
    model = CONFIG_PARAMS['net'](CONFIG_PARAMS['loss'],
                                 float(CONFIG_PARAMS['lr']),
                                 CONFIG_PARAMS['N_CHANNELS_IN'] + CONFIG_PARAMS['DEPTH'],
                                 CONFIG_PARAMS['N_CHANNELS_OUT'],
                                 len(camnames[0]),
                                 CONFIG_PARAMS['NEW_LAST_KERNEL_SIZE'],
                                 CONFIG_PARAMS['NEW_N_CHANNELS_OUT'],
                                 CONFIG_PARAMS['weights'],
                                 CONFIG_PARAMS['N_LAYERS_LOCKED'],
                                 batch_norm=CONFIG_PARAMS['batch_norm'],
                                 instance_norm=CONFIG_PARAMS['instance_norm'],
                                 gridsize=gridsize)
elif CONFIG_PARAMS['train_mode'] == 'continued':
    model = load_model(CONFIG_PARAMS['weights'], 
                       custom_objects={'ops': ops,
                                       'slice_input': nets.slice_input,
                                       'mask_nan_keep_loss': losses.mask_nan_keep_loss,
                                       'euclidean_distance_3D': losses.euclidean_distance_3D,
                                       'centered_euclidean_distance_3D': losses.centered_euclidean_distance_3D})
elif CONFIG_PARAMS['train_mode'] == 'continued_weights_only':
  # This does not work with models created in 'finetune' mode, but will work with models
  # started from scratch ('new' train_mode)
    model = CONFIG_PARAMS['net'](CONFIG_PARAMS['loss'],
                                 float(CONFIG_PARAMS['lr']),
                                 CONFIG_PARAMS['N_CHANNELS_IN'] + CONFIG_PARAMS['DEPTH'],
                                 CONFIG_PARAMS['N_CHANNELS_OUT'],
                                 3 if cam3_train else len(camnames[0]),
                                 batch_norm=CONFIG_PARAMS['batch_norm'],
                                 instance_norm=CONFIG_PARAMS['instance_norm'],
                                 include_top=True,
                                 gridsize=gridsize)
    model.load_weights(CONFIG_PARAMS['weights'])
else:
    raise Exception("Invalid training mode")

model.compile(optimizer=Adam(lr=float(CONFIG_PARAMS['lr'])),
              loss=CONFIG_PARAMS['loss'],
              metrics=metrics)

print("COMPLETE\n")

# Create checkpoint and logging callbacks
if CONFIG_PARAMS['num_validation_per_exp'] > 0:
    kkey = 'weights.{epoch:02d}-{val_loss:.5f}.hdf5'
    mon = 'val_loss'
else:
    kkey = 'weights.{epoch:02d}-{loss:.5f}.hdf5'
    mon = 'loss'

model_checkpoint = ModelCheckpoint(os.path.join(RESULTSDIR,
                                   kkey),
                                   monitor=mon,
                                   save_best_only=True,
                                   save_weights_only=True)
csvlog = CSVLogger(os.path.join(RESULTSDIR, 'training.csv'))
tboard = TensorBoard(log_dir=RESULTSDIR + 'logs',
                     write_graph=False,
                     update_freq=100)

model.fit(x=train_generator,
          steps_per_epoch=len(train_generator),
          validation_data=valid_generator,
          validation_steps=len(valid_generator),
          verbose=CONFIG_PARAMS['VERBOSE'],
          epochs=CONFIG_PARAMS['EPOCHS'],
          callbacks=[csvlog, model_checkpoint, tboard])

print("Saving full model at end of training")
sdir = os.path.join(CONFIG_PARAMS['RESULTSDIR'], 'fullmodel_weights')
if not os.path.exists(sdir):
    os.makedirs(sdir)
model.save(os.path.join(sdir, 'fullmodel_end.hdf5'))

print("done!")
