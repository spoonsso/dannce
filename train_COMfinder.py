"""
Trains the COMfinder U-Net.

Usage: python train_COMfinder.py settings_config

"""
import sys
import numpy as np
import os
import dannce.engine.serve_data_DANNCE as serve_data
import dannce.engine.processing as processing
from dannce.engine.generator_aux import DataGenerator_downsample
from dannce.engine import nets
from dannce.engine import losses
from six.moves import cPickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from copy import deepcopy
from tensorflow.random import set_seed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set up parameters
base_params = processing.read_config(sys.argv[1])
base_params = processing.make_paths_safe(base_params)

com_params = processing.read_config(base_params['COM_CONFIG'])
com_params = processing.make_paths_safe(com_params)
com_params = processing.inherit_config(com_params, base_params, list(base_params.keys()))


com_params['loss'] = getattr(losses, com_params['loss'])
com_params['net'] = getattr(nets, com_params['net'])

os.environ["CUDA_VISIBLE_DEVICES"] = com_params['gpuID']

samples = []
datadict = {}
datadict_3d = {}
cameras = {}
camnames = {}

exps = processing.grab_exp_file(com_params)
num_experiments = len(exps)
com_params['experiment'] = {}
MULTI_MODE = com_params['N_CHANNELS_OUT'] > 1
com_params['N_CHANNELS_OUT'] = com_params['N_CHANNELS_OUT'] + int(MULTI_MODE)
for e, exp_file in enumerate(exps):
    exp = processing.read_config(exp_file)
    exp = processing.make_paths_safe(exp)

    exp = \
    processing.inherit_config(exp,
                              base_params,
                              ['CAMNAMES',
                               'CALIBDIR',
                               'calib_file',
                               'extension',
                               'datafile'])
    
    for k in ['datadir', 'viddir', 'CALIBDIR']:
      exp[k] = os.path.join(exp['base_exp_folder'], exp[k])
    exp['datafile'] = exp['com_datafile']
    com_params['experiment'][e] = exp
    samples_, datadict_, datadict_3d_, data_3d_, cameras_ = \
        serve_data.prepare_data(com_params['experiment'][e], 
                                nanflag=False, 
                                com_flag= not MULTI_MODE,
                                multimode = MULTI_MODE)

    # No need to prepare any COM file (they don't exist yet).
    # We call this because we want to support multiple experiments,
    # which requires appending the experiment ID to each data object and key
    samples, datadict, datadict_3d, ddd = serve_data.add_experiment(
        e, samples, datadict, datadict_3d, {},
        samples_, datadict_, datadict_3d_, {})
    cameras[e] = cameras_
    camnames[e] = com_params['experiment'][e]['CAMNAMES']

RESULTSDIR = com_params['RESULTSDIR']
print(RESULTSDIR)

if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)

# Additionally, to keep videos unique across experiments, need to add
# experiment labels in other places. E.g. experiment 0 CameraE's "camname"
# Becomes 0_CameraE.
cameras, datadict, com_params = serve_data.prepend_experiment(com_params, datadict,
                                                  num_experiments, camnames, cameras)

samples = np.array(samples)

e = 0

# Initialize video objects
vids = {}
for e in range(num_experiments):
    vids = processing.initialize_vids_train(com_params, datadict, e,
                                                vids, pathonly=True)

print("Using {} downsampling".format(com_params['dsmode'] if 'dsmode' 
      in com_params.keys() else 'dsm'))

params = {'dim_in': (com_params['CROP_HEIGHT'][1]-com_params['CROP_HEIGHT'][0],
                     com_params['CROP_WIDTH'][1]-com_params['CROP_WIDTH'][0]),
          'n_channels_in': com_params['N_CHANNELS_IN'],
          'batch_size': 1,
          'n_channels_out': com_params['N_CHANNELS_OUT'],
          'out_scale': com_params['SIGMA'],
          'camnames': camnames,
          'crop_width': com_params['CROP_WIDTH'],
          'crop_height': com_params['CROP_HEIGHT'],
          'downsample': com_params['DOWNFAC'],
          'shuffle': False,
          'chunks': com_params['chunks'],
          'dsmode': com_params['dsmode'] if 'dsmode' in com_params.keys() else 'dsm',
          'preload': False}

valid_params = deepcopy(params)
valid_params['shuffle'] = False

partition = {}
if 'load_valid' not in com_params.keys():

    all_inds = np.arange(len(samples))

    # extract random inds from each set for validation
    v = com_params['num_validation_per_exp']
    valid_inds = []
    for e in range(num_experiments):
        tinds = [i for i in range(len(samples))
                 if int(samples[i].split('_')[0]) == e]
        valid_inds = valid_inds + list(np.random.choice(tinds,
                                                        (v,), replace=False))
        valid_inds = list(np.sort(valid_inds))

    train_inds = [i for i in all_inds if i not in valid_inds]
    assert (set(valid_inds) & set(train_inds)) == set()

    partition['train'] = samples[train_inds]
    partition['valid'] = samples[valid_inds]
else:
    # Load validation samples from elsewhere
    with open(os.path.join(com_params['load_valid'], 'val_samples.pickle'),
              'rb') as f:
        partition['valid'] = cPickle.load(f)
    partition['train'] = [f for f in samples if f not in partition['valid']]

# Optionally, we can subselect a number of random train indices
if 'num_train_per_exp' in com_params.keys():
    nt = com_params['num_train_per_exp']
    subtrain = []
    for e in range(num_experiments):
        tinds = np.array([i for i in partition['train']
                 if int(i.split('_')[0]) == e])
        tinds_ = np.random.choice(np.arange(len(tinds)), (nt,), replace=False)
        tinds_ = np.sort(tinds_)
        subtrain = subtrain + list(tinds[tinds_])

    partition['train'] = subtrain

# Save train/val inds
with open(RESULTSDIR + 'val_samples.pickle', 'wb') as f:
    cPickle.dump(partition['valid'], f)

with open(RESULTSDIR + 'train_samples.pickle', 'wb') as f:
    cPickle.dump(partition['train'], f)

labels = datadict

# Build net
print("Initializing Network...")

# with tf.device("/gpu:0"):
model = com_params['net'](com_params['loss'], float(com_params['lr']),
                             com_params['N_CHANNELS_IN'],
                             com_params['N_CHANNELS_OUT'],
                             com_params['metric'], multigpu=False)
print("COMPLETE\n")

if com_params['weights'] is not None:
    weights = os.listdir(com_params['weights'])
    weights = [f for f in weights if '.hdf5' in f]
    weights = weights[0]

    try:
        model.load_weights(os.path.join(com_params['weights'],weights))
    except:
        print("Note: model weights could not be loaded due to a mismatch in dimensions.\
               Assuming that this is a fine-tune with a different number of outputs and removing \
              the top of the net accordingly")
        model.layers[-1].name = 'top_conv'
        model.load_weights(os.path.join(com_params['weights'],weights), by_name=True)

if 'lockfirst' in com_params.keys() and com_params['lockfirst']:
    for layer in model.layers[:2]:
        layer.trainable = False
    
model.compile(optimizer=Adam(lr=float(com_params['lr'])),
              loss=com_params['loss'], metrics=['mse'])

# Create checkpoint and logging callbacks
model_checkpoint = ModelCheckpoint(os.path.join(RESULTSDIR,
                                   'weights.{epoch:02d}-{val_loss:.5f}.hdf5'),
                                   monitor='loss',
                                   save_best_only=True,
                                   save_weights_only=True)
csvlog = CSVLogger(os.path.join(RESULTSDIR, 'training.csv'))
tboard = TensorBoard(log_dir=RESULTSDIR + 'logs',
                     write_graph=False,
                     update_freq=100)

# Initialize data structures
ncams = len(camnames[0])
dh = (com_params['CROP_HEIGHT'][1]-com_params['CROP_HEIGHT'][0]) \
        // com_params['DOWNFAC']
dw = (com_params['CROP_WIDTH'][1]-com_params['CROP_WIDTH'][0]) \
        // com_params['DOWNFAC']
ims_train = np.zeros((ncams*len(partition['train']),
                     dh, dw, 3), dtype='float32')
y_train = np.zeros((ncams*len(partition['train']),
                    dh, dw, com_params['N_CHANNELS_OUT']),
                   dtype='float32')
ims_valid = np.zeros((ncams*len(partition['valid']),
                      dh, dw, 3), dtype='float32')
y_valid = np.zeros((ncams*len(partition['valid']),
                    dh, dw, com_params['N_CHANNELS_OUT']),
                   dtype='float32')

# When there are a lot of videos 
train_generator = DataGenerator_downsample(partition['train'],
                                           labels, vids, **params)
valid_generator = DataGenerator_downsample(partition['valid'],
                                           labels, vids, **valid_params)

print("Loading data")
for i in range(len(partition['train'])):
    print(i, end='\r')
    ims = train_generator.__getitem__(i)
    ims_train[i*ncams:(i+1)*ncams] = ims[0]
    y_train[i*ncams:(i+1)*ncams] = ims[1]

for i in range(len(partition['valid'])):
    ims = valid_generator.__getitem__(i)
    ims_valid[i*ncams:(i+1)*ncams] = ims[0]
    y_valid[i*ncams:(i+1)*ncams] = ims[1]

if com_params['debug'] and not MULTI_MODE:
    # Plot all training images and save
    # create new directory for images if necessary
    debugdir = os.path.join(com_params['RESULTSDIR'], 'debug_im_out')
    print("Saving debug images to: " + debugdir)
    if not os.path.exists(debugdir):
        os.makedirs(debugdir)

    plt.figure()
    for i in range(ims_train.shape[0]):
        plt.cla()
        processing.plot_markers_2d(processing.norm_im(ims_train[i]),
                                   y_train[i],
                                   newfig=False)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        imname = str(i) + '.png'
        plt.savefig(os.path.join(debugdir, imname),
                    bbox_inches='tight', pad_inches=0)
elif com_params['debug'] and MULTI_MODE:
    print("Note: Cannot output debug information in COM multi-mode")

model.fit(ims_train,
          y_train,
          validation_data=(ims_valid, y_valid),
          batch_size=com_params['BATCH_SIZE']*ncams,
          epochs=com_params['EPOCHS'],
          callbacks=[csvlog, model_checkpoint, tboard],
          shuffle=True)

if com_params['debug'] and not MULTI_MODE:
    # Plot predictions on validation frames
    debugdir = os.path.join(com_params['RESULTSDIR'], 'debug_im_out_valid')
    print("Saving debug images to: " + debugdir)
    if not os.path.exists(debugdir):
        os.makedirs(debugdir)

    plt.figure()
    for i in range(ims_valid.shape[0]):
        plt.cla()
        processing.plot_markers_2d(processing.norm_im(ims_valid[i]),
                                   model.predict(ims_valid[i:i+1])[0],
                                   newfig=False)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        imname = str(i) + '.png'
        plt.savefig(os.path.join(debugdir, imname),
                    bbox_inches='tight', pad_inches=0)
elif com_params['debug'] and MULTI_MODE:
    print("Note: Cannot output debug information in COM multi-mode")

print("Saving full model at end of training")
sdir = os.path.join(com_params['RESULTSDIR'], 'fullmodel_weights')
if not os.path.exists(sdir):
    os.makedirs(sdir)
model.save(os.path.join(sdir, 'fullmodel_end.hdf5'))