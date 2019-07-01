"""
Trains the COMfinder U-Net.

Usage: python train_COMfinder.py settings_config path_to_experiment1_config,
     ..., path_to_experimentN_config

In contrast to predict_DANNCE.py, train_DANNCE.py can process multiple
experiment config files  to support training over multiple animals.
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
from keras.layers import Conv3D, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set up parameters
CONFIG_PARAMS = processing.read_config(sys.argv[1])
CONFIG_PARAMS['loss'] = getattr(losses, CONFIG_PARAMS['loss'])
CONFIG_PARAMS['net'] = getattr(nets, CONFIG_PARAMS['net'])

os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG_PARAMS['gpuID']

samples = []
datadict = {}
datadict_3d = {}
cameras = {}
camnames = {}

exps = sys.argv[2:]
num_experiments = len(exps)
CONFIG_PARAMS['experiment'] = {}
for e in range(num_experiments):
    CONFIG_PARAMS['experiment'][e] = processing.read_config(exps[e])

    samples_, datadict_, datadict_3d_, data_3d_, cameras_ = \
        serve_data.prepare_data(CONFIG_PARAMS['experiment'][e],nanflag=False)

    # No need to prepare any COM file (they don't exist yet).
    # We call this because we want to support multiple experiments,
    # which requires appending the experiment ID to each data object and key
    samples, datadict, datadict_3d, ddd = serve_data.add_experiment(
        e, samples, datadict, datadict_3d, {},
        samples_, datadict_, datadict_3d_, {})
    cameras[e] = cameras_
    camnames[e] = CONFIG_PARAMS['experiment'][e]['CAMNAMES']

RESULTSDIR = CONFIG_PARAMS['RESULTSDIR']
print(RESULTSDIR)

if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)

# Additionally, to keep videos unique across experiments, need to add
# experiment labels in other places. E.g. experiment 0 CameraE's "camname"
# Becomes 0_CameraE.
# TODO: Add this to serve_data.add_experiment() above
cameras_ = {}
datadict_ = {}
for e in range(num_experiments):
    # Create a unique camname for each camera in each experiment
    cameras_[e] = {}
    for key in cameras[e]:
        cameras_[e][str(e) + '_' + key] = cameras[e][key]

    camnames[e] = [str(e) + '_' + f for f in camnames[e]]

    CONFIG_PARAMS['experiment'][e]['CAMNAMES'] = camnames[e]

# Change the camnames in the data dictionaries as well
for key in datadict.keys():
    enum = key.split('_')[0]
    datadict_[key] = {}
    datadict_[key]['data'] = {}
    datadict_[key]['frames'] = {}
    for key_ in datadict[key]['data']:
        datadict_[key]['data'][enum + '_' + key_] = datadict[key]['data'][key_]
        datadict_[key]['frames'][enum + '_' + key_] =  \
            datadict[key]['frames'][key_]

datadict = datadict_
cameras = cameras_

samples = np.array(samples)

# Open videos for all experiments
vids = {}
for e in range(num_experiments):

    for i in range(len(CONFIG_PARAMS['experiment'][e]['CAMNAMES'])):
        if CONFIG_PARAMS['vid_dir_flag']:
            addl = ''
        else:
            addl = os.listdir(os.path.join(
                CONFIG_PARAMS['experiment'][e]['viddir'],
                CONFIG_PARAMS['experiment'][e]['CAMNAMES'][i].split('_')[1]))[0]
        r = \
            processing.generate_readers(
                CONFIG_PARAMS['experiment'][e]['viddir'],
                os.path.join(CONFIG_PARAMS['experiment'][e]
                             ['CAMNAMES'][i].split('_')[1], addl),
                maxopt=np.inf,  # Large enough to encompass all videos in directory.
                extension=CONFIG_PARAMS['experiment'][e]['extension'])

        # Add e to key
        vids[CONFIG_PARAMS['experiment'][e]['CAMNAMES'][i]] = {}
        for key in r:
            vids[CONFIG_PARAMS['experiment'][e]['CAMNAMES'][i]][str(e) +
                                                                '_' + key]\
                                                                = r[key]


params = {'dim_in': (CONFIG_PARAMS['INPUT_HEIGHT'],
                     CONFIG_PARAMS['INPUT_WIDTH']),
          'n_channels_in': CONFIG_PARAMS['N_CHANNELS_IN'],
          'dim_out': (CONFIG_PARAMS['OUTPUT_HEIGHT'],
                      CONFIG_PARAMS['OUTPUT_WIDTH']),
          'batch_size': 1,
          'n_channels_out': CONFIG_PARAMS['N_CHANNELS_OUT'],
          'out_scale': CONFIG_PARAMS['SIGMA'],
          'camnames': camnames,
          'crop_width': CONFIG_PARAMS['CROP_WIDTH'],
          'crop_height': CONFIG_PARAMS['CROP_HEIGHT'],
          'bbox_dim': (CONFIG_PARAMS['BBOX_HEIGHT'],
                       CONFIG_PARAMS['BBOX_WIDTH']),
          'downsample': CONFIG_PARAMS['DOWNFAC'],
          'shuffle': True}

valid_params = deepcopy(params)
valid_params['shuffle'] = True

partition = {}
if CONFIG_PARAMS['load_valid'] is None:

    all_inds = np.arange(len(samples))

    # extract random inds from each set for validation
    v = CONFIG_PARAMS['num_validation_per_exp']
    valid_inds = []
    for e in range(num_experiments):
        tinds = [i for i in range(len(samples))
                 if int(samples[i].split('_')[0]) == e]
        valid_inds = valid_inds + list(np.random.choice(tinds,
                                                        (v,), replace=False))

    train_inds = [i for i in all_inds if i not in valid_inds]
    assert (set(valid_inds) & set(train_inds)) == set()

    partition['train'] = samples[train_inds]
    partition['valid'] = samples[valid_inds]
else:
    # Load validation samples from elsewhere
    with open(os.path.join(CONFIG_PARAMS['load_valid'], 'val_samples.pickle'),
              'rb') as f:
        partition['valid'] = cPickle.load(f)
    partition['train'] = [f for f in samples if f not in partition['valid']]

# Save train/val inds
with open(RESULTSDIR + 'val_samples.pickle', 'wb') as f:
    cPickle.dump(partition['valid'], f)

with open(RESULTSDIR + 'train_samples.pickle', 'wb') as f:
    cPickle.dump(partition['train'], f)

labels = datadict
labels_3d = datadict_3d

train_generator = DataGenerator_downsample(partition['train'],
                                           labels, vids, **params)
valid_generator = DataGenerator_downsample(partition['valid'],
                                           labels, vids, **valid_params)

# Build net
print("Initializing Network...")

# with tf.device("/gpu:0"):
model = CONFIG_PARAMS['net'](CONFIG_PARAMS['loss'], CONFIG_PARAMS['lr'],
                             CONFIG_PARAMS['N_CHANNELS_IN'],
                             CONFIG_PARAMS['N_CHANNELS_OUT'],
                             CONFIG_PARAMS['metric'], multigpu=False)
print("COMPLETE\n")

if CONFIG_PARAMS['weights'] is not None:
    model.load_weights(CONFIG_PARAMS['weights'])

if CONFIG_PARAMS['lockfirst']:
    for layer in model.layers[:2]:
        layer.trainable = False
    
model.compile(optimizer=Adam(lr=CONFIG_PARAMS['lr']), loss=CONFIG_PARAMS['loss'], metrics=['mse'])

# Create checkpoint and logging callbacks
model_checkpoint = ModelCheckpoint(os.path.join(RESULTSDIR,
                                   'weights.{epoch:02d}-{val_loss:.5f}.hdf5'),
                                   monitor='val_loss',
                                   save_best_only=True,
                                   save_weights_only=True)
csvlog = CSVLogger(os.path.join(RESULTSDIR, 'training.csv'))
tboard = TensorBoard(log_dir=RESULTSDIR + 'logs',
                     write_graph=False,
                     update_freq=100)

# Initialize data structures
ncams = len(camnames[0])
dh = CONFIG_PARAMS['INPUT_HEIGHT']//CONFIG_PARAMS['DOWNFAC']
dw = CONFIG_PARAMS['INPUT_WIDTH']//CONFIG_PARAMS['DOWNFAC']
ims_train = np.zeros((ncams*len(partition['train']),
                     dh, dw, 3), dtype='float32')
y_train = np.zeros((ncams*len(partition['train']),
                    dh, dw, CONFIG_PARAMS['N_CHANNELS_OUT']),
                   dtype='float32')
ims_valid = np.zeros((ncams*len(partition['valid']),
                      dh, dw, 3), dtype='float32')
y_valid = np.zeros((ncams*len(partition['valid']),
                    dh, dw, CONFIG_PARAMS['N_CHANNELS_OUT']),
                   dtype='float32')

print("Loading data")
for i in range(len(partition['train'])):
    ims = train_generator.__getitem__(i)
    ims_train[i*ncams:(i+1)*ncams] = ims[0]
    y_train[i*ncams:(i+1)*ncams] = ims[1]

for i in range(len(partition['valid'])):
    ims = valid_generator.__getitem__(i)
    ims_valid[i*ncams:(i+1)*ncams] = ims[0]
    y_valid[i*ncams:(i+1)*ncams] = ims[1]

if CONFIG_PARAMS['debug']:
    # Plot all training images and save
    # create new directory for images if necessary
    debugdir = os.path.join(CONFIG_PARAMS['RESULTSDIR'], 'debug_im_out')
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

model.fit(ims_train,
          y_train,
          validation_data=(ims_valid, y_valid),
          batch_size=CONFIG_PARAMS['BATCH_SIZE']*ncams,
          epochs=CONFIG_PARAMS['EPOCHS'],
          callbacks=[csvlog, model_checkpoint, tboard],
          shuffle=True)

if CONFIG_PARAMS['debug']:
    # Plot predictions on validation frames
    debugdir = os.path.join(CONFIG_PARAMS['RESULTSDIR'], 'debug_im_out_valid')
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

print("Saving full model at end of training")
model.save(os.path.join(RESULTSDIR, 'fullmodel_end.hdf5'))