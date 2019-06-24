"""
Trains DANNCE. Currently only supports fine-tuning with hand-labeled data.
That is, this code loads all volumes and labels into memory before training.
This memory-based training is only available when using a small number of
hand-labeled frames. Code for training over chronic recordings / motion
capture to come.

Usage: python train_DANNCE.py settings_config path_to_experiment1_config,
     ..., path_to_experimentN_config

In contrast to predict_DANNCE.py, train_DANNCE.py can process multiple
experiment config files  to support training over multiple animals.
"""
import sys
import numpy as np
import os
import time
import ast
import keras.backend as K
import dannce.engine.serve_data_DANNCE as serve_data
import dannce.engine.processing as processing
from dannce.engine.processing import savedata_tomat, savedata_expval
from dannce.engine.generator_kmeans import DataGenerator_3Dconv_kmeans
from dannce.engine.generator_kmeans import DataGenerator_3Dconv_frommem
from dannce.engine import nets
from dannce.engine import losses
from six.moves import cPickle
from keras.layers import Conv3D, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard

# TODO(Devices): Is it necessary to set the device environment?
# This could mess with people's setups.
# Tim: For a multi-gpu system, this is necessary to not tie up all
# GPUs with one script, as keras/TF will automatically allocate all
# system GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set up parameters
CONFIG_PARAMS = processing.read_config(sys.argv[1])
CONFIG_PARAMS['loss'] = getattr(losses, CONFIG_PARAMS['loss'])
CONFIG_PARAMS['net'] = getattr(nets, CONFIG_PARAMS['net'])

samples = []
datadict = {}
datadict_3d = {}
com3d_dict = {}
cameras = {}
camnames = {}

exps = sys.argv[2:]
num_experiments = len(exps)
CONFIG_PARAMS['experiment'] = {}
for e in range(num_experiments):
    CONFIG_PARAMS['experiment'][e] = processing.read_config(exps[e])

    samples_, datadict_, datadict_3d_, data_3d_, cameras_ = \
        serve_data.prepare_data(CONFIG_PARAMS['experiment'][e])

    datadict_, com3d_dict_ = serve_data.prepare_COM(
        CONFIG_PARAMS['experiment'][e]['COMfilename'],
        datadict_,
        comthresh=CONFIG_PARAMS['comthresh'],
        weighted=CONFIG_PARAMS['weighted'],
        retriangulate=False,
        camera_mats=cameras_,
        method=CONFIG_PARAMS['com_method'])

    # Need to cap this at the number of samples included in our
    # COM finding estimates
    tf = list(com3d_dict_.keys())
    samples_ = samples_[:len(tf)]
    data_3d_ = data_3d_[:len(tf)]
    samples_, data_3d_ = \
        serve_data.remove_samples_com(samples_,
                                      data_3d_,
                                      com3d_dict_,
                                      rmc=True)
    pre = len(samples_)
    msg = "Detected {} bad COMs and removed the associated frames from the dataset"
    print(msg.format(pre - len(samples_)))

    samples, datadict, datadict_3d, com3d_dict = serve_data.add_experiment(
        e, samples, datadict, datadict_3d, com3d_dict,
        samples_, datadict_, datadict_3d_, com3d_dict_)

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
    if CONFIG_PARAMS['IMMODE'] == 'vid':

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

# Parameters
valid_params = {
    'dim_in': (CONFIG_PARAMS['INPUT_HEIGHT'], CONFIG_PARAMS['INPUT_WIDTH']),
    'n_channels_in': CONFIG_PARAMS['N_CHANNELS_IN'],
    'dim_out': (CONFIG_PARAMS['OUTPUT_HEIGHT'], CONFIG_PARAMS['OUTPUT_WIDTH']),
    'batch_size': 1,
    'n_channels_out': CONFIG_PARAMS['NEW_N_CHANNELS_OUT'],
    'out_scale': CONFIG_PARAMS['SIGMA'],
    'crop_width': CONFIG_PARAMS['CROP_WIDTH'],
    'crop_height': CONFIG_PARAMS['CROP_HEIGHT'],
    'bbox_dim': (CONFIG_PARAMS['BBOX_HEIGHT'], CONFIG_PARAMS['BBOX_WIDTH']),
    'vmin': CONFIG_PARAMS['VMIN'],
    'vmax': CONFIG_PARAMS['VMAX'],
    'nvox': CONFIG_PARAMS['NVOX'],
    'interp': CONFIG_PARAMS['INTERP'],
    'depth': CONFIG_PARAMS['DEPTH'],
    'channel_combo': CONFIG_PARAMS['CHANNEL_COMBO'],
    'mode': CONFIG_PARAMS['OUT_MODE'],
    'camnames': camnames,
    'immode': CONFIG_PARAMS['IMMODE'],
    'training': False,  # This means we are not sampling from K-Means clusters
    'shuffle': False,  # We will shuffle later
    'rotation': False,  # We will rotate later if desired
    'pregrid': None,
    'pre_projgrid': None,
    'stamp': False,
    'vidreaders': vids,
    'distort': CONFIG_PARAMS['DISTORT'],
    'expval': CONFIG_PARAMS['EXPVAL'],
    'crop_im': False}   # This should stay False

# Setup a generator that will read videos and labels
tifdirs = []  # Trainign from single images not yet supported in this demo

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

partition = {}

partition['valid_sampleIDs'] = samples[valid_inds]
partition['train_sampleIDs'] = samples[train_inds]

# Save train/val inds
with open(RESULTSDIR + 'val_samples.pickle', 'wb') as f:
    cPickle.dump(partition['valid_sampleIDs'], f)

with open(RESULTSDIR + 'train_samples.pickle', 'wb') as f:
    cPickle.dump(partition['train_sampleIDs'], f)

train_generator = DataGenerator_3Dconv_kmeans(partition['train_sampleIDs'],
                                              datadict,
                                              datadict_3d,
                                              cameras,
                                              partition['train_sampleIDs'],
                                              com3d_dict,
                                              tifdirs,
                                              **valid_params)
valid_generator = DataGenerator_3Dconv_kmeans(partition['valid_sampleIDs'],
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
y_train = np.zeros((len(partition['train_sampleIDs']),
                    CONFIG_PARAMS['NVOX'],
                    CONFIG_PARAMS['NVOX'],
                    CONFIG_PARAMS['NVOX'],
                    CONFIG_PARAMS['NEW_N_CHANNELS_OUT']),
                   dtype='float32')

X_valid = np.zeros((len(partition['valid_sampleIDs']),
                    CONFIG_PARAMS['NVOX'],
                    CONFIG_PARAMS['NVOX'],
                    CONFIG_PARAMS['NVOX'],
                    CONFIG_PARAMS['N_CHANNELS_IN']*len(camnames[0])),
                   dtype='float32')
y_valid = np.zeros((len(partition['valid_sampleIDs']),
                    CONFIG_PARAMS['NVOX'],
                    CONFIG_PARAMS['NVOX'],
                    CONFIG_PARAMS['NVOX'],
                    CONFIG_PARAMS['NEW_N_CHANNELS_OUT']),
                   dtype='float32')

print("Loading training data into memory")
for i in range(len(partition['train_sampleIDs'])):
    rr = train_generator.__getitem__(i)
    X_train[i] = rr[0]
    y_train[i] = rr[1]

print("Loading validation data into memory")
for i in range(len(partition['valid_sampleIDs'])):
    rr = valid_generator.__getitem__(i)
    X_valid[i] = rr[0]
    y_valid[i] = rr[1]

# Now we can generate from memory with shuffling, rotation, etc.
if CONFIG_PARAMS['CHANNEL_COMBO'] == 'random':
    randflag = True

train_generator = \
    DataGenerator_3Dconv_frommem(np.arange(len(partition['train_sampleIDs'])),
                                 X_train,
                                 y_train,
                                 batch_size=CONFIG_PARAMS['BATCH_SIZE'],
                                 random=randflag,
                                 rotation=CONFIG_PARAMS['ROTATE'])
valid_generator = \
    DataGenerator_3Dconv_frommem(np.arange(len(partition['valid_sampleIDs'])),
                                 X_valid,
                                 y_valid,
                                 batch_size=CONFIG_PARAMS['BATCH_SIZE'],
                                 random=randflag,
                                 rotation=CONFIG_PARAMS['ROTATE'])

# Build net
print("Initializing Network...")

assert not (CONFIG_PARAMS['batch_norm'] == True) & (CONFIG_PARAMS['instance_norm'] == True)
# with tf.device("/gpu:0"):
model = CONFIG_PARAMS['net'](CONFIG_PARAMS['loss'],
                             CONFIG_PARAMS['lr'],
                             CONFIG_PARAMS['N_CHANNELS_IN'] + CONFIG_PARAMS['DEPTH'],
                             CONFIG_PARAMS['N_CHANNELS_OUT'],
                             len(camnames[0]),
                             batch_norm=CONFIG_PARAMS['batch_norm'],
                             instance_norm=CONFIG_PARAMS['instance_norm'],
                             include_top=False)

print("COMPLETE\n")

# For fine-tuning:
# load in weights
model.load_weights(CONFIG_PARAMS['WEIGHTS'], by_name=True)

# lock first conv. layer
for layer in model.layers[:2]:
    layer.trainable = False

# Do forward pass all the way until end
idim = CONFIG_PARAMS['N_CHANNELS_IN'] + CONFIG_PARAMS['DEPTH']
ncams = len(camnames[0])
input_ = inputs = Input((None, None, None, idim*ncams))

old_out = model(input_)

# Add new output conv. layer
new_conv = Conv3D(CONFIG_PARAMS['NEW_N_CHANNELS_OUT'],
                  CONFIG_PARAMS['NEW_LAST_KERNEL_SIZE'],
                  activation='sigmoid',
                  padding='same')(old_out)

model = Model(inputs=[input_], outputs=[new_conv])

model.compile(optimizer=Adam(lr=CONFIG_PARAMS['lr']),
              loss=CONFIG_PARAMS['loss'],
              metrics=CONFIG_PARAMS['metric'])

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

# Train model on dataset
model.fit_generator(generator=train_generator,
                    steps_per_epoch=len(train_generator),
                    validation_data=valid_generator,
                    validation_steps=len(valid_generator),
                    use_multiprocessing=False,
                    workers=CONFIG_PARAMS['WORKERS'],
                    verbose=CONFIG_PARAMS['VERBOSE'],
                    epochs=CONFIG_PARAMS['EPOCHS'],
                    max_queue_size=CONFIG_PARAMS['MAX_QUEUE_SIZE'],
                    callbacks=[csvlog, model_checkpoint, tboard])

print("done!")
