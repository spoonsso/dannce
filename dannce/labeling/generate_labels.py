"""Randomly sample videos to generate images for labeling.

Usage: python generate_labels.py path_to_config_file num_images_per_camera
       path_to_s3_folder
"""
import numpy as np
import scipy.io as sio
import os
import sys
from dannce.engine import processing as processing
from dannce.engine.generator_aux import DataGenerator_downsample
from dannce.engine import serve_data_DANNCE as serve_data

# Load params from config
CONFIG_PARAMS = processing.read_config(sys.argv[1])
print("Loading configuration from: " + sys.argv[1])
RESULTSDIR = os.path.join(CONFIG_PARAMS['RESULTSDIR'], 'imDir/')
print("Saving images to: " + RESULTSDIR)
if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)

if 'seed' in CONFIG_PARAMS.keys():
    np.random.seed(CONFIG_PARAMS['seed'])
# Load data structure for indices
samples_, datadict_, datadict_3d_, data_3d_ = \
    serve_data.prepare_data(CONFIG_PARAMS, com_flag=False)

# Zero any negative frames -- DEPRECATED
for key in datadict_.keys():
	for key_ in datadict_[key]['frames'].keys():
		if datadict_[key]['frames'][key_] < 0:
			datadict_[key]['frames'][key_] = 0

# Generate video readers. should move this into processing.py
vid_dir_flag = CONFIG_PARAMS['vid_dir_flag']
vids = {}
for i in range(len(CONFIG_PARAMS['CAMNAMES'])):
    if vid_dir_flag:
        addl = ''
    else:
        addl = os.listdir(os.path.join(
            CONFIG_PARAMS['viddir'], CONFIG_PARAMS['CAMNAMES'][i]))[0]

    # Get max video
    v = os.listdir(os.path.join(
        CONFIG_PARAMS['viddir'], CONFIG_PARAMS['CAMNAMES'][i], addl))
    v = [int(f.split('.')[0]) for f in v if CONFIG_PARAMS['extension'] in f]
    v = sorted(v)

    vids[CONFIG_PARAMS['CAMNAMES'][i]] = \
        processing.generate_readers(
            CONFIG_PARAMS['viddir'],
            os.path.join(CONFIG_PARAMS['CAMNAMES'][i], addl),
            maxopt=v[-1], extension=CONFIG_PARAMS['extension'])


params = {
    'dim_in': (CONFIG_PARAMS['INPUT_HEIGHT'], CONFIG_PARAMS['INPUT_WIDTH']),
    'n_channels_in': CONFIG_PARAMS['N_CHANNELS_IN'],
    'dim_out': (CONFIG_PARAMS['OUTPUT_HEIGHT'], CONFIG_PARAMS['OUTPUT_WIDTH']),
    'batch_size': CONFIG_PARAMS['BATCH_SIZE'],
    'n_channels_out': CONFIG_PARAMS['N_CHANNELS_OUT'],
    'camnames': {0: CONFIG_PARAMS['CAMNAMES']},
    'crop_width': CONFIG_PARAMS['CROP_WIDTH'],
    'crop_height': CONFIG_PARAMS['CROP_HEIGHT'],
    'bbox_dim': (CONFIG_PARAMS['BBOX_HEIGHT'], CONFIG_PARAMS['BBOX_WIDTH']),
    'chunks': CONFIG_PARAMS['chunks'],
    'shuffle': True}

labels = datadict_
labels_3d = datadict_3d_

partition = {}
partition['train'] = \
    np.random.choice(samples_, size=(int(sys.argv[2]),), replace=False)

generator = \
    DataGenerator_downsample(partition['train'], labels, vids, **params)

# generate images. Should create argument that sets the png output quality --
# currently uses imageio's default, which is optimzied compression. However,
# doing an image diff between default compression and no compression
# (which creates significantly large file sizes) reveals no major differences.
generator.save_for_dlc(RESULTSDIR,
                       compress_level=CONFIG_PARAMS['compress_level'])

# Write manifest file
folder = sys.argv[3]
allc = sio.loadmat(os.path.join(RESULTSDIR, 'allcoords.mat'))['filenames']
with open(os.path.join(
        CONFIG_PARAMS['RESULTSDIR'], 'dataset.manifest'), 'w') as f:
	for it in allc:
		fname = it.strip()
		fname = fname.split('/')[-1]
		write_string = "\"source-ref\":\"{}\"".format(folder + fname)
		write_string = "{" + write_string + "}\n"
		f.write(write_string)
print("done!")
