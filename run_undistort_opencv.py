"""To finalize COM prediction after a crash, with COM checkpoint saved

Usage: python run_undistort.py path_to_config
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
import os
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')

# Load in the params
PARENT_PARAMS = processing.read_config(sys.argv[1])
params = processing.read_config(PARENT_PARAMS['COM_CONFIG'])

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

sdfile = 'COM_undistorted.pickle'

RESULTSDIR = os.path.join(params['RESULTSDIR_PREDICT'])
sdfile = os.path.join(RESULTSDIR,  sdfile)
print("Distorting data in " + sdfile)


# Process config to get camera params
print("Loading camera params")
samples, datadict, datadict_3d, cameras, camera_mats, vids = \
    serve_data.prepare_data(
        params, vid_dir_flag=params['vid_dir_flag'], minopt=0, maxopt=0)

# Load in save_data template
with open(sdfile,'rb') as f:
  save_data = cPickle.load(f)

# Save data to a pickle file
save_data_u = deepcopy(save_data)
num_cams = len(params['CAMNAMES'])
for (i, key) in enumerate(save_data.keys()):
    print(key)
    for c in range(num_cams):
        pts1 = save_data_u[key][params['CAMNAMES'][c]]['COM']
        pts1 = pts1[np.newaxis, :]
        pts1 = ops.unDistortPoints(
            pts1, cameras[params['CAMNAMES'][c]]['K'],
            cameras[params['CAMNAMES'][c]]['RDistort'],
            cameras[params['CAMNAMES'][c]]['TDistort'],
            cameras[params['CAMNAMES'][c]]['R'],
            cameras[params['CAMNAMES'][c]]['t'])
        save_data_u[key][params['CAMNAMES'][c]]['COM'] = np.squeeze(pts1)

                # Triangulate for all unique pairs
    for j in range(num_cams):
        for k in range(j + 1, num_cams):
            pts1 = save_data[key][params['CAMNAMES'][j]]['COM']
            pts2 = save_data[key][params['CAMNAMES'][k]]['COM']
            pts1 = pts1[np.newaxis, :]
            pts2 = pts2[np.newaxis, :]
            
            test3d = ops.triangulate(
                pts1, pts2, camera_mats[params['CAMNAMES'][j]],
                camera_mats[params['CAMNAMES'][k]]).squeeze()

            save_data_u[key]['triangulation']["{}_{}".format(
                params['CAMNAMES'][j], params['CAMNAMES'][k])] = test3d    

f = open(os.path.join(RESULTSDIR,'COM_undistorted.pickle'), 'wb')
cPickle.dump(save_data_u, f)
f.close()


print('done!')
