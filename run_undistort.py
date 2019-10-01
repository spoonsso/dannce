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
import matplotlib.pyplot as plt
import matlab
import matlab.engine

# Set up environment
eng = matlab.engine.start_matlab()
# undistort_allCOMS.m needs to be in the same directory as predict_COMfinder.py
eng.addpath(os.path.dirname(os.path.abspath(__file__)))
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

comfile = 'allCOMs_distorted.mat'
sdfile = 'save_data.pickle'

RESULTSDIR = os.path.join(params['RESULTSDIR_PREDICT'])
print(RESULTSDIR)

if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)

# Use Matlab undistort function to undistort COMs
eng.undistort_allCOMS(
    comfile, [os.path.join(params['CALIBDIR'], f)
              for f in params['calib_file']],
    nargout=0)

# Get undistorted COMs frames and clean up
allCOMs_u = sio.loadmat('allCOMs_undistorted.mat')['allCOMs_u']
os.remove('allCOMs_distorted.mat')
os.remove('allCOMs_undistorted.mat')

# Load in save_data template
with open(sdfile,'rb') as f:
  save_data = cPickle.load(f)

# Save data to a pickle file
save_data_u = deepcopy(save_data)
num_cams = len(params['CAMNAMES'])
for (i, key) in enumerate(save_data.keys()):
    for c in range(num_cams):
        save_data_u[key][params['CAMNAMES'][c]]['COM'] = allCOMs_u[c, i]
f = open(os.path.join(RESULTSDIR,'COM_undistorted.pickle'), 'wb')
cPickle.dump(save_data_u, f)
f.close()


print('done!')
