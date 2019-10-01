"""
This script converts the output from predict_DANNCE into a predescribed
structured data format, with meta data, that can be used easily by our
downstream analysis pipeline

Usage:
python path_to_file/makeStructuredData.py path_to_config [path_to_template]

path_to_template is an optional parameter for times when I don't have any labeling directory
"""
import numpy as np
import scipy.io as sio
import sys
import os
import dannce.engine.processing as processing
import ast

# Set up parameters
PARENT_PARAMS = processing.read_config(sys.argv[1])
PARENT_PARAMS = processing.make_paths_safe(PARENT_PARAMS)
CONFIG_PARAMS = processing.read_config(PARENT_PARAMS['DANNCE_CONFIG'])
CONFIG_PARAMS = processing.make_paths_safe(CONFIG_PARAMS)

CONFIG_PARAMS['experiment'] = PARENT_PARAMS
RESULTSDIR = CONFIG_PARAMS['RESULTSDIR_PREDICT']
print("Reading results from: " + RESULTSDIR)

# 
dfiles = os.listdir(RESULTSDIR)
sfile = [f for f in dfiles if 'save_data' in f]
sfile = sfile[0]

pred = sio.loadmat(os.path.join(RESULTSDIR,sfile))

pred['sampleID'] = np.squeeze(pred['sampleID'])

# config file needs to have an extra DLC_PATH field
dlc = sio.loadmat(CONFIG_PARAMS['DLC_PATH'])

# take the median to get the final dlc predictions, 
#also transpose to match dannce pred shape
pred_dlc = np.median(dlc['data_3d'], axis=-1)
pred_dlc = np.transpose(pred_dlc, [0, 2, 1])

"""
We want the following to be saved:
1) animal name & data
2) network name
3) predictions in struct form, which includes the markernames, incorporate sampleID and data_frame for each camera into this.
4) camera calibration params, and links to the videos in the above struct

"""

fullpath = os.path.realpath('./')
name = os.path.split(os.path.realpath('./'))[-1]
date = os.path.split(os.path.realpath('../'))[-1]

# Load the markernames from mouse.template
# Get markernames from template file
if len(sys.argv) > 2:
    PARENT_PARAMS['lbl_template'] = sys.argv[2]

with open(PARENT_PARAMS['lbl_template'], 'r') as f:
    for line in f:
        if 'labels' in line:
            line = line.strip()
            line = line[8:-1]
            markernames = ast.literal_eval(line)

# In Matlab, we cannot keep parentheses in the marker names
markernames = [m.replace('(', '_') for m in markernames]
markernames = [m.replace(')', '_') for m in markernames]

cameras = {}

cnames = PARENT_PARAMS['CAMNAMES']

# Load in the first camera's matched frames and use this for all cameras
mframes = sio.loadmat(os.path.join(PARENT_PARAMS['datadir'],
                      PARENT_PARAMS['datafile'][0]))

mframes['data_sampleID'] = np.squeeze(mframes['data_sampleID'])
mframes['data_frame'] = np.squeeze(mframes['data_frame'])

df = np.zeros((len(pred['sampleID']), 1))
pred_locked = np.zeros((df.shape[0], pred_dlc.shape[1], pred_dlc.shape[2]))
# To get proper matched frames for dlc, we need to chain sampleID/frame 
# indices from multiple files
#
# The matched frames files contain the alignment between sampleID and frames
# The dannce pred file has the sampleIDs we want to use
# The pred_dlc frame indices go 0:pred_dlc.shape[0]
#
# So we walk thru the dannce pred sampleIDs (we need to track these because 
# some could be discarded due to COM error thresholding), find its matchign frame in the matched frames,
# then associate the pred_dlc prediction with that frame
frames_dlc = np.arange(pred_dlc.shape[0])

for j in range(df.shape[0]):
    if pred['sampleID'][j] in mframes['data_sampleID']:
        df[j] = mframes['data_frame'][mframes['data_sampleID'] == pred['sampleID'][j]]
        pred_locked[j] = pred_dlc[frames_dlc == df[j]]
    else:
        raise Exception("Could not find sampleID in matched frames")

# Assemble prediction struct
predictions = {}
for m in range(len(markernames)):
    predictions[markernames[m]] = pred_locked[:, :, m]

predictions['sampleID'] = pred['sampleID']

for i in range(len(cnames)):
    cameras[cnames[i]] = {}

    cameras[cnames[i]]['frame'] = df

    camparams = sio.loadmat(os.path.join(PARENT_PARAMS['CALIBDIR'],
                            PARENT_PARAMS['calib_file'][i]))

    cameras[cnames[i]]['IntrinsicMatrix'] = camparams['K']
    cameras[cnames[i]]['rotationMatrix'] = camparams['r']
    cameras[cnames[i]]['translationVector'] = camparams['t']
    cameras[cnames[i]]['TangentialDistortion'] = camparams['TDistort']
    cameras[cnames[i]]['RadialDistortion'] = camparams['RDistort']

    cameras[cnames[i]]['video_directory'] = \
        os.path.realpath(os.path.join(PARENT_PARAMS['viddir'],
                                      cnames[i]))

# save everything as matlab struct
pfile = os.path.join(RESULTSDIR, 'predictions_dlc.mat')
sio.savemat(pfile,
            {'fullpath': fullpath,
             'name': name,
             'session': date,
             'predictions': predictions,
             'cameras': cameras,
             'netname': 'DLC',
             'p_max': np.zeros_like(pred['p_max'])*np.nan,
             'dlcpath': CONFIG_PARAMS['DLC_PATH']})

print("Saved predictions to " + pfile)

