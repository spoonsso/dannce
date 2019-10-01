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

# This is agnostic to the EXPVAL setting, i.e. MAX or AVG net
# However, we will eventually add the COM back in only for MAX-type results
dfiles = os.listdir(RESULTSDIR)
sfile = [f for f in dfiles if 'save_data' in f]
sfile = sfile[0]

pred = sio.loadmat(os.path.join(RESULTSDIR,sfile))

pred['sampleID'] = np.squeeze(pred['sampleID'])

if not CONFIG_PARAMS['EXPVAL']:
    print("adding 3D COM back in")
    com3d = sio.loadmat(os.path.join(RESULTSDIR, 'COM3D_undistorted.mat'))
    # We should make sure the sampleIDs match up
    #assert np.all(com3d['sampleID'] == pred['sampleID'])
    # We need to loop over all, double check sampleID alignment, and then add com back
    for j in range(len(pred['sampleID'].ravel())):
        comind = np.where(com3d['sampleID']==pred['sampleID'][j])[0]
        assert len(comind) == 1
        pred['pred'][j] = pred['pred'][j] + com3d['com'][j, :, np.newaxis]

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
netname = CONFIG_PARAMS['net']

# Get the weights path, a useful piece of metadata
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

weightspath = mdl_file

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

# Assemble prediction struct
predictions = {}
for m in range(len(markernames)):
    predictions[markernames[m]] = pred['pred'][:, :, m]

predictions['sampleID'] = pred['sampleID']

cameras = {}

cnames = PARENT_PARAMS['CAMNAMES']
for i in range(len(cnames)):
    cameras[cnames[i]] = {}

    # Load in this camera's matched frames file to align the sampleID 
    # with that particular cameras' frame #
    mframes = sio.loadmat(os.path.join(PARENT_PARAMS['datadir'],
                          PARENT_PARAMS['datafile'][i]))

    mframes['data_sampleID'] = np.squeeze(mframes['data_sampleID'])
    mframes['data_frame'] = np.squeeze(mframes['data_frame'])

    df = np.zeros((len(pred['sampleID']), 1))

    for j in range(df.shape[0]):
        if pred['sampleID'][j] in mframes['data_sampleID']:
            df[j] = mframes['data_frame'][mframes['data_sampleID'] == pred['sampleID'][j]]
        else:
            raise Exception("Could not find sampleID in matched frames")

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
pfile = os.path.join(RESULTSDIR, 'predictions.mat')
sio.savemat(pfile,
            {'fullpath': fullpath,
             'name': name,
             'session': date,
             'netname': netname,
             'predictions': predictions,
             'cameras': cameras,
             'p_max': pred['p_max'],
             'weightspath': mdl_file})

print("Saved predictions to " + pfile)

