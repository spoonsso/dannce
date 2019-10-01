"""Takes the median of triangulated 3D COM predictions,
thresholds away crazy outliers, reprojects the 3D projections
into each camera, then saves COM data files that can be used for training.

Usage python generate_COM_bootstraps.py path_to_config com_thresh [max_num_samples]
"""
import sys
import numpy as np
import os
import dannce.engine.serve_data_DANNCE as serve_data
import dannce.engine.processing as processing
import dannce.engine.ops as ops
import scipy.io as sio

_N_VIEWS = 6

# Set up parameters
PARENT_PARAMS = processing.read_config(sys.argv[1])
PARENT_PARAMS = processing.make_paths_safe(PARENT_PARAMS)
CONFIG_PARAMS = processing.read_config(PARENT_PARAMS['DANNCE_CONFIG'])
CONFIG_PARAMS = processing.make_paths_safe(CONFIG_PARAMS)

com_thresh = float(sys.argv[2])
# If desired, crop the data at max_num_samples
if len(sys.argv) > 3:
    max_num_samples = int(sys.argv[3])

# While we can use experiment files for DANNCE training, 
# for prediction we use the base data files present in the main config

CONFIG_PARAMS['experiment'] = PARENT_PARAMS
RESULTSDIR = './COM/bootstrapdata/'
print("Writing projected median, com3d to: " + RESULTSDIR)

if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)

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

datadict_, com3d_dict_ = serve_data.prepare_COM(
    comfn,
    datadict_,
    comthresh=com_thresh,
    weighted=False,
    retriangulate=True,
    camera_mats=cameras_,
    method='median')

# Need to cap this at the number of samples included in our
# COM finding estimates
tf = list(com3d_dict_.keys())
samples_ = samples_[:len(tf)]
data_3d_ = data_3d_[:len(tf)]
if len(sys.argv) > 3:
    samples_ = samples_[:max_num_samples]

    data_3d_ = data_3d_[:max_num_samples]
pre = len(samples_)
samples_, data_3d_ = \
    serve_data.remove_samples_com(samples_, data_3d_, com3d_dict_, rmc=True, cthresh=CONFIG_PARAMS['cthresh'])
msg = "Detected {} bad COMs and removed the associated frames from the dataset"
print(msg.format(pre - len(samples_)))

# OK, now it's just a matter of projecting the 3D COMs down and then 
# saving everything in the proper format w/ proper shape
for i in range(len(PARENT_PARAMS['CAMNAMES'])):
    tcam_name = PARENT_PARAMS['CAMNAMES'][i]
    tcam = cameras_[tcam_name]

    data_3d = np.zeros((len(samples_), 3))
    data_2d = np.zeros((len(samples_), 2))
    data_frame = np.zeros((len(samples_),))
    data_sampleID = np.zeros((len(samples_),))

    for k in range(len(samples_)):
        s = samples_[k]
        t3d = com3d_dict_[s]
        # Project down for this camera
        t2d = ops.project_to2d(t3d[:, np.newaxis].T,
                               tcam['K'],
                               tcam['R'],
                               tcam['t'])

        t2d = t2d[:, :2]

        # And distort
        t2d = ops.distortPoints(t2d,
                                tcam['K'],
                                np.squeeze(tcam['RDistort']),
                                np.squeeze(tcam['TDistort']))

        # Just duplicate this and save to data_3d
        data_3d[k] = t3d
        data_2d[k] = np.squeeze(t2d)
        data_sampleID[k] = s
        data_frame[k] = datadict_[s]['frames'][tcam_name]

    # make sure data_2d and data_3d match expected diemnsionality for future use
    data_3d = np.tile(data_3d, (1, CONFIG_PARAMS['N_CHANNELS_OUT']))
    data_2d = np.tile(data_2d, (1, CONFIG_PARAMS['N_CHANNELS_OUT']))

    # Save for this camera
    fname = os.path.join(RESULTSDIR, tcam_name + '_bootstrapDATA.mat')

    print("Saving: " + fname)

    # if len(sys.argv) > 3:
    #     data_2d = data_2d[:max_num_samples]
    #     data_3d = data_3d[:max_num_samples]
    #     data_sampleID = data_sampleID[:max_num_samples]
    #     data_frame = data_frame[:max_num_samples]

    sio.savemat(fname, {'data_2d': data_2d,
                        'data_3d': data_3d,
                        'data_sampleID': data_sampleID,
                        'data_frame': data_frame})

print("done!")
