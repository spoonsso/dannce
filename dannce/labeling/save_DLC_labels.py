"""
Given processed hand label, outptus a directory of images, with marker locations written
to a separate file, to be used for training DLC

Usage: python save_DLC_labels.py labeling_config inkey in_imdir out_imdir bool_reproject [worker #]
"""

import numpy as np
import scipy.io as sio
import os
import sys
import imageio
from dannce.engine import processing as processing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load params from config
CONFIG_PARAMS = processing.read_config(sys.argv[1])
CONFIG_PARAMS = processing.make_paths_safe(CONFIG_PARAMS)
print("Loading configuration from: " + sys.argv[1])

workerID = None
if len(sys.argv) == 7:
    workerID = int(sys.argv[6])

inkey = sys.argv[2]
indir = sys.argv[3]
outdir = sys.argv[4]

bool_reproject = eval(sys.argv[5])

CONFIG_PARAMS['RESULTSDIR'] = os.path.join(CONFIG_PARAMS['RESULTSDIR'], 'labeling')
RESULTSDIR = os.path.join(CONFIG_PARAMS['RESULTSDIR'], indir)
R_O = os.path.join(CONFIG_PARAMS['RESULTSDIR'], outdir)

if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)

if not os.path.exists(R_O):
    os.makedirs(R_O)

# For each camera, load in manlabels file, all associated images. Then plot labels on top of image and save
# the output
plt.figure(figsize=(14,14))
imdir = os.path.join(CONFIG_PARAMS['RESULTSDIR'], indir)
allcoords = []
fnames = []

for cam in CONFIG_PARAMS['CAMNAMES']:
    if workerID is not None:
        r = os.path.join(CONFIG_PARAMS['RESULTSDIR'],cam + inkey + '_worker{}.mat'.format(workerID))
    else:
        r = os.path.join(CONFIG_PARAMS['RESULTSDIR'],cam + inkey + '.mat')
    print("Loading labels from " + r)
    l = sio.loadmat(r)
    if 'data2d' in l.keys():
        lbls = l['data2d']
    elif 'data_2d' in l.keys():
        lbls = l['data_2d']
    else:
        raise Exception("No 2D data in file")

    # The data may need to be reshaped
    if len(lbls.shape) == 2:
        lbls = np.reshape(lbls, [lbls.shape[0], -1, 2])

    skey = [f for f in l.keys() if 'sample' in f]
    skey = skey[0]
    sID = np.squeeze(l[skey])

    for i in range(len(sID)):
        # Load in this image
        imname = 'sample' + str(int(sID[i])) + '_' + cam + '.png'
        impath = os.path.join(imdir,imname)
        im = imageio.imread(impath)

        if bool_reproject:
            lbls[:, :, 0] = lbls[:, :, 0] - CONFIG_PARAMS['CROP_WIDTH'][0]
            lbls[:, :, 1] = lbls[:, :, 1] - CONFIG_PARAMS['CROP_HEIGHT'][0]

        # Append worker ID if necessary
        
        # if workerID is not None:
        #     imname = imname.split('.')[0] + '_worker{}.png'.format(workerID)
        allcoords.append(np.concatenate((np.arange(lbls.shape[1])[:,np.newaxis], lbls[i, :]),axis=1))

        # TODO(os.path): This is unix-specific
        # These paths should be using AWS/UNIX only
        relpath = R_O.split(os.sep)[-1]
        relpath = \
            '..' + os.sep + relpath + os.sep + imname
        fnames.append(relpath)

allcoords = np.stack(allcoords)
sio.savemat(
    os.path.join(R_O,'allcoords.mat'),
    {'allcoords': allcoords,
     'imsize': [im.shape[-1], im.shape[0], im.shape[1]],
     'filenames': fnames})

print("done!")
