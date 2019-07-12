"""
Given processed hand label, outptus a directory of images with markers overlaid, for inspection.

Usage: python review_labels.py labeling_config inkey in_imdir out_imdir bool_reproject [worker #]
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
print("Loading configuration from: " + sys.argv[1])

workerID = None
if len(sys.argv) == 7:
    workerID = int(sys.argv[6])

inkey = sys.argv[2]
indir = sys.argv[3]
outdir = sys.argv[4]

bool_reproject = eval(sys.argv[5])

CONFIG_PARAMS['RESULTSDIR'] = os.path.join(CONFIG_PARAMS['RESULTSDIR'], 'labeling')
RESULTSDIR = os.path.join(CONFIG_PARAMS['RESULTSDIR'], outdir)
print("Saving images to: " + RESULTSDIR)
if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)

# For each camera, load in manlabels file, all associated images. Then plot labels on top of image and save
# the output
plt.figure(figsize=(14,14))
imdir = os.path.join(CONFIG_PARAMS['RESULTSDIR'], indir)
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

        plt.cla()
        plt.imshow(im)
        lbls[np.isnan(lbls)] = 50
        if bool_reproject:
            plt.scatter(lbls[i, :, 0]-CONFIG_PARAMS['CROP_WIDTH'][0],
                     lbls[i, :, 1]-CONFIG_PARAMS['CROP_HEIGHT'][0],c=np.arange(lbls.shape[1]),marker='o',cmap='Dark2')
        else:
            plt.scatter(lbls[i, :, 0], lbls[i, :, 1], marker='o',c=np.arange(lbls.shape[1]),cmap='Dark2')

        # Save the image with markers overlaid
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # Append worker ID if necessary
        
        if workerID is not None:
            imname = imname.split('.')[0] + '_worker{}.png'.format(workerID)

        plt.savefig(os.path.join(RESULTSDIR,imname), bbox_inches = 'tight',pad_inches = 0)

print("done!")
