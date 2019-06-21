"""Process labeled data.

Usage: python consolidate_labels.py path_to_label_config path_to_label_template
"""
import os
import json
import numpy as np
import scipy.io as sio
import sys
import ast
from dannce.labeling import json2manifest as json2manifest
from dannce.engine import processing as processing

# load params from config
CONFIG_PARAMS = processing.read_config(sys.argv[1])
print("Loading configuration from: " + sys.argv[1])

annotdir = CONFIG_PARAMS['RESULTSDIR']

# Create manifest from json
json2manifest.j2m(os.path.join(annotdir, 'annotations.json'))

camnames = CONFIG_PARAMS['CAMNAMES']

# Get markernames from template file
with open(sys.argv[2], 'r') as f:
	for line in f:
		if 'labels' in line:
			line = line.strip()
			line = line[8:-1]
			markernames = ast.literal_eval(line)

# Get all manifests
markernames_ = {}
for i in range(len(markernames)):
    markernames_[markernames[i]] = i
manifests = [f for f in os.listdir(annotdir) if 'manifest' in f]
folders = [f for f in os.listdir(annotdir) if os.path.isdir(
	os.path.join(annotdir, f))]

# Parse manifests
manifest = {}
for i in range(len(manifests)):
    with open(os.path.join(annotdir, manifests[i]), 'rb') as f:
        for line in f:
            splits = str(line).lstrip('b\'').split('\\')
            num = int(splits[0])
            imname = splits[1].split('/')[-1]
            manifest[imname] = num

sortedlist = sorted(list(
	manifest.keys()), key=lambda x: int(x.lstrip('sample').split('_')[0]))

# Get labels
nummarkers = len(markernames)
numcams = len(camnames)
marker_i = np.zeros((len(sortedlist) // numcams, nummarkers, numcams))
marker_coords = np.zeros(
	(len(sortedlist) // numcams, nummarkers, numcams, 2)) * np.nan
sampleID = np.zeros((len(sortedlist) // numcams, numcams))
good = []
for i in range(len(sortedlist) // numcams):
    for j in range(numcams):
        t = sortedlist[i * numcams + j]
        # We need to manually append the camera name to make sure it is in
        # the specified order
        t = t.split('Camera')[0] + camnames[j] + '.png'
        # FInd and load correct json
        fpath = os.path.join(annotdir, folders[-1], str(manifest[t]))
        # We didn't label all of the images here
        if os.path.exists(fpath):
            jj = os.listdir(fpath)
            with open(os.path.join(fpath, jj[0])) as f:
                data = json.load(f)
            data = data['answers'][0]['answerContent'][
                'annotatedResult']['keypoints']
            if len(data) > 0:
                # Save the name of this image file so that it can be removed
				# from the original manifest
                good.append(t)
            for k in range(len(data)):
                keypt = data[k]
                n_keypt = markernames_[keypt['label']]
                marker_i[i, n_keypt, j] = 1
                marker_coords[i, n_keypt, j, 0] = keypt['x']
                marker_coords[i, n_keypt, j, 1] = keypt['y']
            sampleID[i, j] = t.split('sample')[1].split('_')[0]

# save the labels, separate files for each cam
for j in range(numcams):
    thiscam = camnames[j]
    # TODO(undefined): RESILTSDIR
    sio.savemat(os.path.join(
        RESULTSDIR, '{}_manlabels.mat'.format(thiscam)),
    	{'data2d': marker_coords[:, :, j, :], 'sampleID': sampleID[:, j]})

print("done!")
