"""Process labeled data.

Usage: python postlabel.py path_to_label_config [worker #]

Optional parameter: [worker #]. Default 0 for a single worker. When multiple
workers have labeled each image, pass this argument (int) to select individual
workers
"""
import os
import json
import numpy as np
import scipy.io as sio
from scipy.misc import comb
import sys
import ast
from dannce.labeling import json2manifest as json2manifest
from dannce.engine import processing as processing
from dannce.engine import ops as ops
import matlab
import matlab.engine
import subprocess

# load params from config
CONFIG_PARAMS = processing.read_config(sys.argv[1])
print("Loading configuration from: " + sys.argv[1])

workerID = 0
if len(sys.argv) == 4:
    #Then update default worker ID
    workerID = int(sys.argv[3])

annotdir = os.path.join(CONFIG_PARAMS['RESULTSDIR'], 'labeling')
iterdir = os.path.join(annotdir, 'iteration-1')

print("Making new results directory at " + iterdir)
if not os.path.exists(iterdir):
    os.makedirs(iterdir)

createjob = os.path.join(annotdir, 'imDir', 'create_job.sh')
print("Getting S3 output uri from " + createjob)
with open(createjob) as f:
    for line in f:
        if 'S3OutputPath' in line:
            uri = line.split('S3OutputPath=')[-1].split('--role-arn')[0]
            uri = uri.rstrip()
        if 'labeling-job-name' in line:
            jobname = line.split('labeling-job-name ')[-1]
            jobname = jobname.split(' --label-attribute-name')[0]

uri_labels = uri + jobname + '/annotations/worker-response/iteration-1/'
uri_annot = uri + jobname + '/annotations/intermediate/1/annotations.json'

ans = ''
while ans not in ['y', 'n', 'skip']:
    print('Downloading labels from ' + uri_labels + '. Continue? (y/n/skip)')
    ans = input().lower()

if ans == 'y':
    subprocess.call(["aws", "s3", "cp", uri_annot, annotdir])
    subprocess.call(["aws", "s3", "cp", uri_labels, iterdir, "--recursive"])
elif ans == 'n':
    print("OK, exiting.")
    sys.exit()

# Create manifest from json
json2manifest.j2m(os.path.join(annotdir, 'annotations.json'))

camnames = CONFIG_PARAMS['CAMNAMES']

# Get markernames from template file
with open(CONFIG_PARAMS['lbl_template'], 'r') as f:
    for line in f:
        if 'labels' in line:
            line = line.strip()
            line = line[8:-1]
            markernames = ast.literal_eval(line)

# Get all manifests
markernames_ = {}
for i in range(len(markernames)):
    markernames_[markernames[i]] = i
manifests = [f for f in os.listdir(annotdir) if 'manifest' in f and 'annot' in f]
folders = [f for f in os.listdir(annotdir) if os.path.isdir(
    os.path.join(annotdir, f)) and 'iteration' in f]

# Parse manifests
manifest = {}
for i in range(len(manifests)):
    print(os.path.join(annotdir, manifests[i]))
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
        t = t.split('_')[0] + '_' + camnames[j] + '.png'
        # FInd and load correct json
        fpath = os.path.join(annotdir, folders[-1], str(manifest[t]))
        # If we didn't label all of the images here
        if os.path.exists(fpath):
            jj = os.listdir(fpath)
            with open(os.path.join(fpath, jj[0])) as f:
                data = json.load(f)
            data = data['answers'][workerID]['answerContent'][
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

    sio.savemat(os.path.join(
        annotdir, '{}_manlabels_worker{}.mat'.format(thiscam, workerID)),
        {'data2d': marker_coords[:, :, j, :], 'sampleID': sampleID[:, j]})

undistort_coords = np.zeros_like(marker_coords)
undistort_coords = np.reshape(undistort_coords, (-1, numcams, 2))

cammats = {}
cammats_sep = {}
print("Unprojecting...")
# Undistort the labels
for j in range(numcams):
    # Load in this camera's params
    cammat = sio.loadmat(os.path.join(CONFIG_PARAMS['CALIBDIR'],
                                      CONFIG_PARAMS['calib_file'][j]))
    # Do the undistortion. We don't call our other Matlab script directly because
    # here we have few enough points that the undistortion won't take too long
    mc = np.reshape(marker_coords[:, :, j, :], (-1, 2)).copy()
    mc[:, 0] += CONFIG_PARAMS['CROP_WIDTH'][0]
    mc[:, 1] += CONFIG_PARAMS['CROP_HEIGHT'][0]

    # Mask NaNs
    inds = np.where(np.isnan(mc[:, 0]))[0]
    mc[inds, :] = 0

    undistort_coords[:, j, :] = ops.unDistortPoints(
                                mc,
                                cammat['K'],
                                np.squeeze(cammat['RDistort']),
                                np.squeeze(cammat['TDistort']),
                                np.squeeze(cammat['r']),
                                np.squeeze(cammat['t']))

    # Replace NaNs
    undistort_coords[inds, j, :] = np.nan

    cammats_sep[j] = {}
    cammats_sep[j]['K'] = cammat['K']
    cammats_sep[j]['r'] = cammat['r']
    cammats_sep[j]['t'] = cammat['t']
    cammats_sep[j]['RDistort'] = np.squeeze(cammat['RDistort'])
    cammats_sep[j]['TDistort'] = np.squeeze(cammat['TDistort'])

    cammats[j] = ops.camera_matrix(cammat['K'], cammat['r'], cammat['t'])


data3d = np.zeros((undistort_coords.shape[0], int(comb(numcams, 2)), 3))
cnt = 0
print("Triangulating...")
# Now we triangulate each pair
for i in range(numcams):
    for j in range(i+1, numcams):
        pts1 = undistort_coords[:, i, :].copy()
        pts2 = undistort_coords[:, j, :].copy()

        # mask NaNs again, this time across both arrays
        inds = np.where(np.isnan(pts1[:, 0]) | np.isnan(pts2[:, 0]))[0]
        pts1[inds, :] = 0
        pts2[inds, :] = 0

        test3d = ops.triangulate(
                        pts1, pts2, cammats[i],
                        cammats[j]).squeeze().T

        test3d[inds, :] = np.nan

        data3d[:, cnt, :] = test3d
        cnt = cnt + 1

# Take the median vector across all pairs
data3d = np.nanmedian(data3d, axis=1)
wpts = data3d.copy()
# Reshape back
data3d = np.reshape(data3d, (marker_coords.shape[0], -1, 3))
# # The saved data3d is expected to come reshaped in a specific manner
data3d = np.transpose(data3d, [0, 2, 1])
data3d = np.reshape(data3d, (data3d.shape[0], -1), 'F')

print("Reprojecting 3D and saving...")
# Save data structures with reprojection
for j in range(numcams):

    # To make sure these frames stay synchronized, we need to load in
    # the original data files, which should be in the config

    df = sio.loadmat(os.path.join(CONFIG_PARAMS['datadir'],
                                  CONFIG_PARAMS['datafile'][j]))

    data_frame = np.zeros((sampleID.shape[0],))
    for i in range(sampleID.shape[0]):
        data_frame[i] = df['data_frame'][df['data_sampleID'] == sampleID[i, j]]

    # Now do the reprojection
    wpts2d = ops.project_to2d(wpts, cammats_sep[j]['K'],
                              cammats_sep[j]['r'],
                              cammats_sep[j]['t'])
    wpts2d = wpts2d[:, :2]

    # Apply distortion
    wpts2d = ops.distortPoints(wpts2d, cammats_sep[j]['K'],
                               cammats_sep[j]['RDistort'],
                               cammats_sep[j]['TDistort']).T

    # Reshape twice to get the correct format
    wpts2d = np.reshape(wpts2d, (marker_coords.shape[0], -1, 2))


    wpts2d = np.transpose(wpts2d, [0, 2, 1])
    wpts2d = np.reshape(wpts2d, (marker_coords.shape[0], -1), 'F')

    # Now save

    sio.savemat(os.path.join(
        annotdir, '{}_dataApplyDistort_worker{}.mat'.format(camnames[j], workerID)),
        {'data_2d': wpts2d, 
         'data_sampleID': sampleID[:, j],
         'data_frame': data_frame,
         'data_3d': data3d})

print("done!")
