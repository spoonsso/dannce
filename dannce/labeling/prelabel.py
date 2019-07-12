"""Randomly sample videos to generate images for labeling.

This script performs all pre-labeling steps, going from videos
to the uploading of images to your amazon bucket to the initiation
of a labeling job

Usage: python prelabel.py path_to_config_file num_images_per_camera
       path_to_s3_folder [create-job-only?]

[create-job-only?] is an optional boolean parameter that can be used to
skip bucket and data creation

"""
import numpy as np
import scipy.io as sio
import os
import sys
import subprocess
from dannce.engine import processing as processing
from dannce.engine.generator_aux import DataGenerator_downsample
from dannce.engine import serve_data_DANNCE as serve_data

import time

apath = sys.argv[3]
# remove path characters (cross-platform)
apath = apath.rstrip('/')
apath = apath.rstrip('\\')
folder = os.path.split(apath)[-1]
bucket = os.path.split(os.path.split(apath)[0])[-1]

# Load params from config
CONFIG_PARAMS = processing.read_config(sys.argv[1])
print("Loading configuration from: " + sys.argv[1])
RESULTSDIR = os.path.join(CONFIG_PARAMS['RESULTSDIR'], 'labeling','imDir') + os.path.sep
print("Saving images to: " + RESULTSDIR)
if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)

# Only generate images and upload to S3 if the create-job-only? flag is True
if len(sys.argv) == 4 or not eval(sys.argv[4]):

    ans = ''
    while ans != 'y' and ans != 'n':
        print("Creating new folder, {}, in s3 bucket {}. Continue? (y/n)".format(folder,bucket))
        ans = input().lower()

    if ans == 'n':
        print("Ok, exiting.")
        sys.exit()

    # Create the folder if it doesn't already exist
    mout = subprocess.Popen(["aws", "s3", "ls", bucket],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)

    stdout, stderr = mout.communicate()
    stdout = str(stdout)
    if 'PRE {}/'.format(folder) in stdout:
        ans = ''
        while ans != 'y' and ans != 'n':
            print("Folder already exists in bucket. Continue anyway (y/n)?")
            ans = input().lower()

        if ans == 'n':
            print("Ok, exiting.")
            sys.exit()
    else:
        subprocess.call(["aws", "s3api", "put-object", "--bucket", 
                         bucket, "--key", folder + '/'])

    if 'seed' in CONFIG_PARAMS.keys():
        np.random.seed(CONFIG_PARAMS['seed'])
    # Load data structure for indices
    if 'calib_file' in CONFIG_PARAMS.keys():
        # then need to handle 5 returns
        samples_, datadict_, datadict_3d_, data_3d_, cc = \
            serve_data.prepare_data(CONFIG_PARAMS, com_flag=False)
    else:
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
        'dim_in': (CONFIG_PARAMS['CROP_HEIGHT'][1]-CONFIG_PARAMS['CROP_HEIGHT'][0],
                   CONFIG_PARAMS['CROP_WIDTH'][1]-CONFIG_PARAMS['CROP_WIDTH'][0]),
        'n_channels_in': CONFIG_PARAMS['N_CHANNELS_IN'],
        'batch_size': 1,
        'n_channels_out': 1,
        'camnames': {0: CONFIG_PARAMS['CAMNAMES']},
        'crop_width': CONFIG_PARAMS['CROP_WIDTH'],
        'crop_height': CONFIG_PARAMS['CROP_HEIGHT'],
        'chunks': CONFIG_PARAMS['chunks'],
        'shuffle': True}

    labels = datadict_
    labels_3d = datadict_3d_

    partition = {}
    partition['train'] = \
        np.random.choice(samples_, size=(int(sys.argv[2]),), replace=False)

    generator = \
        DataGenerator_downsample(partition['train'], labels, vids, **params)

    # generate images.
    # Compression is turned on by default. Because it is lossless png comrpession,
    # I see no reason to enable less compression.
    generator.save_for_dlc(RESULTSDIR)

    # Write manifest file
    folder = sys.argv[3]
    allc = sio.loadmat(os.path.join(RESULTSDIR, 'allcoords.mat'))['filenames']
    with open(os.path.join(
            RESULTSDIR, 'dataset.manifest'), 'w') as f:
        for it in allc:
            fname = it.strip()
            fname = fname.split('/')[-1]
            write_string = "\"source-ref\":\"{}\"".format(folder + fname)
            write_string = "{" + write_string + "}\n"
            f.write(write_string)
    print("done!")

    ans = ''
    while ans != 'y' and ans != 'n':
        print("Uploading necessary files to your S3 bucket. Continue? (y/n)")
        ans = input().lower()

    if ans == 'n':
        print("Ok, exiting.")
        sys.exit()

    R_AWS = sys.argv[3]
    R_AWS = R_AWS.rstrip('/')
    R_AWS = R_AWS + '/'

    subprocess.call(["aws", "s3", "cp",
                     os.path.join(RESULTSDIR),
                     R_AWS, "--recursive"])
    subprocess.call(["aws", "s3", "cp",
                     CONFIG_PARAMS['lbl_template'],
                     R_AWS])

# Create labeling job bash script. 
# When ready to label, call the generated script from the terminal
print("Creating bash script file that can used to start the labeling job on Amazon SageMaker...")

# If on Windows, need to change \ to / for AWS
R_AWS = sys.argv[3]
R_AWS = R_AWS.rstrip('/')
R_AWS = R_AWS + '/'

# Get template name only
lbl_template = CONFIG_PARAMS['lbl_template']
lbl_template = lbl_template.replace('\\', '/')
lbl_template = lbl_template.split('/')[-1]

if 'WorkteamArn' not in CONFIG_PARAMS.keys() or \
   'PreHumanTaskLambdaArn' not in CONFIG_PARAMS.keys() or \
   'AnnotationConsolidationLambdaArn' not in CONFIG_PARAMS.keys() or \
   'role-arn' not in CONFIG_PARAMS.keys():
    raise Exception("Missing required ARNs in config file. Cannot set up labeling script.")

# Name the labeling job as the project directory + timestamp
jobname = os.getcwd().split(os.path.sep)[-1]
# Cannot use _
jobname = jobname.replace('_', '-')
jobname = jobname + '-' + str(int(time.time()))
# Format the long bash script string
base = "#!/bin/bash\n" + \
    "# Creates labeling job\n" + \
    "\n" + \
    "aws sagemaker create-labeling-job" + \
    " --labeling-job-name {}".format(jobname) + \
    " --label-attribute-name keypoint" + \
    " --input-config" + \
    " DataSource={{S3DataSource={{ManifestS3Uri={}}}}}".format(R_AWS + 'dataset.manifest') + \
    " --output-config S3OutputPath={}".format(R_AWS + 'output/') + \
    " --role-arn {}".format(CONFIG_PARAMS['role-arn']) + \
    " --human-task-config '{\n" + \
    "\"WorkteamArn\": \"{}\", ".format(CONFIG_PARAMS['WorkteamArn']) + \
    "\"UiConfig\": { \n" + \
    "\"UiTemplateS3Uri\": \"{}\"}},\n".format(R_AWS + lbl_template) + \
    "\"PreHumanTaskLambdaArn\": \"{}\",\n".format(CONFIG_PARAMS['PreHumanTaskLambdaArn']) + \
    "\"TaskKeywords\": [\"labeling\"],\n" + \
    "\"TaskTitle\": \"labeling\",\n" + \
    "\"TaskDescription\": \"Select unoccluded keypoints\",\n" + \
    "\"NumberOfHumanWorkersPerDataObject\": 1,\n" + \
    "\"TaskTimeLimitInSeconds\": 28800,\n" + \
    "\"TaskAvailabilityLifetimeInSeconds\": 28800,\n" + \
    "\"MaxConcurrentTaskCount\": 1000,\n" + \
    "\"AnnotationConsolidationConfig\": {\n" + \
    "\"AnnotationConsolidationLambdaArn\": \"{}\"\n".format(CONFIG_PARAMS['AnnotationConsolidationLambdaArn']) + \
    "}\n" + \
    "}\'"

with open(os.path.join(RESULTSDIR, 'create_job.sh'), 'w') as f:
    f.write(base)
