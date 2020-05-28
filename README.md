![Image](./common/dannce_logo.png)

Repository Contributors: Timothy Dunn, Jesse Marshall, Diego Aldarondo, William Wang, Kyle Severson 

DANNCE (3-Dimensional Aligned Neural Network for Computational Ethology) is a convolutional neural network (CNN) that extracts the 3D positions of user-defined anatomical keypoints from videos of behaving animals. The key innovations of DANNCE compared to existing approaches for 2D keypoint detection in animals (e.g. LEAP, DeepLabCut) are that (1) the network is fully 3D, so that it can learn more abstract 3D features about how cameras and keypoints relate to one another and (2) we pre-train DANNCE using a large dataset of rat motion capture and synchronized video, so that the network learns what rodents look like ahead of time. The network's ability to track keypoints transfers well to mice and other mammals, and works across different camera views, camera types, and illumination conditions. To use DANNCE, first you must collect video recordings of animals from at least two views using synchronized, calibrated cameras. Calibrating cameras is a straightforward procedure, with toolboxes available in Matlab and Python, and we include our calibration scripts below. After acquisition, one can use DANNCE to detect keypoints. The DANNCE algorithm is currently configured as a two-stage system. First, videos are processed to extract the overall position of the animal in each video frame. Then, these positions are used to create *unprojected* 3D image volumes for each view that contain the animal. These image volumes are used as input to the CNN to make keypoint predictions.

![Image](./common/Figure1.png)

## Examples

#### Mouse
![Image](./common/KyleMouse_longfast.gif)

#### Mouse (slowmo)
![Image](./common/KyleMouse_shortslow.gif)

#### Rat
![Image](./common/rat_JDM52.gif)

## Camera Calibration
To use DANNCE, acquisition cameras must be synchronized, calibrated, and ideally compressed. Synchronization is best done with a frametime trigger and a supplementary readout of frame times. Calibration is the process of determining the distortion introduced into an image from the camera lens (Camera Intrinsics), and the relative position and orientation of cameras to one another in space (Camera Extrinsics). We calibrate cameras in a two-step process, where first we use a checkerboard to find the camera intrinsics, and then an 'L-frame' to determine the camera extrinsics. The L-frame is a calibrated grid of four or more points that are labeled in each camera. A checkerboard can also be used for both procedures. We have included two examples of calibration in Matlab (in `Calibration/`), one that is a long script, and a second that performs the steps independently. 

Some tips:
1) Try to sample the whole volume of the arena with the checkerboard to fully map the distortion of the lenses.
2) If you are using a confined arena (eg a plexiglass cylinder) that is hard to wand, it often works to compute the calibration without the cylinder present.
3) More complicated L-Frames can be used, and can help, for computing the extrinsics. Sometimes using only a four point co-planar L-frame can result in a 'flipped' camera, so be sure to check camera poses after calibration. 

It is often helpful to compress videos as they are acquired to reduce diskspace needed for streaming long recordings from multiple cameras. This can be done using ffmpeg or x264, and we have included two example scripts in `Compression/`. One, `campy.py`, was written by Kyle Severson and runs ffmpeg compression on a GPU for streaming multiple Basler cameras. A second, CameraCapture was originally written by Raj Poddar and uses x264 on the CPU to stream older Point Grey/FLIR cameras (eg Grasshopper, Flea3). We have included both a compiled version of the program and the original F-Sharp code that can be edited in Visual Studio. 

Mirrors. Mirrors are a handy way to create new views, but there are some important details when using them with DANNCE. The easiest way to get it all to work with the dannce pipeline is to create multiple videos from the video with mirrors, with all but one sub-field of view (FOV) blacked out in each video. This plays well with the center-of-mass finding network, which currently expects to find only one animal in a given frame.

When calibrating the mirror setup, we have used one intrinsic parameter calibration over the entire FOV of the camera, typically by moving the experimental setup away from the camera (moving the camera could cause changes in the intrinsics). We used these intrinsic parameters for all sub-FOVs. We also used the one set of distortion parameters to undistort the entire FOV. After you have these the parameters, you take images of the calibration target with the mirrors in place and calibrate the extrinsics for each sub-FOV independently and go from there.

Cameras tested:
1) Point Grey Flea3
2) Blackfly  BFS-U3-162M/C-CS
3)  Basler Aca1920-155uc, Aca640-750um, Aca720-510um



## DANNCE Installation

The following combinations of operating systems, python, tensorflow, cuda, and cudnn distributions have been used for development.

|      OS      | python | tensorflow | cuda | cudnn | pytorch |
|:------------:|:------:|:----------:|:----:|:-----:|:-------:|
| Ubuntu 16.04 |  3.7.x |   2.2.0   |  10.1 |  7.6  |  1.5.0  |
|  Windows 10  |  3.7.x |   2.2.0   |  10.1 |  7.6  |  1.5.0  |

`DANNCE` requires a CUDA-enabled GPU and appropriate drivers. We have tested DANNCE on NVIDIA Titan V, Titan X Pascal, Titan RTX, and V100. On an NVIDIA Titan V, DANNCE can make predictions at ~10.5 samples per second when using 6 cameras. DANNCE is also embarrassingly parallel over multiple GPUs. 

We recommend installing `DANNCE` in the following steps:

1. Set up a new conda environment with the following configuration: `conda create -n danncetemp python=3.7 cudatoolkit=10.1 cudnn`

2. Install dannce with the included setup script by running `python setup.py install` within the base dannce repo folder.

You should be good to go!

## Demo Quickstart
To test your DANNCE installation and familiarize yourself with DANNCE file and configuration formatting, run DANNCE predictions over `markerless_mouse_1`. Because the videos and network weights files are too large to host on GitHub, use the links in `demo/markerless_mouse_1/DANNCE/train_results/link_to_weights.txt`, `demo/markerless_mouse_1/DANNCE/AVG/train_results/link_to_weights.txt`, `demo/markerless_mouse_1/videos/link_to_videos.txt` to download necessary files and place them in each associated location.

Alternatively, on Linux you can run the following commands from the base dannce directory:

`wget -O vids.zip https://www.dropbox.com/sh/wn1x8erb5k3n9vr/AADE_Ca-2farKhd38ZvsNi84a?dl=1 & unzip vids.zip -d vids ; mv vids/* demo/markerless_mouse_1/videos/ ; rm -r vids vids.zip ; wget -O demo/markerless_mouse_1/DANNCE/train_results/AVG/weights.1200-12.77642.hdf5 https://www.dropbox.com/s/4b97fg5ciznllnt/weights.1200-12.77642.hdf5?dl=1 ; wget -O demo/markerless_mouse_1/DANNCE/train_results/weights.12000-0.00014.hdf5 https://www.dropbox.com/s/wnjlfhylaxtecax/weights.12000-0.00014.hdf5?dl=1`

Once the files are downloaded and in their correct places, run:

`cd demo/markerless_mouse_1`

`python ../../predict_DANNCE.py config.yaml`

This demo will run the `AVG` version of DANNCE (more on `AVG` vs. `MAX1` below) over 1000 frames of mouse data and save the results to `demo/markerless_mouse_1/DANNCE/predict_results/save_data_AVG.mat`

## Formatting The Data
During training and evaluation, DANNCE requires a set of videos across multiple views, a camera calibration parameters file, and a "MatchedFrames" file that indexes the videos to ensure synchrony.

We recommend setting up individual project folders for each video recording session, as in the dannce demos (`./demo`), although the dannce configuration files are flexible enough to support more custom file and directory organizations. The demo project folders also contain examples of all of the following formatting information.

**configuration files**

`DANNCE` uses 5 different types of configuration files

- *main config*, e.g. `demo/markerless_mouse_1/config.yaml`. This file coordinates pathing to required files, including videos, camera calibrations, and synchronization files.
- *COM config*, e.g. `demo/markerless_mouse_1/COM/config.yaml`. This file sets parameters for COMfinder training and prediction and sets output directories.
- *DANNCE config*, e.g. `demo/markerless_mouse_1/DANNCE/config_AVG.yaml`. This file sets parameters for DANNCE training and prediction and sets output directories.
- *COM experiment configs*, e.g. `demo/markerless_mouse_1/COM/exp1.yaml` and `demo/markerless_mouse_1/COM/exp2.yaml`. These files coordinate pathing to animal-specific files, including hand-labeled data, enabling COMfinder training over multiple animals.
- *DANNCE experiment configs*, e.g. `demo/markerless_mouse_1/DANNCE/exp1.yaml` and `demo/markerless_mouse_1/DANNCE/exp2.yaml`. These files coordinate pathing to animal-specific files, including hand-labeled data, enabling DANNCE training over multiple animals.

**video directories**.
DANNCE requires a parent video directory with *n* sub-directories, one for each of *n* cameras. Within each subdirectory, videos must be named according the frame index of the first frame in the file. For example, for a three-camera system, the video directory must look like:

./videos/

+-- Camera1

|\_\_+--0.mp4

+-- Camera2

|\_\_+--0.mp4

+-- Camera3

|\_\_+--0.mp4

DANNCE can also accommodate an additional level of subdirectories if `vid_dir_flag` is set to `False` in the main configuration files.

./videos/

+-- Camera1

|\_\_+--3503723726252562

|\_\_\_\_\_+--0.mp4

+-- Camera2

|\_\_+--3503723999451111

|\_\_\_\_\_+--0.mp4

+-- Camera3

|\_\_+--3503723711118999

|\_\_\_\_\_+--0.mp4

**camera calibration parameters**.
DANNCE requires a .mat file for each camera containing the camera's rotation matrix, translation vector, intrinsic matrix, radial distortion, and tangential distortion. To convert from Jesse's calibration format to the required format, use `utils/convert_calibration.m`.

A properly formatted calibration file has the following fields, `['R','t','K','RDistort','TDistort']`.

**matched frames file**.
To ensure that individual video frames are synchronized at each time point, DANNCE requires an array that associates each time point (in any unit) to an associated video frame in each camera. During dataset formatting, these indices are combined with any available training labels to form the core data representation. For making predictions with DANNCE, these training labels are ignored, although DANNCE still expects to find placeholder label arrays.

**Matched Frames**.
`DANNCE` requires a set of "MatchedFrames" files, one for each camera, which delineate frame synchrony across the different cameras over time. If you know your cameras are reliably synchronized at all times (e.g. via hardware triggering), these files can be generated with the aid 

Use this script if you know your camera frames are triggered and reliably synchronized.

If your cameras are not natively synchronized, but you can collect timestaps for each
    frame, MatchedFrames files should be generated by `preprocess_data.M`, together
    with a formatted `.mat` file listing the frameID for each camera and each timepoint.
    See `/dannce/utils/example_matchedframs.mat` file for how these timestamp data
    should be formatted.

## Predicting Keypoints With DANNCE

Making predictions with a trained DANNCE network requires 3 steps.

#### 1) Format the data (see above)
#### 2) Find the animal center of mass (COM) in each video frame
You can use your favorite method to find an estimate of the animal COM in each frame. We trained a U-Net to do it. To find the COM with our U-Net, run `predict_COMfinder.py`. This will generate a COM file that is used by DANNCE.

#### 3) Run the everything through DANNCE
Given formatted data and a COM file, DANNCE can now predict keypoints from your video streams.
To do this, run `predict_DANNCE.py`. See the python file for specific usage instructions. 

We have configured DANNCE to work best with a specific organization of data directories: 

./MyProject/

+-- videos

|\_\_+--Camera1

|\_\_\_\_\_+--0.mp4

|\_\_+--Camera2

|\_\_\_\_\_+--0.mp4

+-- data

|\_\_+--Camera1_MatchedFrames.mat

|\_\_+--Camera2_MatchedFrames.mat

+-- calibration

|\_\_+--Camera1_params.mat

|\_\_+--Camera2_params.mat

See the demos for more details. 

## Hand-Labeling
For fine-tuning DANNCE to work with your animal and system, we developed a labeling GUI, which can be found in a separate repo: https://github.com/diegoaldarondo/Label3D. The `Label3D` repository should be cloned with `DANNCE` automatically as a submodule when using `git clone --recursive https://github.com/spoonsso/DANNCE` When labeling is completed, the labels can be used to train DANNCE (see below).

## Training The COMfinder U-Net
DANNCE requires a reasonable estimate of the 3D position of the animal in each frame. We obtain this by triangulating the 2D COM of the animal in each frame. Our U-Net is brittle and typically requires some additional training data to get it working on new views, new environments, and new species. If working with hand-labeled data, your same data structures can be used to train the COMfinder network.

Given formatted data, a properly organized directory structure, and a config file (see demo folder), navigate to your project folder and run

`python $my_path_to_DANNCE/train_COMfinder.py ./config.yaml`, where `$my_path_to_DANNCE` is a path to the root DANNCE directory.

After training, run `python $my_path_to_DANNCE/predict_COMfinder.py ./config.yaml`. To generate center of mass predictions.

## Training DANNCE

Once the COM is found, the main DANNCE network can be trained by running:

`python $my_path_to_DANNCE/train_DANNCE.py ./config.yaml`

After training, run

`python $my_path_to_DANNCE/predict_DANNCE.py ./config.yaml` to make 3D predictions using the trained model.

Consult the demo folder for directory and config file formatting examples




