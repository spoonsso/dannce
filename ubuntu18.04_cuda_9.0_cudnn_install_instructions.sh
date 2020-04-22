#!/bin/bash
## This gist contains step by step instructions to install cuda v9.0 and cudnn 7.2 in ubuntu 18.04

### steps ####
# verify the system has a cuda-capable gpu
# download and install the nvidia cuda toolkit and cudnn
# setup environmental variables
# verify the installation
###

### to verify your gpu is cuda enable check
lspci | grep -i nvidia

### gcc compiler is required for development using the cuda toolkit. to verify the version of gcc install enter
gcc --version

# first get the PPA repository driver
# sudo add-apt-repository ppa:graphics-drivers/ppa

# install nvidia driver
# sudo apt install nvidia-384 nvidia-384-dev

# install other import packages
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

# CUDA 9 requires gcc 6
# sudo apt install gcc-6
# sudo apt install g++-6

# downoad one of the "runfile (local)" installation packages from cuda toolkit archive
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run

# make the download file executable
chmod +x cuda_9.0.176_384.81_linux.run
sudo ./cuda_9.0.176_384.81_linux.run --override

# answer following questions while installation begin
# You are attempting to install on an unsupported configuration. Do you wish to continue? y
# Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 384.81? n
# Install the CUDA 9.0 Toolkit? y

# set up symlinks for gcc/g++
sudo ln -s /usr/bin/gcc-6 /usr/local/cuda/bin/gcc
sudo ln -s /usr/bin/g++-6 /usr/local/cuda/bin/g++

# setup your paths
echo 'export PATH=/usr/local/cuda-9.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# install cuDNN v7.2
# in order to download cuDNN you have to be regeistered here https://developer.nvidia.com/developer-program/signup
# then download cuDNN v7.2 form https://developer.nvidia.com/cudnn

# This doesn't always work and I recommend downloading this manually through the website
# CUDNN_TAR_FILE="cudnn-9.0-linux-x64-v7.2.1.38"
# wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.2.1/prod/9.0_20180806/${CUDNN_TAR_FILE}
# tar -xzvf ${CUDNN_TAR_FILE}

# copy the following files into the cuda toolkit directory.
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-9.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64/
sudo chmod a+r /usr/local/cuda-9.0/lib64/libcudnn*

# Finally, to verify the installation, check
nvidia-smi
nvcc -V
