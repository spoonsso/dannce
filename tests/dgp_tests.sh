#!/bin/bash
set -e
# Set of tests to run.
#
# List of tests:
# 1) Comparing layer norm on max net with instance norm on 

python setup.py install

cd tests/configs
cp ./label3d_temp_dannce.mat ./alabel3d_temp_dannce.mat
echo "Testing DANNCE training, MAX net from scratch with layer norm"
dannce-train dgptest_config.yaml --dannce-train-dir=../../demo/markerless_mouse_1/DANNCE/train_test_ln/ --n-channels-out=22

echo "Testing DANNCE training, MAX net from scratch with instance norm"
dannce-train dgptest_config.yaml --norm-method=instance --n-channels-out=22 --dannce-train-dir=../../demo/markerless_mouse_1/DANNCE/train_test_in/

# echo "Testing DANNCE training, dgp MAX net from scratch with layer norm and sigmoid cross entropy with gaussian targets"
# dannce-train dgptest_config.yaml --loss=gaussian_cross_entropy_loss --n-channels-out=22 --dannce-train-dir=../../demo/markerless_mouse_1/DANNCE/train_test_ln_dgp/

echo "Testing DANNCE training, dgp MAX net from scratch with instance norm and sigmoid cross entropy with gaussian targets"
dannce-train dgptest_config.yaml --norm-method=instance --loss=gaussian_cross_entropy_loss --n-channels-out=22

echo "Finished"
