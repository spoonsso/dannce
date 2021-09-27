#!/bin/bash
set -e
# Set of tests to run.
#
# List of tests:
# 1) Comparing layer norm on max net with instance norm on 

python setup.py install

cd tests/configs

# echo "Testing DANNCE training, MAX net from scratch with layer norm"
# dannce-train dgptest_config.yaml --dannce_train_dir=../../demo/markerless_mouse_1/DANNCE/train_test_ln/

# echo "Testing DANNCE training, MAX net from scratch with instance norm"
# dannce-train dgptest_config.yaml --norm_method=instance --dannce_train_dir=../../demo/markerless_mouse_1/DANNCE/train_test_in/

echo "Testing DANNCE training, dgp MAX net from scratch with instance norm and sigmoid cross entropy with gaussian targets"
dannce-train dgptest_config.yaml --norm_method=instance --loss=gaussian_cross_entropy_loss

echo "Finished"
