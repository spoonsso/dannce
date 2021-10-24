#!/bin/bash
#SBATCH --job-name=jhwTest
#SBATCH --mem=60000
#SBATCH -t 6-23:59
#SBATCH -N 1
#SBATCH -n 12
#SBATCH -p tdunn
#SBATCH --gres=gpu:4

source activate dannce_test

dannce-train dgptest_config.yaml --dannce-train-dir=./DANNCE/train_test_ln/

dannce-train dgptest_config.yaml --norm-method=instance --dannce-train-dir=./DANNCE/train_test_in/

# echo "Testing DANNCE training, dgp MAX net from scratch with layer norm and sigmoid cross entropy with gaussian targets"
# dannce-train dgptest_config.yaml --loss=gaussian_cross_entropy_loss --n-channels-out=22 --dannce-train-dir=./DANNCE/train_test_ln_dgp/

dannce-train dgptest_config.yaml --norm-method=instance --loss=gaussian_cross_entropy_loss