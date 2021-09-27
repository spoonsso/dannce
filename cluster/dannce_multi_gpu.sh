#!/bin/bash
# Script to run all steps of dannce in a single job using multi-gpu prediction. 
#
# Inputs: dannce_config - path to com config.
# Example: sbatch dannce_multi_gpu.sh /path/to/dannce_config.yaml
#SBATCH --job-name=dannce_multi_gpu
#SBATCH --mem=15000
#SBATCH -t 5-00:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p common
set -e

# Setup the dannce environment
# module load Anaconda3/5.1.0
. ~/.bashrc
source activate dannce_cuda11

# Commented to make script DCC compatible
# module load cuda/11.0.3-fasrc01
# module load cudnn/8.0.4.30_cuda11.0-fasrc01

# Train dannce network
sbatch --wait holy_dannce_train.sh $1
wait

# Predict with dannce network in parallel and merge results
dannce-predict-multi-gpu $1
dannce-merge $1
