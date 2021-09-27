#!/bin/bash
# Script to run all steps of com in a single job using multi-gpu prediction. 
#
# Inputs: com_config - path to com config.
#     
# Example: sbatch com_multi_gpu.sh /path/to/com_config.yaml 
#SBATCH --job-name=com_multi_gpu
#SBATCH --mem=10000
#SBATCH -t 5-00:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p common
set -e

# Setup the dannce environment
# module load Anaconda3/5.1.0
. ~/.bashrc
source activate dannce_cuda11

# Commented out to make script DCC compatible
# module load cuda/11.0.3-fasrc01
# module load cudnn/8.0.4.30_cuda11.0-fasrc01

# Train com network
sbatch --wait holy_com_train.sh $1
wait

# Predict with com network in parallel and merge results
com-predict-multi-gpu $1
com-merge $1

