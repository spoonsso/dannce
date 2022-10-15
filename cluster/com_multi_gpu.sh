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
#SBATCH -p olveczky
set -e

# Setup the dannce environment
module load Anaconda3/5.0.1-fasrc02
module load ffmpeg/4.0.2-fasrc01
source activate dannce

# Train com network
sbatch --wait holy_com_train.sh $1
wait

# Predict with com network in parallel and merge results
com-predict-multi-gpu $1
com-merge $1

