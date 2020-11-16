#!/bin/bash
# Script to run all steps of dannce in a single job using multi-gpu prediction. 
#
# Inputs: dannce_config - path to com config.
# Example: sbatch dannce_multi_gpu.sh /path/to/dannce_config.yaml
#SBATCH --job-name=dannce_multi_gpu
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

# Train dannce network
sbatch --wait holy_dannce_train.sh $1
wait

# Predict with dannce network in parallel and merge results
dannce-predict-multi-gpu $1
dannce-merge $1
