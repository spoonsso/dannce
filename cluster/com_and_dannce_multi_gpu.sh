#!/bin/bash
# Script to run all steps of dannce in a single job using multi-gpu prediction. 
#
# Inputs: com_config - path to com config.
#         dannce_config - path to com config.
# Example: sbatch com_and_dannce_multi_gpu.sh /path/to/com_config.yaml /path/to/dannce_config.yaml
#SBATCH --job-name=com_and_dannce
#SBATCH --mem=10000
#SBATCH -t 5-00:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p common
set -e

# Setup the dannce environment
module load Anaconda3/5.1.0
source activate dannce

# Predict with com network in parallel and merge results
com-predict-multi-gpu $1
com-merge $1

# Predict with dannce network in parallel and merge results
dannce-predict-multi-gpu $2
dannce-merge $2
