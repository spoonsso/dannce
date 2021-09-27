#!/bin/bash
# Script to run all steps of dannce in a single job. 
#
# Inputs: com_config - path to com config.
#         dannce_config - path to com config.
# Example: sbatch com_and_dannce.sh /path/to/com_config.yaml /path/to/dannce_config.yaml
#SBATCH --job-name=com_and_dannce
#SBATCH --mem=5000
#SBATCH -t 5-00:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p common

# module load Anaconda3/5.1.0
. ~/.bashrc
source activate dannce_cuda11

# Commented out to make script DCC compatible
# module load cuda/11.0.3-fasrc01
# module load cudnn/8.0.4.30_cuda11.0-fasrc01

set -e
sbatch --wait holy_com_train.sh $1
wait
sbatch --wait holy_com_predict.sh $1
wait
sbatch --wait holy_dannce_train.sh $2
wait
sbatch --wait holy_dannce_predict.sh $2
