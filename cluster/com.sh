#!/bin/bash
# Script to run all steps of dannce in a single job. 
#
# Inputs: com_config - path to com config.
# Example: sbatch com.sh /path/to/com_config.yaml
#SBATCH --job-name=com
#SBATCH --mem=5000
#SBATCH -t 5-00:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p olveczky
set -e
sbatch --wait holy_com_train.sh $1
wait
sbatch --wait holy_com_predict.sh $1
