#!/bin/bash
# Script to run all steps of dannce in a single job. 
#
# Inputs: dannce_config - path to com config.
# Example: sbatch dannce.sh /path/to/dannce_config.yaml
#SBATCH --job-name=dannce
#SBATCH --mem=5000
#SBATCH -t 5-00:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p olveczky
set -e
sbatch --wait holy_dannce_train.sh $1
wait
sbatch --wait holy_dannce_predict.sh $1
