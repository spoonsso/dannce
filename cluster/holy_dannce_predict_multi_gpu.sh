#!/bin/bash

# This script will not be called directly from the console.
# It is called by other scripts only

#SBATCH --job-name=predictDannce
# Job name
#SBATCH --mem=30000
# Job memory request
#SBATCH -t 0-03:00
# Time limit hrs:min:sec
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p scavenger-gpu,tdunn --account=tdunn 
#SBATCH --gres=gpu:1

# module load Anaconda3/5.1.0
. ~/.bashrc
# conda info - For debugging
conda activate dannce_cuda11
# conda info - For debugging
# module load cuda/11.0.3-fasrc01
# module load cudnn/8.0.4.30_cuda11.0-fasrc01
dannce-predict-single-batch "$@"
