#!/bin/bash

# This script will not be called directly from the console.
# It is called by other scripts only

#SBATCH --job-name=predCOM
# Job name
#SBATCH --mem=30000
# Job memory request
#SBATCH -t 3-00:00
# Time limit hrs:min:sec
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -p common,tdunn
#SBATCH --gres=gpu:1

# module load Anaconda3/5.1.0
. ~/.bashrc
source activate dannce_cuda11
# module load cuda/11.0.3-fasrc01
# module load cudnn/8.0.4.30_cuda11.0-fasrc01
com-predict "$@"
