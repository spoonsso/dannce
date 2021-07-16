#!/bin/bash
#SBATCH --job-name=trainDannce
# Job name
#SBATCH --mem=80000
# Job memory request
#SBATCH -t 3-00:00
# Time limit hrs:min:sec
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -p olveczkygpu,gpu
#SBATCH --constraint=cc5.2
#SBATCH --gres=gpu:1
module load Anaconda3/5.0.1-fasrc02
source activate dannce_cuda11
dannce-train-single-batch "$@"
