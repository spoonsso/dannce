#!/bin/bash
#SBATCH --job-name=trainDannce
# Job name
#SBATCH --mem=60000
# Job memory request
#SBATCH -t 2-00:00
# Time limit hrs:min:sec
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -p olveczkygpu,gpu
#SBATCH --gres=gpu:1
module load Anaconda3/5.0.1-fasrc02
source activate dannce
dannce-train "$@"
