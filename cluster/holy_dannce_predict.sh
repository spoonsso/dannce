#!/bin/bash
#SBATCH --job-name=predictDannce
# Job name
#SBATCH --mem=10000
# Job memory request
#SBATCH -t 1-00:00
# Time limit hrs:min:sec
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -p olveczkygpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=cc5.2
module load Anaconda3/5.0.1-fasrc02
source activate dannce
dannce-predict "$@"