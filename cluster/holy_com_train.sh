#!/bin/bash
#SBATCH --job-name=trainCOM
# Job name
#SBATCH --mem=30000
# Job memory request
#SBATCH -t 3-00:00
# Time limit hrs:min:sec
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -p olveczkygpu,gpu
#SBATCH --constraint=cc5.2
#SBATCH --gres=gpu:1
module load Anaconda3/5.0.1-fasrc02
source activate dannce_cuda11
module load cuda/11.0.3-fasrc01
module load cudnn/8.0.4.30_cuda11.0-fasrc01
com-train "$@"
