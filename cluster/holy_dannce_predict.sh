#!/bin/bash
#SBATCH --job-name=predictDannce
# Job name
#SBATCH --mem=30000
# Job memory request
#SBATCH -t 1-00:00
# Time limit hrs:min:sec
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -p olveczkygpu,gpu
#SBATCH --constraint=cc5.2
#SBATCH --gres=gpu:1
source ~/.bashrc
activate_dannce_cuda11
dannce-predict "$@"
