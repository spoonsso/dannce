#!/bin/bash
#SBATCH --job-name=predictDannce
# Job name
#SBATCH --mem=30000
# Job memory request
#SBATCH -t 0-03:00
# Time limit hrs:min:sec
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --constraint=cc5.2
#SBATCH -p olveczkygpu,gpu,cox,gpu_requeue
#SBATCH --gres=gpu:1
source ~/.bashrc
activate_dannce_cuda11
dannce-predict-single-batch "$@"
