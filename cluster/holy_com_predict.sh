#!/bin/bash
#SBATCH --job-name=predCOM
# Job name
#SBATCH --mem=30000
# Job memory request
#SBATCH -t 3-00:00
# Time limit hrs:min:sec
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -p olveczkygpu,gpu
#SBATCH --gres=gpu:1
module load Anaconda3/5.0.1-fasrc02
source activate dannce
com-predict "$@"
