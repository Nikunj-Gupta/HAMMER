#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=10:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=34046
#SBATCH --output=34046.out

source ../venvs/hammer/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5
time python3 hammer-run.py  --envname cn --config configs/cn.yaml --nagents 5 --dru_toggle 0 --meslen 1 --partialobs 0 --heterogeneity 0 --randomseed 34046