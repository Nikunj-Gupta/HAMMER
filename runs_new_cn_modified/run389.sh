#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=10:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=94644
#SBATCH --output=94644.out

source ../venvs/hammer/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5
time python3 hammer-run.py  --envname cn --config configs/cn.yaml --nagents 7 --dru_toggle 0 --meslen 2 --partialobs 0 --heterogeneity 1 --randomseed 94644