#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=48:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=66962
#SBATCH --output=66962.out

source ../venvs/hammer/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5
time python3 hammer-run.py  --envname cn --config configs/cn.yaml --nagents 10 --dru_toggle 0 --meslen 0 --partialobs 0 --heterogeneity 0 --randomseed 66962