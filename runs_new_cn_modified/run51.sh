#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=10:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=2528
#SBATCH --output=2528.out

source ../venvs/hammer/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5
time python3 hammer-run.py  --envname cn --config configs/cn.yaml --nagents 3 --dru_toggle 1 --meslen 2 --partialobs 1 --heterogeneity 0 --randomseed 2528