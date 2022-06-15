#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=48:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=92959
#SBATCH --output=92959.out

source ../venvs/hammer/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5
time python3 hammer-run.py  --envname mw --config configs/mw.yaml --nagents 3 --dru_toggle 0 --meslen 3 --randomseed 92959