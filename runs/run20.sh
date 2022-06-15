#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=10:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=90300
#SBATCH --output=90300.out

source ../venvs/hammer/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5
time python3 hammer-run.py  --envname mw --config configs/mw.yaml --nagents 3 --dru_toggle 1 --meslen 1 --randomseed 90300