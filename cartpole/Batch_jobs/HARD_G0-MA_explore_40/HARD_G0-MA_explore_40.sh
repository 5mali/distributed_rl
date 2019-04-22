#!/bin/bash

#SBATCH --job-name=HARD_G0-MA_explore_40
#SBATCH --output=HARD_G0-MA_explore_40.out
##SBATCH -p knm 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem 100GB



srun python ./HARD_G0-MA_explore_40.py $@ >> ./HARD_G0-MA_explore_40.data



