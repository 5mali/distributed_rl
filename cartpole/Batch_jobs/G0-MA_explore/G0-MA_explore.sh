#!/bin/bash

#SBATCH --job-name=G0-MA_explore
#SBATCH --output=G0-MA_explore.out
##SBATCH -p knm 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem 10GB



srun python ./G0-MA_explore.py $@ >> ./G0-MA_explore.data



