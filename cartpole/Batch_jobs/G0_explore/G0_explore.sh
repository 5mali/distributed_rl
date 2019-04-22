#!/bin/bash

#SBATCH --job-name=G0_explore
#SBATCH --output=G0_explore.out
##SBATCH -p knm 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem 10GB



srun python ./G0_explore.py $@ >> ./G0_explore.data



