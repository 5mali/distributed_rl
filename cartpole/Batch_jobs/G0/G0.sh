#!/bin/bash

#SBATCH --job-name=G0
#SBATCH --output=G0.out
##SBATCH -p knm 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem 10GB



srun python ./G0.py $@ >> ./G0.data



