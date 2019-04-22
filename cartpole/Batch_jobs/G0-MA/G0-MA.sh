#!/bin/bash

#SBATCH --job-name=G0-MA
#SBATCH --output=G0-MA.out
##SBATCH -p knm 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem 10GB



srun python ./G0-MA.py $@ >> ./G0-MA.data



