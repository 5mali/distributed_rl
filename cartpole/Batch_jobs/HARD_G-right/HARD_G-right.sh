#!/bin/bash

#SBATCH --job-name=HARD_G-right
#SBATCH --output=HARD_G-right.out
#SBATCH -p knm 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=35
#SBATCH --mem 20GB

## srun parallel -j10  python ./HARD_G.py ::: 741 9086 4507 7551 5987 862 5955 5095 3119 3664 >> BASE_G.data 

srun parallel -j35 python ./HARD_G-right.py ::: 741 9086 4507 7551 5987 862 5955 5095 3119 3664 6654 7551 6118 7628 3615 9948 7957 6192 1259 2712 9740 6040 1896 9684 5584 3531 371 5731 7100 1145 9382 4068 3713 2758 5632 >> HARD_G-right.data 
