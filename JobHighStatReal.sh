#!/bin/bash

#SBATCH -N 1                # nodes
#SBATCH --ntasks-per-node=1 # tasks per node out of 32
#SBATCH --gres=gpu:1        # gpus per node out of 4
#SBATCH --mem=60000        # memory per node out of 246000MB

#SBATCH -p m100_usr_prod
#SBATCH --time 0:05:00
#SBATCH -A IscrC_fastAAA
#SBATCH --job-name=sfGPU
#SBATCH --mail-type=ALL
#SBATCH --mail-user=scarpma@gmail.com

module load profile/deeplrn
#module load autoload pytorch/1.7--cuda--10.2
module load wmlce/1.6.2

export CUDA_VISIBLE_DEVICES=0
python compute_high_stat_real.py \
-npy_path ../velocities.npy \
-out_dir HighStat/real/

wait # Waits for background jobs to finish ....
