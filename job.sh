#!/bin/bash

#SBATCH -N 1                # nodes
#SBATCH --ntasks-per-node=4 # tasks per node out of 32
#SBATCH --gres=gpu:1        # gpus per node out of 4
#SBATCH --mem=60000        # memory per node out of 246000MB

#SBATCH -p m100_usr_prod
#SBATCH --time 0:20:00
#SBATCH -A IscrC_fastAAA
#SBATCH --job-name=sfGPU
#SBATCH --mail-type=ALL
#SBATCH --mail-user=scarpma@gmail.com

module load profile/deeplrn
module load autoload pytorch/1.7--cuda--10.2

export CUDA_VISIBLE_DEVICES=0
python prova_sf.py

wait # Waits for background jobs to finish ....
