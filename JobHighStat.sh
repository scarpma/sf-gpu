#!/bin/bash

#SBATCH -N 1                # nodes
#SBATCH --ntasks-per-node=8 # tasks per node out of 32
#SBATCH --gres=gpu:1        # gpus per node out of 4
#SBATCH --mem=60000        # memory per node out of 246000MB

#SBATCH -p m100_usr_prod
#SBATCH --time 4:00:00
#SBATCH -A IscrC_fastAAA
#SBATCH --job-name=sfGPU
#SBATCH --mail-type=ALL
#SBATCH --mail-user=scarpma@gmail.com

module load profile/deeplrn
#module load autoload pytorch/1.7--cuda--10.2
module load wmlce/1.6.2

export CUDA_VISIBLE_DEVICES=0
python compute_high_stat.py \
-model_path ../wgangp3-short-filters/runs/TRAIN1/4/3000_gen.h5 \
-out_dir HighStat/wgangp3-tracers-short_filters/ -iters 100 \
>& log0.txt &

export CUDA_VISIBLE_DEVICES=1
python compute_high_stat.py \
-model_path ../wgangp3-short-filters/runs/TRAIN2/4/3000_gen.h5 \
-out_dir HighStat/wgangp3-tracers-short_filters/ -iters 100 \
>& log1.txt &

export CUDA_VISIBLE_DEVICES=2
python compute_high_stat.py \
-model_path ../wgangp3-short-filters/runs/TRAIN3/4/3000_gen.h5 \
-out_dir HighStat/wgangp3-tracers-short_filters/ -iters 100 \
>& log2.txt &

export CUDA_VISIBLE_DEVICES=3
python compute_high_stat.py \
-model_path ../wgangp3-short-filters/runs/TRAIN4/4/3000_gen.h5 \
-out_dir HighStat/wgangp3-tracers-short_filters/ -iters 100 \
>& log3.txt &

wait # Waits for background jobs to finish ....


