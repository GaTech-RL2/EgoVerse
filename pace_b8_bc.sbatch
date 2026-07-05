#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/pace_b8_bc_launch.sh $1 $2 $3 $4
