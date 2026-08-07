#!/bin/bash
#SBATCH --account=hoffman-lab
#SBATCH --partition=hoffman-lab
#SBATCH --qos=long
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=12
#SBATCH --mem=378G
#SBATCH --gres=gpu:a40:3
#SBATCH --requeue
#SBATCH --exclude=cyborg,ig-88,hk47
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse-gmm/logs/launcher_logs/%x_%j.out
mkdir -p /coc/flash7/paphiwetsa3/projects/EgoVerse-gmm/logs/launcher_logs
bash /coc/flash7/paphiwetsa3/projects/EgoVerse-gmm/hnet_ed_launch.sh "$1"
