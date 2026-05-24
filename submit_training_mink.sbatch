#!/bin/bash
#SBATCH -J egoverse_RBY1_mink
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=32G
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH -o ./logs/slurm/%A/task_%a/out.log
#SBATCH -e ./logs/slurm/%A/task_%a/err.log
#SBATCH --exclude='puma,deebot,qt-1'

cd /coc/flash7/zhenyang/EgoVerse
source emimic/bin/activate
export TMPDIR=/tmp

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

DATASET=RBY1_0423_cart_mobile_mink_human_data
CONFIG=train_rby1_sew_human_mobie_img_act32_proprio_dropout
CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
        --config-name=${CONFIG} \
        name=${DATASET} \
        data.train_datasets.dataset1.datasets.rl2_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET} \
        data.valid_datasets.dataset1.datasets.eth_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET}
