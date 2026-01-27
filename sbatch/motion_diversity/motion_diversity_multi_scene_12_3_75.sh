#!/bin/bash
#SBATCH --job-name=motion_diversity_multi_scene_12_3_75
#SBATCH --output=sbatch_logs/motion_diversity_multi_scene_12_3_75.out
#SBATCH --error=sbatch_logs/motion_diversity_multi_scene_12_3_75.err
#SBATCH --partition="hoffman-lab"
#SBATCH --account="hoffman-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="clippy"

source /coc/flash7/bli678/Shared/emimic/bin/activate

# Extract number of GPUs from SLURM_GPUS_PER_NODE (format: "l40s:4" -> 4)
NUM_GPUS_PER_NODE=$(echo ${SLURM_GPUS_PER_NODE} | cut -d: -f2)
export SLURM_GPUS=$((NUM_GPUS_PER_NODE * SLURM_NNODES))
echo "Using node: $SLURM_NODELIST, GPUs per node: $NUM_GPUS_PER_NODE, total GPUs: $SLURM_GPUS"

# Set PyTorch memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# python egomimic/trainHydra.py \
#     --config-name=train.yaml \
#     data=motion_diversity/motion_diversity_multi_scene_12_3_75 \
#     logger.wandb.project=everse_motion_diversity_multi_scene_fold_clothes \
#     name=fold-clothes \
#     description=operator-12-time-3_75

python egomimic/trainHydra.py \
    --config-name=train.yaml \
    data=motion_diversity/motion_diversity_multi_scene_12_3_75 \
    logger.wandb.project=everse_motion_diversity_multi_scene_fold_clothes \
    name=eval-fold-clothes-motion-diversity \
    description=operator-12-time-3_75 \
    train=false \
    validate=true \
    ckpt_path=/coc/cedarp-dxu345-0/bli678/EgoVerse/logs/fold_clothes/motion_diversity/operator-12-time-3_75_2026-01-21_12-13-57/everse_motion_diversity_multi_scene_fold_clothes/fold-clothes_operator-12-time-3_75_2026-01-21_12-13-57/checkpoints/epoch_1999.ckpt