#!/bin/bash
#SBATCH --job-name=scene_diversity_cotrain_1_7_5
#SBATCH --output=sbatch_logs/scene_diversity_cotrain_1_7_5.out
#SBATCH --error=sbatch_logs/scene_diversity_cotrain_1_7_5.err
#SBATCH --partition="hoffman-lab"
#SBATCH --account="hoffman-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
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

python egomimic/trainHydra.py \
    --config-name=train.yaml \
    data=scene_diversity_cotrain/scene_diversity_cotrain_1_7_5 \
    logger.wandb.project=everse_scenes_diversity_fold_clothes_cotrain \
    name=fold-clothes-cotrain-2 \
    trainer.limit_val_batches=30 \
    model=hpt_cotrain_flow_shared_head \
    description=scenes-1-time-7_5-cotrain \
    ckpt_path=/coc/cedarp-dxu345-0/bli678/EgoVerse/logs/fold-clothes-cotrain/scenes-1-time-7_5-cotrain_2026-01-24_03-11-23/checkpoints/last.ckpt