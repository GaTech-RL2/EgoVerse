#!/bin/bash
#SBATCH --job-name=scene_diversity_cotrain_1_30
#SBATCH --output=sbatch_logs/scene_diversity_cotrain_1_30.out
#SBATCH --error=sbatch_logs/scene_diversity_cotrain_1_30.err
#SBATCH --partition="overcap"
#SBATCH --account="rl2-lab"
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

python egomimic/trainHydra.py \
    --config-name=train.yaml \
    data=scene_diversity_cotrain/scene_diversity_cotrain_1_30 \
    logger.wandb.project=everse_scenes_diversity_fold_clothes_cotrain \
    name=fold-clothes-cotrain \
    description=scenes-1-time-30-cotrain
