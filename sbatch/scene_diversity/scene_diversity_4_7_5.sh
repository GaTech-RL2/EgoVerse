#!/bin/bash
#SBATCH --job-name=scene_diversity_4_7_5
#SBATCH --output=sbatch_logs/scene_diversity_4_7_5.out
#SBATCH --error=sbatch_logs/scene_diversity_4_7_5.err
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

python egomimic/trainHydra.py \
    --config-name=train.yaml \
    data=scene_diversity/scene_diversity_4_7_5 \
    logger.wandb.project=everse_scenes_diveristy_fold_clothes \
    name=fold-clothes \
    description=scenes-4-time-7_5