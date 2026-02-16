#!/bin/bash
#SBATCH --job-name=robot_bc
#SBATCH --output=sbatch_logs/robot_bc_2.out
#SBATCH --error=sbatch_logs/robot_bc_2.err
#SBATCH --partition="rl2-lab"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node="l40s:1"
#SBATCH --qos="short"
#SBATCH --exclude="bishop"

source /coc/flash7/bli678/Projects/EgoVerse/emimic/bin/activate

# Extract number of GPUs from SLURM_GPUS_PER_NODE (format: "l40s:4" -> 4)
NUM_GPUS_PER_NODE=$(echo ${SLURM_GPUS_PER_NODE} | cut -d: -f2)
export SLURM_GPUS=$((NUM_GPUS_PER_NODE * SLURM_NNODES))
echo "Using node: $SLURM_NODELIST, GPUs per node: $NUM_GPUS_PER_NODE, total GPUs: $SLURM_GPUS"

# Set PyTorch memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python egomimic/trainHydra.py \
    --config-name=train.yaml \
    data=eva_bc_s3 \
    model=hpt_bc_flow_eva \
    logger.wandb.project=egowm_cup_saucer \
    name=cup_saucer \
    description=robot_hpt \
    ckpt_path="/coc/flash7/bli678/Projects/EgoVerse/logs/cup_saucer/robot_hpt_2026-02-11_00-58-14/checkpoints/last.ckpt"

