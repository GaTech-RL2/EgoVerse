#!/bin/bash
#SBATCH --job-name=T_cup_BC
#SBATCH --account=a144
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/multi_node_slurm_out_v2/50hz/BC/cup/slurm-cup-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/multi_node_slurm_out_v2/50hz/BC/cup/slurm-cup-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=10:00:00
#SBATCH --partition=normal
#SBATCH --environment=/users/jiaqchen/.edf/faive2lerobot.toml
#SBATCH --requeue
#SBATCH --signal=USR1@600

# Parse command-line arguments
export debug=false
export new_wandb=false
export skip_preflight=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug) export debug=true; shift ;;
        --new-wandb) export new_wandb=true; shift ;;
        --skip-preflight) export skip_preflight=true; shift ;;
        *) shift ;;
    esac
done

##################### VARIANT CONFIG #####################
export VARIANT="BC"
export DATA_CONFIG="cup/multi_data_BC"
export CONFIG_SUFFIX="_BC"
export SBATCH_TIME="10:00:00"
export RLDB_WORKERS=16
export WANDB_VARIANT_TAG=""  # No tag for base BC
##########################################################

# Source common logic
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
