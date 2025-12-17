#!/bin/bash
#SBATCH --job-name=ego_ee_REQ_stp1
#SBATCH --account=a144
#SBATCH --output=slurm-ego_ee_REQ_stp1-%j.out
#SBATCH --error=slurm-ego_ee_REQ_stp1-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --environment=/users/jiaqchen/.edf/faive2lerobot.toml
#SBATCH --requeue
#SBATCH --signal=USR1@600


# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"

# Check some specs
free -h
nvidia-smi --query-gpu=memory.total --format=csv

# Lightning will automatically requeue the job if it crashes, so following is unnecessary: On SIGUSR1, request requeue and exit gracefully
# trap 'echo "$(date -Ins) SIGUSR1 received; requeuing..."; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1

# Check if ckpt_path exists, if not then ckpt_path=null, otherwise use the path
export frame_type=ee_frame
export hydra_run_dir=/iopsstor/scratch/cscs/jiaqchen/egomim_out/requeue/cup_${frame_type}_stp1
export ckpt_path=${hydra_run_dir}/checkpoints/last.ckpt
ckpt_path=$( [[ -f "$ckpt_path" ]] && echo "$ckpt_path" || echo null )
echo "CHECKPOINT PATH! ckpt_path: $ckpt_path"

export dataset_root=/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/srl_data/output/release/stp_1/cup_lerobot_${frame_type}

# Use srun to launch all 4 processes simultaneously for DDP
# Similar to the example script, use bash -c to ensure venv is activated in each process
# SLURM allocates 4 tasks (--ntasks-per-node=4), and srun launches one process per task
# Each process will source clariden.sh to activate venv and set up environment variables
# Expose following variables to sbatch:
# - name
# - description
# - chosen_frame
# - ckpt_path
# - hydra.run.dir
# - data.train_datasets.dataset1.root
# - data.valid_datasets.dataset1.root
CMD="
source /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse/eth_clariden/clariden.sh
cd /iopsstor/scratch/cscs/jiaqchen/egomim_out
python /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse/egomimic/trainHydra.py \
    name=cup_${frame_type} \
    description=cup_${frame_type} \
    chosen_frame=${frame_type} \
    ckpt_path=$ckpt_path \
    hydra.run.dir=$hydra_run_dir \
    data.train_datasets.dataset1.root=$dataset_root \
    data.valid_datasets.dataset1.root=$dataset_root $@
"
srun bash -c "$CMD"

# Copy the results to capstor


# Print completion information
echo "Job finished at: $(date)"
