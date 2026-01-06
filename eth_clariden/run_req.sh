#!/bin/bash
#SBATCH --job-name=fold_clothes
#SBATCH --account=a144
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/multi_node_slurm_out/50hz/fold_clothes/slurm-fold_clothes-%j-%t.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/multi_node_slurm_out/50hz/fold_clothes/slurm-fold_clothes-%j-%t.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --environment=/users/jiaqchen/.edf/faive2lerobot.toml
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@600

# Stop the script if a command fails or if an undefined variable is used
set -eo pipefail

# The sbatch script is executed by only one node.
echo "[sbatch-master] running on $(hostname)"

echo "[sbatch-master] SLURM_NODELIST: $SLURM_NODELIST"
echo "[sbatch-master] SLURM_NNODES: $SLURM_NNODES"
echo "[sbatch-master] SLURM_NODEID: $SLURM_NODEID"

echo "[sbatch-master] define some env vars that will be passed to the compute nodes"

# The defined environment vars will be shared with the other compute nodes.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)  
export MASTER_PORT=12345   # Choose an unused port
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))


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

export task=fold_clothes
export frame_type=base_frame
export domain=eve_bimanual
export hydra_run_dir=/iopsstor/scratch/cscs/jiaqchen/egomim_out/multi_node/50hz/${task}/${task}_${frame_type}
export ckpt_path=${hydra_run_dir}/checkpoints/last.ckpt
ckpt_path=$( [[ -f "$ckpt_path" ]] && echo "$ckpt_path" || echo null )
echo "CHECKPOINT PATH! ckpt_path: $ckpt_path"

export dataset_root=/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/srl_data/output/release_2_0/${task}_lerobot_${frame_type}

CMD="
source /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse/eth_clariden/clariden.sh
cd /iopsstor/scratch/cscs/jiaqchen/egomim_out
exec python /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse/egomimic/trainHydra.py \
    name=${task}_${frame_type} \
    description=${task}_${frame_type} \
    chosen_frame=${frame_type} \
    ckpt_path=$ckpt_path \
    hydra.run.dir=$hydra_run_dir \
    trainer.num_nodes=$SLURM_NNODES \
    data.train_datasets.dataset1.root=$dataset_root \
    data.valid_datasets.dataset1.root=$dataset_root \
    model.robomimic_model.domains=[$domain] $@
"
srun --signal=SIGUSR1@600 bash -c "$CMD"

echo "Job finished at: $(date)"
