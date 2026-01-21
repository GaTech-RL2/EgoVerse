#!/bin/bash
#SBATCH --job-name=debug_dump_begin
#SBATCH --account=a144
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/multi_node_slurm_out_v2/50hz/DEBUG_799/cup/slurm-cup-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/multi_node_slurm_out_v2/50hz/DEBUG_799/cup/slurm-cup-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --environment=/users/jiaqchen/.edf/faive2lerobot.toml
#SBATCH --requeue
#SBATCH --signal=USR1@600

# Stop the script if a command fails or if an undefined variable is used
set -eo pipefail

ulimit -c 0

echo $PG # POINT_GAP_ACT
echo $CL # CHUNK_LENGTH_ACT, need to change action horizon
echo $CL_OUT # CHUNK_LENGTH_ACT_OUT, need to change action horizon
if [ "$CL_OUT" = "None" ]; then
    export EXPERIMENT=pg${PG}_cl${CL}
else
    export EXPERIMENT=pg${PG}_cl${CL}_clout${CL_OUT}
fi
echo $EXPERIMENT

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
export NCCL_NET="AWS Libfabric"


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
# Trap and send to all children processes
# trap 'echo "USR1 trapped"; kill -USR1 -- -$$' USR1

##################### SET THESE VARIABLES #####################
export task=cup
export frame_type=base_frame
export arm=bimanual
export debug=false
###############################################################

##################### MAYBE CHANGE THIS PATH #####################
export hydra_run_dir=/iopsstor/scratch/cscs/jiaqchen/egomim_out/multi_node_v2/50hz/DEBUG_799/${EXPERIMENT}/${task}/${task}_${frame_type}
export dataset_root=/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/srl_data/output/release_2_0/50hz/${EXPERIMENT}/${task}_lerobot_${frame_type}
# export dataset_root=/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/srl_data/output/debug_2_0/${task}_lerobot_${frame_type}_1_debug
##################################################################

if [ "$debug" = true ]; then
    export trainer=debug
    export logger=debug
else
    export trainer=ddp
    export logger=wandb
fi

export config_name=train_eth_${arm}
# export ckpt_path=${hydra_run_dir}/checkpoints/last.ckpt
export ckpt_path='/iopsstor/scratch/cscs/jiaqchen/egomim_out/multi_node_v2/50hz/pg2_cl75_clout100/cup/cup_base_frame/checkpoints/epoch_epoch=799.ckpt'
ckpt_path=$( [[ -f "$ckpt_path" ]] && echo "\\\"$ckpt_path\\\"" || echo null )
echo "CHECKPOINT PATH! ckpt_path: $ckpt_path"

if [ "$CL_OUT" = "None" ]; then
    export CL_param=${CL}
else
    export CL_param=${CL_OUT}
fi

CMD="
source /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse/eth_clariden/clariden.sh
cd /iopsstor/scratch/cscs/jiaqchen/egomim_out
python /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse/egomimic/trainHydra.py \
    --config-name=${config_name} \
    trainer=${trainer} \
    logger=${logger} \
    name=${task}_${frame_type} \
    description=${task}_${frame_type} \
    chosen_frame=${frame_type} \
    ckpt_path=${ckpt_path} \
    hydra.run.dir=${hydra_run_dir} \
    trainer.num_nodes=${SLURM_NNODES} \
    data.train_datasets.dataset1.root=${dataset_root} \
    data.valid_datasets.dataset1.root=${dataset_root} \
    model.robomimic_model.head_specs.eve_${arm}.action_horizon=${CL_param} \
    model.robomimic_model.head_specs.eve_${arm}.model.act_seq=${CL_param} \
    model.robomimic_model.head_specs.eve_${arm}_actions_joints.action_horizon=${CL_param} \
    model.robomimic_model.head_specs.eve_${arm}_actions_joints.model.act_seq=${CL_param} \
    $@
"
srun bash -lc "$CMD"

echo "Job finished at: $(date)"
