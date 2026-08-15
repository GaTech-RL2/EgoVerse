#!/bin/bash
#SBATCH --job-name=val_sweep_mecka_eva
#SBATCH --partition=gpu-h100
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-gpu=180G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mecka_eva_cotrain_wam/val_sweep_%j.out
#SBATCH --error=logs/mecka_eva_cotrain_wam/val_sweep_%j.err

set -euo pipefail
REPO_ROOT="/storage/project/r-dxu345-0/acheluva3/EgoVerse"
cd "${REPO_ROOT}"
mkdir -p logs/mecka_eva_cotrain_wam
export PATH="${REPO_ROOT}/emimic/bin:${PATH}"
PY="${REPO_ROOT}/emimic/bin/python"

TRAIN_DIR="${TRAIN_DIR:-/storage/home/hcoda1/5/acheluva3/r-dxu345-0/EgoVerse/logs/mecka_eva_cotrain_wam/mecka_eva_cotrain_wan22_5b_bs1x2_4h200_gradaccum4_effbs32_0804_2026-08-04_23-24-18}"
CKPT_DIR="${TRAIN_DIR}/checkpoints"
NUM_VAL_EPISODES="${NUM_VAL_EPISODES:-20}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-val_sweep_mecka_eva_cotrain_$(basename ${TRAIN_DIR})}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((RANDOM + 20000))}"
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

"${PY}" -m egomimic.eval.val_sweep \
    --config-name=train_zarr_human_wam_wan22_5b \
    +checkpoints_dir="${CKPT_DIR}" \
    +num_val_episodes=${NUM_VAL_EPISODES} \
    +wandb_project=wam \
    +wandb_run_name="${WANDB_RUN_NAME}" \
    data=mecka_eva_cotrain_wam \
    evaluator=eval_dreamzero_tf \
    evaluator/viz@evaluator.viz_func=wam_cartesian_aria_eva \
    name=mecka_eva_cotrain_wam \
    description=val_sweep_${SLURM_JOB_ID} \
    launch_params.gpus_per_node=1 \
    launch_params.nodes=1 \
    data.train_dataloader_params.human_bimanual.batch_size=1 \
    data.valid_dataloader_params.human_bimanual.batch_size=1 \
    data.train_dataloader_params.eva_bimanual.batch_size=1 \
    data.valid_dataloader_params.eva_bimanual.batch_size=1 \
    data.train_datasets.human_bimanual.resolver.key_map.cam_horizon=33 \
    data.train_datasets.human_bimanual.resolver.key_map.action_horizon=192 \
    data.train_datasets.human_bimanual.resolver.key_map.state_horizon=8 \
    +data.train_datasets.human_bimanual.resolver.key_map.video_stride=6 \
    +data.train_datasets.human_bimanual.resolver.transform_list.video_stride=6 \
    data.train_datasets.eva_bimanual.resolver.key_map.cam_horizon=99 \
    data.train_datasets.eva_bimanual.resolver.key_map.action_horizon=192 \
    data.train_datasets.eva_bimanual.resolver.key_map.state_horizon=8 \
    +data.train_datasets.eva_bimanual.resolver.key_map.video_stride=6 \
    +data.train_datasets.eva_bimanual.resolver.transform_list.video_stride=6 \
    model.robomimic_model.num_video_frames=9 \
    model.robomimic_model.action_horizon=192 \
    model.robomimic_model.dit.num_action_per_block=48 \
    +model.robomimic_model.dit.num_frame_per_block=2 \
    model.robomimic_model.dit.num_state_per_block=2 \
    model.robomimic_model.dit.lora=False \
    model.robomimic_model.dit.freeze_video=False \
    ~model.scheduler \
    'model.robomimic_model.domains=[human_bimanual,eva_bimanual]' \
    +model.robomimic_model.ac_keys.eva_bimanual=actions_cartesian \
    norm_stats.precomputed_norm_path=/storage/project/r-dxu345-0/acheluva3/EgoVerse/logs/norm_stats_cache/mecka_eva_wam_norm_stats_perdim.json
