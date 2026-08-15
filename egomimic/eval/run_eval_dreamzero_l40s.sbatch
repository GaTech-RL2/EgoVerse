#!/bin/bash
#SBATCH --job-name=eval_dreamzero
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem-per-gpu=180G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_dreamzero/%x_%j.out
#SBATCH --error=logs/eval_dreamzero/%x_%j.err

set -euo pipefail

REPO_ROOT="/storage/project/r-dxu345-0/acheluva3/EgoVerse"
cd "${REPO_ROOT}"

mkdir -p logs/eval_dreamzero

# Use the same virtualenv the training run used.
PY="${REPO_ROOT}/emimic/bin/python"

# --- User-tunable knobs ----------------------------------------------------
CKPT_PATH="${CKPT_PATH:-/storage/home/hcoda1/5/acheluva3/r-dxu345-0/EgoVerse/logs/aria_wam22_foldclothes/aria_wan22_5b_unfrozen_bs1_4gpu_flatlr_0713_2026-07-20_11-36-52/0/checkpoints/last.ckpt}"
NUM_VAL_EPISODES="${NUM_VAL_EPISODES:-3}"
WINDOWS_PER_EPISODE="${WINDOWS_PER_EPISODE:-4}"
DESCRIPTION="${DESCRIPTION:-aria_wan22_5b_dreamzero_eval}"
NAME="${NAME:-aria_wam22_foldclothes}"
NORM_CACHE="${NORM_CACHE:-/storage/project/r-dxu345-0/acheluva3/EgoVerse/logs/norm_stats_cache/aria_5fps_wamalign_norm_stats.json}"

# DDP needs these when a single process runs standalone (no torchrun).
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((RANDOM + 20000))}"
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# --- Overrides mirror the training run (see multirun.yaml).
# The eval script forces mode=eval, single GPU, and rewrites train/valid ---
# ``mode`` to run the OOD split via ``split_dataset_names``.
"${PY}" -m egomimic.eval.eval_dreamzero \
    --config-name=train_zarr_human_wam_wan22_5b \
    +num_val_episodes=${NUM_VAL_EPISODES} \
    +windows_per_episode=${WINDOWS_PER_EPISODE} \
    ckpt_path="${CKPT_PATH}" \
    data=aria_wam \
    name="${NAME}" \
    description="${DESCRIPTION}" \
    launch_params.gpus_per_node=1 \
    launch_params.nodes=1 \
    data.train_dataloader_params.human_bimanual.batch_size=1 \
    data.valid_dataloader_params.human_bimanual.batch_size=1 \
    data.train_datasets.human_bimanual.resolver.key_map.cam_horizon=33 \
    data.train_datasets.human_bimanual.resolver.key_map.action_horizon=192 \
    data.train_datasets.human_bimanual.resolver.key_map.state_horizon=8 \
    +data.train_datasets.human_bimanual.resolver.key_map.video_stride=6 \
    +data.train_datasets.human_bimanual.resolver.transform_list.video_stride=6 \
    model.robomimic_model.num_video_frames=9 \
    model.robomimic_model.action_horizon=192 \
    model.robomimic_model.dit.num_action_per_block=48 \
    +model.robomimic_model.dit.num_frame_per_block=2 \
    model.robomimic_model.dit.num_state_per_block=2 \
    model.robomimic_model.dit.lora=False \
    model.robomimic_model.dit.freeze_video=False \
    ~model.scheduler \
    norm_stats.precomputed_norm_path="${NORM_CACHE}"
