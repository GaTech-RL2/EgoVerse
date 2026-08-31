#!/bin/bash
#SBATCH --job-name=eval_dreamzero
#SBATCH --partition=gpu-h100
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
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
# Prepend the venv's ``bin`` to PATH so subprocess.run("s5cmd", ...) resolves
# — the WAM S3 resolver uses s5cmd (installed as a venv-bin binary) to sync
# episodes into ``folder_path`` when they aren't cached locally yet.
export PATH="${REPO_ROOT}/emimic/bin:${PATH}"

# --- User-tunable knobs ----------------------------------------------------
CKPT_PATH="${CKPT_PATH:-/storage/home/hcoda1/5/acheluva3/r-dxu345-0/EgoVerse/logs/aria_wam22_foldclothes/aria_wan22_5b_unfrozen_bs1_4gpu_flatlr_0713_2026-07-20_11-36-52/0/checkpoints/last.ckpt}"
NUM_VAL_EPISODES="${NUM_VAL_EPISODES:-3}"
DESCRIPTION="${DESCRIPTION:-aria_wan22_5b_dreamzero_eval}"
NAME="${NAME:-aria_wam22_foldclothes}"
# Which dataset-dict key the data yaml uses (human_bimanual for aria/mecka
# WAM, eva_bimanual for the cross-embodiment eva run). Used to build the
# dataloader and keymap override paths below.
EMB_KEY="${EMB_KEY:-human_bimanual}"
# Hydra data-group selection (yaml filename under hydra_configs/data/).
DATA_GROUP="${DATA_GROUP:-aria_wam}"
NORM_CACHE="${NORM_CACHE:-/storage/project/r-dxu345-0/acheluva3/EgoVerse/logs/norm_stats_cache/aria_5fps_wamalign_norm_stats.json}"
# Optional filter/folder overrides so the same sbatch can point at other
# lab/task/embodiment slices (e.g. lab=mecka, or eva_bimanual). Space-separated
# hydra overrides; empty by default (uses the aria_wam filter unchanged).
EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES:-}"

# Assemble the norm-cache override: pass ``norm_stats.precomputed_norm_path=null``
# when NORM_CACHE is unset / literal ``null`` so the norm-stats step recomputes
# fresh (needed for cross-embodiment eval where the cached JSON is keyed by
# the wrong embodiment id).
if [[ -z "${NORM_CACHE}" || "${NORM_CACHE}" == "null" ]]; then
    NORM_CACHE_OVERRIDE='norm_stats.precomputed_norm_path=null'
else
    NORM_CACHE_OVERRIDE="norm_stats.precomputed_norm_path=${NORM_CACHE}"
fi

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
    ckpt_path="${CKPT_PATH}" \
    data=${DATA_GROUP} \
    name="${NAME}" \
    description="${DESCRIPTION}" \
    launch_params.gpus_per_node=1 \
    launch_params.nodes=1 \
    data.train_dataloader_params.${EMB_KEY}.batch_size=1 \
    data.valid_dataloader_params.${EMB_KEY}.batch_size=1 \
    data.train_datasets.${EMB_KEY}.resolver.key_map.cam_horizon=33 \
    data.train_datasets.${EMB_KEY}.resolver.key_map.action_horizon=192 \
    data.train_datasets.${EMB_KEY}.resolver.key_map.state_horizon=8 \
    +data.train_datasets.${EMB_KEY}.resolver.key_map.video_stride=6 \
    +data.train_datasets.${EMB_KEY}.resolver.transform_list.video_stride=6 \
    model.robomimic_model.num_video_frames=9 \
    model.robomimic_model.action_horizon=192 \
    model.robomimic_model.dit.num_action_per_block=48 \
    +model.robomimic_model.dit.num_frame_per_block=2 \
    model.robomimic_model.dit.num_state_per_block=2 \
    model.robomimic_model.dit.lora=False \
    model.robomimic_model.dit.freeze_video=False \
    ~model.scheduler \
    "${NORM_CACHE_OVERRIDE}" \
    ${EXTRA_HYDRA_OVERRIDES}
