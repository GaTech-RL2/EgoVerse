#!/bin/bash
#SBATCH --job-name=eval_mecka_eva
#SBATCH --partition=gpu-h100
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-gpu=180G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mecka_eva_cotrain_wam/eval_%x_%j.out
#SBATCH --error=logs/mecka_eva_cotrain_wam/eval_%x_%j.err

set -euo pipefail
REPO_ROOT="/storage/project/r-dxu345-0/acheluva3/EgoVerse"
cd "${REPO_ROOT}"
mkdir -p logs/mecka_eva_cotrain_wam
export PATH="${REPO_ROOT}/emimic/bin:${PATH}"
PY="${REPO_ROOT}/emimic/bin/python"

# --- Knobs -----------------------------------------------------------------
CKPT_PATH="${CKPT_PATH:-/storage/home/hcoda1/5/acheluva3/r-dxu345-0/EgoVerse/logs/mecka_eva_cotrain_wam/mecka_eva_cotrain_wan22_5b_bs1x2_4h200_gradaccum4_effbs32_0804_2026-08-04_23-24-18/checkpoints/epoch_799.ckpt}"
NUM_VAL_EPISODES="${NUM_VAL_EPISODES:-5}"
MODE="${MODE:-tf}"
DESCRIPTION="${DESCRIPTION:-mecka_eva_cotrain_eval_${MODE}_longeva}"

# Human (mecka) read window stays at training values; eva read window ×3
# → 3× longer predicted video for eva (its episodes are shorter per-clip).
# Only cam_horizon extends — action_horizon and state_horizon are locked to
# training-time values because the model's ROPE table is baked at that seq_len.
HUMAN_CAM_HORIZON=33
HUMAN_ACTION_HORIZON=192
HUMAN_STATE_HORIZON=8
EVA_CAM_HORIZON=99
EVA_ACTION_HORIZON=192
EVA_STATE_HORIZON=8

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((RANDOM + 20000))}"
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

"${PY}" -m egomimic.eval.eval_dreamzero \
    --config-name=train_zarr_human_wam_wan22_5b \
    +num_val_episodes=${NUM_VAL_EPISODES} \
    ckpt_path="${CKPT_PATH}" \
    data=mecka_eva_cotrain_wam \
    evaluator=eval_dreamzero_${MODE} \
    evaluator/viz@evaluator.viz_func=wam_cartesian_aria_eva \
    name=mecka_eva_cotrain_wam \
    description="${DESCRIPTION}" \
    launch_params.gpus_per_node=1 \
    launch_params.nodes=1 \
    data.train_dataloader_params.human_bimanual.batch_size=1 \
    data.valid_dataloader_params.human_bimanual.batch_size=1 \
    data.train_dataloader_params.eva_bimanual.batch_size=1 \
    data.valid_dataloader_params.eva_bimanual.batch_size=1 \
    data.train_datasets.human_bimanual.resolver.key_map.cam_horizon=${HUMAN_CAM_HORIZON} \
    data.train_datasets.human_bimanual.resolver.key_map.action_horizon=${HUMAN_ACTION_HORIZON} \
    data.train_datasets.human_bimanual.resolver.key_map.state_horizon=${HUMAN_STATE_HORIZON} \
    +data.train_datasets.human_bimanual.resolver.key_map.video_stride=6 \
    +data.train_datasets.human_bimanual.resolver.transform_list.video_stride=6 \
    data.train_datasets.eva_bimanual.resolver.key_map.cam_horizon=${EVA_CAM_HORIZON} \
    data.train_datasets.eva_bimanual.resolver.key_map.action_horizon=${EVA_ACTION_HORIZON} \
    data.train_datasets.eva_bimanual.resolver.key_map.state_horizon=${EVA_STATE_HORIZON} \
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
