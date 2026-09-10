#!/bin/bash
#SBATCH --job-name=hpt_kp_freeform_opsplit
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=long
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/coc/flash7/rco3/EgoVerse2/logs/hpt_kp_fold_freeform_opsplit_%j.log

# SKYNET full run: ~300M HPT flow BC (+ frozen Qwen3-Embedding text stem) on
# the 138-D wrist-first MANO KEYPOINT action in wrist frame
# (keypoints_wristframe_ypr: per hand [wrist xyz+ypr (6) | 21 keypoints (63)]),
# mecka FREEFORM fold-clothes, held-out-OPERATOR split
# (data=mecka_fold_freeform_opsplit_hpt_keypoints):
#   train              147 operators, 412 eps / 1.01M frames
#   Valid/*            31 held-out operators, 50 eps / 116k frames (10.2 %)
#   train_viz/*        on-train spot check (same operators as train, unshuffled slice)
# Constant lr 5e-5 (model config), bs 64 x 1 A40. Cadence (epoch =
# limit_train_batches = 100 steps): MAX_EPOCHS=1000 -> 100k steps; val every
# 20 epochs (2k steps), viz every 100 (both heads, <=10 batches), ckpt every
# 50 epochs keeping the 3 newest (monitor=step) + last (~6 GB each). Resume
# with CKPT_PATH=.../last.ckpt and a larger MAX_EPOCHS. Eval =
# eval_hpt_keypoints_wrist (wrist -> head-frame keypoint revert; 126-D
# cam-frame keypoint MSE + skeleton overlay video) + the train_viz head.
# Metrics: Valid/human_bimanual_actions_keypoints_{paired,final,cam_paired,cam_final}_mse_avg.
#
# Requires the CPU norm job's output (norm_fold_freeform_opsplit_hpt_kp_skynet.sbatch);
# submit with --dependency=afterok:<norm job id>. The held-out-operator
# episodes are synced from S3 at startup if missing.
#
# The parent data config hard-codes the PACE mirror as folder_path, so the
# resolver override below points it at the Skynet mirror.
#
# The Qwen stem loads Qwen/Qwen3-Embedding-0.6B from the HF cache
# (/coc/flash7/rco3/.cache/huggingface, already populated); compute nodes run
# with HF_HUB_OFFLINE=1.
#
# W&B: online if WANDB_API_KEY is in ~/.egoverse_env, else offline.
#
# Env knobs:
#   DATA_DIR   - zarr mirror (default: /coc/flash7/scratch/egoverseS3ZarrDataset = paths.dataset_dir)
#   NORM_ROOT  - where norm-stats dirs live (default: $REPO/norm_stats)
#   NORM_PATH  - precomputed norm_stats dir (default: the norm job's output)
#   CKPT_PATH  - resume from a Lightning checkpoint
#   LR         - optimizer lr (default 5e-5)
#   BS         - per-GPU batch size (default 64)
#   VAL_EVERY  - epochs between vals (default 20)
#   MAX_EPOCHS - trainer.max_epochs (default 1000 = 100k steps)

set -euo pipefail
case "$(hostname)" in sky1*|sky2*) echo "ABORT: on login node"; exit 1;; esac
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

REPO=/coc/flash7/rco3/EgoVerse2
DATA_DIR="${DATA_DIR-/coc/flash7/scratch/egoverseS3ZarrDataset}"
NORM_ROOT="${NORM_ROOT-$REPO/norm_stats}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY="$REPO/emimic/bin/python"
export PATH="$REPO/emimic/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if grep -q "^WANDB_API_KEY=" ~/.egoverse_env 2>/dev/null; then
  export WANDB_API_KEY="$(grep -oP '^WANDB_API_KEY=\K.*' ~/.egoverse_env | tr -d "\"'")"
  export WANDB_MODE=online
else
  export WANDB_MODE=offline
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Private compile caches per job (same-user jobs on one node otherwise share
# /tmp/torchinductor_<user>).
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="/tmp/triton_${USER}_${SLURM_JOB_ID}"

NORM_PATH="${NORM_PATH-$NORM_ROOT/fold_freeform_opsplit_hpt_kp/norm_stats}"
CKPT_PATH="${CKPT_PATH-}"
LR="${LR-5e-5}"
BS="${BS-64}"
VAL_EVERY="${VAL_EVERY-20}"
MAX_EPOCHS="${MAX_EPOCHS-1000}"
LABEL=hpt300m_keypoints_fold_clothes_freeform_opsplit

if [[ ! -f "$NORM_PATH/norm_stats.json" ]]; then
  echo "ERROR: precomputed norm stats not found: $NORM_PATH/norm_stats.json" >&2
  echo "Run egomimic/scripts/norm_fold_freeform_opsplit_hpt_kp_skynet.sbatch first." >&2
  exit 1
fi

ARGS=(
  --config-name=train_zarr_mecka_kp_wrist_hpt
  "paths.dataset_dir=$DATA_DIR"
  "data.train_datasets.human_bimanual.resolver.folder_path=$DATA_DIR"
  launch_params.gpus_per_node=1
  "model.optimizer.lr=$LR"
  "data.train_dataloader_params.human_bimanual.batch_size=$BS"
  "data.valid_dataloader_params.human_bimanual.batch_size=$BS"
  "data.train_viz_dataloader_params.human_bimanual.batch_size=$BS"
  data.train_dataloader_params.human_bimanual.num_workers=10
  data.valid_dataloader_params.human_bimanual.num_workers=6
  data.train_viz_dataloader_params.human_bimanual.num_workers=4
  "norm_stats.precomputed_norm_path=$NORM_PATH"
  "trainer.check_val_every_n_epoch=$VAL_EVERY"
  "trainer.max_epochs=$MAX_EPOCHS"
  callbacks.model_checkpoint.every_n_epochs=50
  # keep the 3 newest 50-epoch checkpoints + last (Lightning needs a monitor
  # for save_top_k > 1; `step` is always a monitor candidate)
  +callbacks.model_checkpoint.monitor=step
  +callbacks.model_checkpoint.mode=max
  callbacks.model_checkpoint.save_top_k=3
  "name=$LABEL"
  "description=bs${BS}x1_const${LR}_kp_opsplit"
  "+logger.wandb.name=$LABEL"
)
# Lightning names checkpoints "epoch_epoch=..." — quote so hydra's grammar
# doesn't choke on '='.
[[ -n "$CKPT_PATH" ]] && ARGS+=("ckpt_path='$CKPT_PATH'")

echo "[train] cwd=$(pwd) commit=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) data=$DATA_DIR wandb=$WANDB_MODE"
echo "[train] srun python -u egomimic/trainHydra.py ${ARGS[*]}"
srun "$PY" -u egomimic/trainHydra.py "${ARGS[@]}"
echo "[train] exit=$?"
