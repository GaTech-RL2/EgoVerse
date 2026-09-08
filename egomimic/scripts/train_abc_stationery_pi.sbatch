#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=72:00:00
#SBATCH --job-name=pi05_abc_stat
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/pi05_abc_stationery_ft_%j.log
# Throttled H200s (2026-09-01 probe: SM clocks pinned ~350-420 MHz, 93 C —
# DDP runs at the slowest rank, so one bad unit is a 3-4x slowdown).
#SBATCH --exclude=atl1-1-02-012-16-0,atl1-1-02-014-9-0

# pi0.5 on ALL 2,066 ABC "sort the stationery into containers" episodes
# (41.6 h) — a zero-shot rollout candidate for the YAM station.
#
# Recipe = the abc-eva finetune (train_abc_eva_ft_foldviz.sbatch) with the task
# restricted and the language prompt carrying both granularities:
#   data=abc_stationery_pi_6d   wristframe-6D actions, rl2yam extrinsics,
#                               ABC's dense sub-task annotation spans (stock
#                               annotation pipeline, no injected task prompt)
#   model=pi0.5_bc_abc_eva_6d   eva_bimanual domain (the mirror zarrs still say
#                               eva_bimanual), 20D native, no embodiment label
#   evaluator=eval_pi_wristframe_6d + the train_viz second head
#   init  = the DEFAULT pi0.5 base checkpoint (pi05_base_pytorch) and nothing
#           else — BC from scratch, no mecka/human pretrain in the lineage
#   norm  = norm_stats/abc_stationery_rl2yam (this task, this extrinsics key —
#           the proprio frame changed, so the abc_fold_viz stats do not apply)
#
# 400 epochs x 100 steps at bs 64 x 4 GPUs (global 256) ~= 10.2M samples
# (~2.3 passes over the 4.49M frames). A checkpoint every 50 epochs is KEPT
# (save_top_k=-1) so the rollout has 8 candidates rather than the default
# rolling top-3 — see the checkpoint-rotation trap in the vault notes.
#
# Env knobs:
#   NORM_PATH  - precomputed norm_stats dir
#   CKPT_PATH  - resume THIS run from its own Lightning checkpoint
#   LR         - optimizer lr (default 5e-5)
#   MAX_EPOCHS - default 400

set -euo pipefail
cd /storage/project/r-dxu345-0/agao81/EgoVerse/.claude/worktrees/abc-stationery-pi
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if grep -q api.wandb.ai ~/.netrc 2>/dev/null; then
  export WANDB_MODE=online
else
  export WANDB_MODE=offline
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_$SLURM_JOB_ID TRITON_CACHE_DIR=/tmp/tr_$SLURM_JOB_ID

NORM_PATH="${NORM_PATH-/storage/project/r-dxu345-0/agao81/norm_stats/abc_stationery_rl2yam/norm_stats}"
CKPT_PATH="${CKPT_PATH-}"
LR="${LR-5e-5}"
MAX_EPOCHS="${MAX_EPOCHS-400}"

if [[ ! -f "$NORM_PATH/norm_stats.json" ]]; then
  echo "ERROR: precomputed norm stats not found: $NORM_PATH/norm_stats.json" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== commit=$(git rev-parse --short HEAD) node=$SLURMD_NODENAME $(date) ==="

ARGS=(
  --config-name=train_zarr_cartesian_pi
  data=abc_stationery_pi_6d
  model=pi0.5_bc_abc_eva_6d
  evaluator=eval_pi_wristframe_6d
  +evaluator@train_viz_evaluator=train_viz_pi_wristframe_6d
  launch_params.gpus_per_node=4
  "trainer.max_epochs=$MAX_EPOCHS"
  trainer.min_epochs=0
  "model.optimizer.lr=$LR"
  "norm_stats.precomputed_norm_path=$NORM_PATH"
  model.robomimic_model.config.pytorch_weight_path=/storage/project/r-dxu345-0/agao81/EgoVerse/egomimic/algo/pi_checkpoints/pi05_base_pytorch
  callbacks.model_checkpoint.every_n_epochs=50
  callbacks.model_checkpoint.save_top_k=-1
  name=pi05_abc_stationery_bc
  description=pi05_abc_stationery_full2066_bc_from_pi05base_bs64x4_wrist6d
)
# Resume THIS run only; there is no weights-only init — training starts from the
# pi0.5 base checkpoint loaded via pytorch_weight_path.
if [[ -n "$CKPT_PATH" ]]; then
  ARGS+=("ckpt_path='$CKPT_PATH'")
fi

echo "[train] srun python -u egomimic/trainHydra.py ${ARGS[*]}"
srun "$PY" -u egomimic/trainHydra.py "${ARGS[@]}"
echo "=== TRAIN EXIT $? ($(date)) ==="
