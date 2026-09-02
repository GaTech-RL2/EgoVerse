#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -p gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:30:00
#SBATCH --exclude=atl1-1-02-012-16-0,atl1-1-02-014-9-0
#SBATCH --job-name=eval_cart_ep
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/eval_cart_%x_%j.log

# Eval-only pass of a CARTESIAN opsplit checkpoint through the same offline
# path as eval_palm_kp*.sbatch (1 GPU, 160 val batches, reverse-KL off, no viz),
# so its cam_xyz_paired_mse_avg can be compared with the keypoint run's
# offline palm-origin MSE without the offline-vs-in-training discrepancy.
#   FRAME=cam|wrist  CKPT=<path>  LABEL=<wandb/hydra name>

set -euo pipefail
WT=/storage/project/r-dxu345-0/agao81/EgoVerse/.claude/worktrees/fold-freeform-kp
REPO=/storage/project/r-dxu345-0/agao81/EgoVerse
cd "$WT"
export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if grep -q api.wandb.ai ~/.netrc 2>/dev/null; then export WANDB_MODE=online; else export WANDB_MODE=offline; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Private compile caches: two same-user jobs autotuning on one node with the
# shared /tmp/torchinductor_agao81 raced and both died in a compile-worker
# subprocess (2026-09-02, jobs 12752435/36 on atl1-1-03-020-18-0).
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/torchinductor_${USER}_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton_${USER}_${SLURM_JOB_ID}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

FRAME="${FRAME:?cam|wrist}"; CKPT="${CKPT:?}"; LABEL="${LABEL:?}"
if [[ "$FRAME" == "cam" ]]; then DATA=mecka_fold_freeform_opsplit_pi_cam6d; EV=eval_pi_camframe_6d; NORM=cam6d; EXTRA="+evaluator.viz_max_batches=0"
else DATA=mecka_fold_freeform_opsplit_pi_6d; EV=eval_pi_wristframe_6d; NORM=wrist6d; EXTRA="evaluator.viz_max_batches=0"; fi

echo "[eval] cwd=$(pwd) commit=$(git rev-parse HEAD) frame=$FRAME ckpt=$CKPT"
"$PY" -u egomimic/trainHydra.py \
  --config-name=train_zarr_cartesian_pi \
  data=$DATA model=pi0.5_bc_mecka_6d evaluator=$EV $EXTRA \
  evaluator.viz_every_n_epochs=1000000 \
  +evaluator.limit_val_batches=160 \
  launch_params.gpus_per_node=1 \
  ++mode=eval \
  "ckpt_path='$CKPT'" \
  norm_stats.precomputed_norm_path=/storage/project/r-dxu345-0/agao81/norm_stats/fold_freeform_opsplit_$NORM/norm_stats \
  model.robomimic_model.config.pytorch_weight_path=$REPO/egomimic/algo/pi_checkpoints/pi05_base_pytorch \
  model.robomimic_model.config.reverse_kl_samples=1 \
  ~data.train_viz_datasets ~data.train_viz_dataloader_params \
  "name=$LABEL" description=ep99_valset_160batches \
  "+logger.wandb.name=$LABEL" \
  'hydra.run.dir=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/${name}/${description}_${now:%Y-%m-%d_%H-%M-%S}'
echo "[eval] exit=$?"
