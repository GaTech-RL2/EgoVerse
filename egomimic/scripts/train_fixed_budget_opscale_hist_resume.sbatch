#!/bin/bash
# Resume an operator-scaling history rung from a Lightning checkpoint.
# Usage: sbatch this_file.sbatch <ops> <checkpoint> <wandb-run-id>
#SBATCH --account=loaner
#SBATCH --partition=rl2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=400G
#SBATCH --time=14-00:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/workspace/users/ani-cheluva/EgoVerse

set -euo pipefail
OPS="${1:?operator count required}"
CKPT="${2:?checkpoint required}"
RUN_ID="${3:?W&B run ID required}"
case "$OPS" in 4|8|12|16) ;; *) exit 2 ;; esac
test -f "$CKPT"
NORM_STATS="${CKPT%/checkpoints/*}/norm_stats/norm_stats.json"
test -f "$NORM_STATS"
REPO=/workspace/users/ani-cheluva/EgoVerse
OUT="$REPO/logs/fixed_budget_opscale_hist/rdt_vitb_op${OPS}_resume_${SLURM_JOB_ID}"
mkdir "$OUT"
ln -s "$CKPT" "$OUT/resume_checkpoint.ckpt"
cd "$REPO"
source emimic/bin/activate
export WANDB_PROJECT=EgoVerse WANDB_ENTITY=rl2-group WANDB_RESUME=allow
export WANDB_DIR="$OUT"
export HF_HOME=/workspace/users/ani-cheluva/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 40000))
echo "resume job=$SLURM_JOB_ID ops=$OPS checkpoint=$CKPT wandb_id=$RUN_ID"
exec srun --kill-on-bad-exit=1 --cpu-bind=none --gpu-bind=none \
  python -u egomimic/trainHydra.py \
  --config-name="experiments/fixed_budget_opscale_hist/rdt_1b_opscale${OPS}" \
  launch_params.gpus_per_node=2 launch_params.nodes=1 mode=train ckpt_path="$OUT/resume_checkpoint.ckpt" norm_stats.precomputed_norm_path="$NORM_STATS" \
  logger.wandb.project=EgoVerse logger.wandb.entity=rl2-group logger.wandb.id="$RUN_ID" \
  +logger.wandb.resume=allow "hydra.run.dir=$OUT"
