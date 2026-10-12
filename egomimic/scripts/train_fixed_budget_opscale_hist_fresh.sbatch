#!/bin/bash
# Fresh two-GPU run of one history-ladder rung; no checkpoint or W&B resume.
# Usage: sbatch --job-name=opscale-hist-fresh-op4 this_file.sbatch 4
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
OPS="${1:?pass the operator count: 4, 8, 12, or 16}"
case "$OPS" in
  4|8|12|16) ;;
  *) echo "Unsupported operator count: $OPS" >&2; exit 2 ;;
esac
if [[ "${SLURM_RESTART_COUNT:-0}" != 0 ]]; then
  echo "This launcher only starts fresh jobs; refusing a Slurm restart." >&2
  exit 2
fi

REPO=/workspace/users/ani-cheluva/EgoVerse
OUT="$REPO/logs/fixed_budget_opscale_hist/rdt_vitb_op${OPS}_fresh_${SLURM_JOB_ID}"
RUN_ID="fixed_budget_opscale_hist_op${OPS}_fresh_${SLURM_JOB_ID}"
# Refuse accidental reuse even if a job is rerun outside Slurm's requeue path.
mkdir "$OUT"
cd "$REPO"
source emimic/bin/activate
export WANDB_PROJECT=EgoVerse WANDB_ENTITY=rl2-group WANDB_RESUME=never
export WANDB_DIR="$OUT"
export HF_HOME=/workspace/users/ani-cheluva/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 40000))
# Keep the real Slurm environment so the repo's rank/worker cleanup and
# per-job Inductor cache remain active. Slurm starts one rank per GPU.
echo "fresh_start job=$SLURM_JOB_ID ops=$OPS output=$OUT wandb_id=$RUN_ID ckpt_path=null"
exec srun --kill-on-bad-exit=1 --cpu-bind=none --gpu-bind=none \
  python -u egomimic/trainHydra.py \
  --config-name="experiments/fixed_budget_opscale_hist/rdt_1b_opscale${OPS}" \
  launch_params.gpus_per_node=2 launch_params.nodes=1 \
  mode=train ckpt_path=null \
  model.robomimic_model.pretrained=false \
  model.robomimic_model.pretrained_checkpoint=null \
  logger.wandb.project=EgoVerse logger.wandb.entity=rl2-group \
  logger.wandb.id="$RUN_ID" "+logger.wandb.resume=never" \
  "hydra.run.dir=$OUT"
