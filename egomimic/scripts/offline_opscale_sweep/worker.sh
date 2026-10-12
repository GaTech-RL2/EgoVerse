#!/bin/bash
set -euo pipefail

OPS="${1:?rung required}"
MANIFEST="${2:?checkpoint manifest required}"
UNSEEN_DIR="${3:?private unseen validation folder required}"
FLAGSHIP="${4:?reviewed flagship episode required}"
UNSEEN_PREVIEW="${5:?unseen preview episode required}"
ROOT=/workspace/users/ani-cheluva/EgoVerse
OUT="$ROOT/logs/fixed_budget_opscale_hist/offline_sweep_20261010"
source "$ROOT/emimic/bin/activate"
export HF_HOME=/workspace/users/ani-cheluva/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"

# Each srun task is an independent single-GPU evaluator.
WORKER_INDEX="${SLURM_ARRAY_TASK_ID:-${SLURM_LOCALID:-0}}"
export SLURM_NTASKS=1 SLURM_NTASKS_PER_NODE=1 SLURM_PROCID=0 SLURM_LOCALID=0 SLURM_NODEID=0
export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 EGOMIMIC_REAP_GPU_ORPHANS=0
export WANDB_MODE=disabled
mapfile -t ROWS < "$MANIFEST"
FAILED=0
for INDEX in "${!ROWS[@]}"; do
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if (( INDEX != WORKER_INDEX )); then continue; fi
  elif (( INDEX % 8 != WORKER_INDEX )); then continue; fi
  IFS=$'\t' read -r EPOCH CKPT NORM SOURCE <<< "${ROWS[$INDEX]}"
  RUN_OUT="$OUT/op${OPS}/epoch_${EPOCH}"
  mkdir -p "$RUN_OUT"
  COMPLETE=1
  if [[ ! -s "$RUN_OUT/validation_metrics.json" ]]; then COMPLETE=0; fi
  for GROUP in seen_op_valid unseen_op_valid flagship_unseen_op_valid; do
    if [[ -z "$(find "$RUN_OUT/videos_open_loop/epoch_0/$GROUP" -maxdepth 1 -name '*.mp4' -print -quit 2>/dev/null)" ]]; then COMPLETE=0; fi
  done
  if (( COMPLETE == 0 )); then
    echo "START op=$OPS epoch=$EPOCH gpu=${CUDA_VISIBLE_DEVICES:-unset} checkpoint=$CKPT"
    EXTRA=(+data.metric_sampling.replace_invalid=true)
    if ! python -u egomimic/trainHydra.py \
      "--config-dir=$SOURCE" --config-name=config \
      mode=validate "ckpt_path='$CKPT'" \
      norm_stats.precomputed_norm_path="$NORM" \
      launch_params.gpus_per_node=1 launch_params.nodes=1 \
      trainer.limit_val_batches=800 \
      +validation_open_loop_metrics=true \
      open_loop_evaluator.execute_steps=30 \
      data.unseen_op_valid_datasets.unseen_human_bimanual.resolver.folder_path="$UNSEEN_DIR" \
      "data.open_loop_episodes.unseen=[$UNSEEN_PREVIEW,$FLAGSHIP]" \
      "${EXTRA[@]}" \
      "hydra.run.dir=$RUN_OUT" > "$RUN_OUT/run.log" 2>&1; then
      echo "FAILED op=$OPS epoch=$EPOCH log=$RUN_OUT/run.log" >&2
      FAILED=1
      continue
    fi
  fi
  for GROUP in seen_op_valid unseen_op_valid flagship_unseen_op_valid; do
    if [[ -z "$(find "$RUN_OUT/videos_open_loop/epoch_0/$GROUP" -maxdepth 1 -name '*.mp4' -print -quit 2>/dev/null)" ]]; then
      echo "FAILED video op=$OPS epoch=$EPOCH group=$GROUP" >&2
      FAILED=1
      continue 2
    fi
  done
  if ! python egomimic/scripts/offline_opscale_sweep/record_metrics.py \
      --op "$OPS" --epoch "$EPOCH" \
      --metrics "$RUN_OUT/validation_metrics.json" \
      --output "$OUT/op${OPS}-metrics.jsonl"; then
    echo "FAILED metrics op=$OPS epoch=$EPOCH" >&2
    FAILED=1
  fi
done
exit "$FAILED"
