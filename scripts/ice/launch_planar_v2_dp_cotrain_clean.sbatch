#!/bin/bash
# Canonical ICE entrypoint for fresh standard-DP clean co-training.
# Submit one GPU with: sbatch --gres=gpu:1 --constraint='H100|H200' ...

set -Eeuo pipefail
umask 077

MODE=${MODE:-preflight}
EXPECTED_HEAD=${EXPECTED_HEAD:?set the exact 40-character source commit}
EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?set the sha256 of this launcher}
ROOT=${ICE_DP_ROOT:-/storage/ice1/5/4/paphiwetsa3/egoverse/planar-v2-dp-cotrain-clean-20260903}
REPO=${ICE_DP_REPO:-$ROOT/source/EgoVerse-${EXPECTED_HEAD:0:8}}
PY=${ICE_DP_PYTHON:-/storage/ice1/5/4/paphiwetsa3/egoverse/unite-usocket-20260902/env/.venv/bin/python}
LAUNCHER=${BASH_SOURCE[0]}
U_DATA=${ICE_DP_U_SOCKET_DATA:-$ROOT/data/u_socket_3000_v2_clean}
CHAIN_DATA=${ICE_DP_CHAIN_GRIPPER_DATA:-$ROOT/data/chain_gripper_3000_v2}
EXPERIMENT=pusht/planar_v2_cotrain_clean_dp_standard
PARAMETER_COUNT=51506981
FULL_STEPS=240000
VALIDATE_EVERY=10000
CHECKPOINT_EVERY=20000
NORM_SHA256=9cc5684c6a9355babd4e39598451a3192c725f2591dafa7e0e9310ebaba01475

fail() { printf '[%s] FAIL line=%s status=%s command=%q\n' "$MODE" "$1" "$2" "$3" >&2; }
trap 'fail "$LINENO" "$?" "$BASH_COMMAND"' ERR

case "$MODE" in preflight|smoke|full) ;; *) printf 'MODE must be preflight, smoke, or full\n' >&2; exit 64 ;; esac
test "${#EXPECTED_HEAD}" = 40
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -x "$PY" && test -d "$U_DATA" && test -d "$CHAIN_DATA"

# This campaign is always a fresh run. Fail even if an old continuation variable
# leaks in from a caller; the Hydra override below is also fixed to null.
test -z "${ICE_INITIAL_CHECKPOINT:-}"
test -z "${ICE_INITIAL_CHECKPOINT_SHA256:-}"
test -z "${CKPT_PATH:-}"

cd "$REPO"
export PYTHONPATH=$REPO HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1

build_common() {
  local output=$1 wandb_id=$2 max_steps=$3 val_interval=$4 val_batches=$5 log_interval=$6
  COMMON=(
    --config-name=train_zarr_cartesian "+experiment=$EXPERIMENT" mode=train ckpt_path=null
    "data.train_datasets.pushshapes_sim_u_socket.resolver.folder_path=$U_DATA"
    "data.valid_datasets.pushshapes_sim_u_socket.resolver.folder_path=$U_DATA"
    "data.train_datasets.pushshapes_sim_chain_gripper.resolver.folder_path=$CHAIN_DATA"
    "data.valid_datasets.pushshapes_sim_chain_gripper.resolver.folder_path=$CHAIN_DATA"
    "++paths.root_dir=$output" "paths.output_dir=$output" "paths.work_dir=$REPO"
    "hydra.run.dir=$output" launch_params.gpus_per_node=1 launch_params.nodes=1
    trainer.devices=1 trainer.num_nodes=1 trainer.strategy=ddp trainer.precision=bf16
    "trainer.max_steps=$max_steps" trainer.limit_train_batches=1.0
    "trainer.val_check_interval=$val_interval" "trainer.limit_val_batches=$val_batches"
    trainer.num_sanity_val_steps=0 "trainer.log_every_n_steps=$log_interval"
    norm_stats.sample_frac=1.0 norm_stats.save_cache_dir=null
    +callbacks.model_checkpoint.monitor=null callbacks.model_checkpoint.save_top_k=-1
    callbacks.model_checkpoint.save_last=false callbacks.model_checkpoint.every_n_epochs=null
    +callbacks.model_checkpoint.train_time_interval=null
    +callbacks.model_checkpoint.save_on_train_epoch_end=false
    "callbacks.model_checkpoint.filename='epoch-{epoch}-step-{step}'"
    logger.wandb.entity=rl2-group logger.wandb.project=pushshapes-planar-v2
    logger.wandb.group=planar-v2-dp-cotrain-clean-20260903
    "logger.wandb.id=$wandb_id" "+logger.wandb.name=$wandb_id"
  )
}

validate_contract() {
  local config=$1 split_audit=$2 norm_path=$3 expected_steps=$4 expected_val=$5 expected_checkpoint=$6
  "$PY" scripts/ice/validate_planar_v2_dp_cotrain_clean_contract.py \
    --config "$config" --split-audit "$split_audit" --norm "$norm_path" \
    --parameter-count "$PARAMETER_COUNT" --expected-steps "$expected_steps" \
    --validation-interval "$expected_val" --checkpoint-interval "$expected_checkpoint"
}

materialize_preflight() {
  local evidence_dir=$1 runtime_dir=$2 wandb_id=$3
  mkdir -p "$evidence_dir"
  "$PY" scripts/data/materialize_local_episode_split.py \
    --domain pushshapes_sim_u_socket="$U_DATA" \
    --domain pushshapes_sim_chain_gripper="$CHAIN_DATA" \
    --valid-ratio 0.01 --seed 42 --output "$evidence_dir/split_audit.json"
  build_common "$runtime_dir" "$wandb_id" 2 1 1 1
  COMMON+=(+callbacks.model_checkpoint.every_n_train_steps=1 +logger.wandb.resume=never)
  "$PY" -m egomimic.trainHydra "${COMMON[@]}" norm_stats_only=true \
    trainer.accelerator=cpu trainer.devices=1 \
    "norm_stats.save_cache_dir=$runtime_dir" "hydra.run.dir=$runtime_dir/norm_materialize" \
    --cfg job --resolve > "$evidence_dir/norm_config.yaml"
  "$PY" -m egomimic.trainHydra "${COMMON[@]}" \
    "norm_stats.precomputed_norm_path=$runtime_dir/norm_stats" \
    --cfg job --resolve > "$evidence_dir/training_config.yaml"
  validate_contract "$evidence_dir/training_config.yaml" "$evidence_dir/split_audit.json" \
    "$runtime_dir/norm_stats" 2 1 1
}

if test "$MODE" = preflight; then
  PREFLIGHT=$ROOT/preflight/${EXPECTED_HEAD:0:12}
  test ! -e "$PREFLIGHT"
  materialize_preflight "$PREFLIGHT" "$PREFLIGHT" preflight
  exit 0
fi

test "${SLURM_JOB_NUM_NODES:?}" = 1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" -ge 8
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" = 1
case "$(nvidia-smi --query-gpu=name --format=csv,noheader)" in
  'NVIDIA H100 80GB HBM3'|'NVIDIA H200') ;;
  *) printf '[%s] unsupported GPU; require exactly one H100 or H200\n' "$MODE" >&2; exit 64 ;;
esac

if test "$MODE" = smoke; then
  test "${SLURM_RESTART_COUNT:-0}" = 0
  RUN_DIR=$ROOT/runs/smoke-job-${SLURM_JOB_ID:?}
  test ! -e "$RUN_DIR"
  mkdir -p "$RUN_DIR/provenance"
  WANDB_ID=planar-v2-dp-cotrain-clean-ice-smoke-${SLURM_JOB_ID}
  materialize_preflight "$RUN_DIR/provenance" "$RUN_DIR" "$WANDB_ID"
  nvidia-smi -q > "$RUN_DIR/provenance/nvidia-smi-q.txt"
  export WANDB_MODE=online WANDB_SILENT=true
  "$PY" -m egomimic.trainHydra "${COMMON[@]}" norm_stats_only=true \
    trainer.accelerator=cpu trainer.devices=1 \
    "norm_stats.save_cache_dir=$RUN_DIR" "hydra.run.dir=$RUN_DIR/norm_materialize"
  test -s "$RUN_DIR/norm_stats/norm_stats.json"
  srun --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 --unbuffered \
    "$PY" -m egomimic.trainHydra "${COMMON[@]}" \
    "norm_stats.precomputed_norm_path=$RUN_DIR/norm_stats" 2>&1 | tee "$RUN_DIR/training.log"
  "$PY" scripts/train/verify_planar_training_smoke.py "$RUN_DIR" \
    --expected-head "$EXPECTED_HEAD" --world-size 1 --parameter-count "$PARAMETER_COUNT"
  sha256sum "$RUN_DIR/norm_stats/norm_stats.json" > "$RUN_DIR/provenance/norm_stats.sha256"
  printf '[smoke] ALL_PASS job=%s wandb=%s\n' "$SLURM_JOB_ID" \
    "https://wandb.ai/rl2-group/pushshapes-planar-v2/runs/$WANDB_ID"
  exit 0
fi

# Full mode may requeue the same Slurm job, but it never restores an experiment
# checkpoint. Lightning's scheduler-generated hpc checkpoint is the only allowed
# intra-job continuation artifact.
SMOKE_DIR=${PASSED_SMOKE_DIR:?full mode requires PASSED_SMOKE_DIR}
test "$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$SMOKE_DIR/SMOKE_RESULT.json")" = passed
NORM=$SMOKE_DIR/norm_stats/norm_stats.json
SPLIT_AUDIT=$SMOKE_DIR/provenance/split_audit.json
test "$(sha256sum "$NORM" | awk '{print $1}')" = "$NORM_SHA256"
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
RUN_DIR=$ROOT/runs/full-job-${SLURM_JOB_ID:?}
WANDB_ID=planar-v2-dp-cotrain-clean-ice-full-${SLURM_JOB_ID}
if test "$RESTART_COUNT" = 0; then
  test ! -e "$RUN_DIR"
  mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/provenance/restart-0"
  cp "$SPLIT_AUDIT" "$SMOKE_DIR/SMOKE_RESULT.json" "$RUN_DIR/provenance/restart-0/"
  sha256sum "$NORM" > "$RUN_DIR/provenance/restart-0/norm_stats.sha256"
  WANDB_RESUME=never
else
  test -d "$RUN_DIR"
  test -e "$RUN_DIR/hpc_ckpt_1.ckpt" -o -e "$RUN_DIR/hpc_ckpt.ckpt"
  mkdir -p "$RUN_DIR/provenance/restart-$RESTART_COUNT"
  WANDB_RESUME=must
fi
nvidia-smi -q > "$RUN_DIR/provenance/restart-$RESTART_COUNT/nvidia-smi-q.txt"
scontrol show job -dd "$SLURM_JOB_ID" > "$RUN_DIR/provenance/restart-$RESTART_COUNT/slurm-job.txt"
build_common "$RUN_DIR" "$WANDB_ID" "$FULL_STEPS" "$VALIDATE_EVERY" 1.0 10
COMMON+=("norm_stats.precomputed_norm_path=$NORM")
COMMON+=(+callbacks.model_checkpoint.every_n_train_steps=$CHECKPOINT_EVERY)
COMMON+=("+logger.wandb.resume=$WANDB_RESUME")
COMMON+=("logger.wandb.tags=[standard-dp,cotrain,clean-only,no-obstacle,planar-v2,h16,ddim100,full,from-scratch,ice,single-gpu]")
if test "$RESTART_COUNT" = 0; then
  "$PY" -m egomimic.trainHydra "${COMMON[@]}" --cfg job --resolve \
    > "$RUN_DIR/provenance/restart-0/resolved-config.yaml"
  validate_contract "$RUN_DIR/provenance/restart-0/resolved-config.yaml" "$SPLIT_AUDIT" \
    "$NORM" "$FULL_STEPS" "$VALIDATE_EVERY" "$CHECKPOINT_EVERY"
fi
export WANDB_MODE=online WANDB_SILENT=true
exec srun --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 --unbuffered \
  "$PY" -m egomimic.trainHydra "${COMMON[@]}"
