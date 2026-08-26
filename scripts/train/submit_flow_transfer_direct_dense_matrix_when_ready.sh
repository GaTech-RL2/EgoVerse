#!/bin/bash
set -Eeuo pipefail

REPO=/coc/flash7/paphiwetsa3/worktrees/flow-transfer-direct-dense-obstacle-20260826
LAUNCHER=$REPO/scripts/train/flow_transfer_direct_dense_matrix.sbatch
EXP_ROOT=/coc/flash7/paphiwetsa3/experiments/flow_transfer_direct_dense_obstacle_20260826
STATE_DIR=$EXP_ROOT/submission
LOCK_DIR=$STATE_DIR/.submit_lock
SLURM_BIN=/opt/slurm/Ubuntu-20.04/24.11.0/bin
export PATH=$SLURM_BIN:$PATH

mkdir -p "$STATE_DIR" "$EXP_ROOT/slurm" "$EXP_ROOT/runs" "$EXP_ROOT/smokes" "$EXP_ROOT/provenance"
if ! mkdir "$LOCK_DIR"; then
  printf 'Submission lock already exists: %s\n' "$LOCK_DIR" >&2
  exit 73
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

EXPECTED_HEAD=$(git -C "$REPO" rev-parse HEAD)
EXPECTED_LAUNCHER_SHA=$(sha256sum "$LAUNCHER" | awk '{print $1}')
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test ! -e "$STATE_DIR/SUBMITTED"

printf 'watcher_start=%s\nexpected_head=%s\nlauncher_sha=%s\n' \
  "$(date --iso-8601=seconds)" "$EXPECTED_HEAD" "$EXPECTED_LAUNCHER_SHA" \
  > "$STATE_DIR/watcher_identity.txt"

while ! timeout 20 "$SLURM_BIN/scontrol" ping 2>&1 | grep -q 'is UP'; do
  printf '%s controllers unavailable; retrying\n' "$(date --iso-8601=seconds)"
  sleep 60
done

while ! timeout 60 "$SLURM_BIN/sbatch" --test-only \
  --partition=rl2-lab --account=rl2-lab --qos=short \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --time=02:00:00 \
  --gres=gpu:l40s:1 --exclude=bishop --wrap=true; do
  printf '%s test-only submission failed; retrying\n' "$(date --iso-8601=seconds)"
  sleep 60
done

for full_spec in '1 4 96G' '2 8 128G'; do
  read -r test_gpus test_cpus test_mem <<< "$full_spec"
  while ! timeout 60 "$SLURM_BIN/sbatch" --test-only \
    --partition=rl2-lab --account=rl2-lab --qos=long \
    --nodes=1 --ntasks=1 --cpus-per-task="$test_cpus" --mem="$test_mem" --time=7-00:00:00 \
    --gres="gpu:l40s:$test_gpus" --exclude=bishop --wrap=true; do
    printf '%s full %sxL40S test-only failed; retrying\n' \
      "$(date --iso-8601=seconds)" "$test_gpus"
    sleep 60
  done
done

ARMS=(
  bc_usocket_latent
  bc_usocket_dp
  bc_chain_latent
  bc_chain_dp
  cotrain_obstacle_latent
  cotrain_obstacle_dp
)

: > "$STATE_DIR/jobs.tsv"
printf 'arm\tsmoke_job\tfull_job\tfull_gpus\n' >> "$STATE_DIR/jobs.tsv"

submit_until_accepted() {
  local output
  while true; do
    if output=$(timeout 90 "$SLURM_BIN/sbatch" --parsable "$@" 2>> "$STATE_DIR/sbatch_errors.log"); then
      printf '%s\n' "${output%%;*}"
      return 0
    fi
    printf '%s sbatch failed; retrying\n' "$(date --iso-8601=seconds)" >> "$STATE_DIR/sbatch_errors.log"
    sleep 60
  done
}

for arm in "${ARMS[@]}"; do
  smoke_job=$(submit_until_accepted \
    --partition=rl2-lab --account=rl2-lab --qos=short \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --time=02:00:00 \
    --gres=gpu:l40s:1 --exclude=bishop --requeue --signal=USR1@300 \
    --job-name="ftsm_${arm:0:12}" \
    --output="$EXP_ROOT/slurm/smoke_${arm}_%j.out" \
    --error="$EXP_ROOT/slurm/smoke_${arm}_%j.err" \
    --export="ALL,ARM=$arm,MODE=smoke,GPUS_EXPECTED=1,EXPECTED_HEAD=$EXPECTED_HEAD,EXPECTED_LAUNCHER_SHA=$EXPECTED_LAUNCHER_SHA" \
    "$LAUNCHER")

  full_gpus=1
  full_cpus=4
  full_mem=96G
  if [[ "$arm" == cotrain_* ]]; then
    full_gpus=2
    full_cpus=8
    full_mem=128G
  fi
  full_job=$(submit_until_accepted \
    --partition=rl2-lab --account=rl2-lab --qos=long \
    --nodes=1 --ntasks=1 --cpus-per-task="$full_cpus" --mem="$full_mem" --time=7-00:00:00 \
    --gres="gpu:l40s:$full_gpus" --exclude=bishop --requeue --signal=USR1@300 \
    --dependency="afterok:$smoke_job" \
    --job-name="ft_${arm:0:14}" \
    --output="$EXP_ROOT/slurm/full_${arm}_%j.out" \
    --error="$EXP_ROOT/slurm/full_${arm}_%j.err" \
    --export="ALL,ARM=$arm,MODE=full,GPUS_EXPECTED=$full_gpus,EXPECTED_HEAD=$EXPECTED_HEAD,EXPECTED_LAUNCHER_SHA=$EXPECTED_LAUNCHER_SHA" \
    "$LAUNCHER")

  printf '%s\t%s\t%s\t%s\n' "$arm" "$smoke_job" "$full_job" "$full_gpus" | tee -a "$STATE_DIR/jobs.tsv"
done

date --iso-8601=seconds > "$STATE_DIR/SUBMITTED"
printf 'All smoke and full jobs submitted.\n'
