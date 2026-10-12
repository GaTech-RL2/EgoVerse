#!/bin/bash
#SBATCH --job-name=opscale-zarr
#SBATCH --account=loaner
#SBATCH --partition=loaner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/workspace/users/%u/logs/%x-%j.out
#SBATCH --error=/workspace/users/%u/logs/%x-%j.err

# Decode the clean folding-only raw episodes into zarr and point all op splits
# plus validation folders at those stores. CPU only: the node is shared, so
# this does not take a GPU. Completed episodes have a .ok marker and are skipped
# if the job is requeued.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
source /workspace/users/ani-cheluva/EgoVerse/emimic/bin/activate
cd /workspace/users/ani-cheluva/EgoVerse
SHARD="${SHARD:-0}"
SHARDS="${SHARDS:-1}"
OFFSET="${OFFSET:-0}"
COUNT="${COUNT:-}"
WORKERS="${WORKERS:-24}"
if [[ "${SKIP_CONVERT:-0}" != "1" ]]; then
  args=(
    --root /workspace/datasets/fixed_budget_opscale
    --manifest egomimic/hydra_configs/experiments/fixed_budget_opscale/manifest.json
    --workers "$WORKERS"
    --shard "$SHARD"
    --shards "$SHARDS"
    --offset "$OFFSET"
  )
  if [[ -n "$COUNT" ]]; then
    args+=(--count "$COUNT")
  fi
  python egomimic/scripts/data_download/mono_to_zarr.py convert "${args[@]}"
fi
if [[ "${DO_LINK:-1}" == "1" ]]; then
  python egomimic/scripts/data_download/mono_to_zarr.py link \
    --root /workspace/datasets/fixed_budget_opscale \
    --manifest egomimic/hydra_configs/experiments/fixed_budget_opscale/manifest.json
fi
