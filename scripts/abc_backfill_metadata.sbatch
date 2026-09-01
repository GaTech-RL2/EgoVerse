#!/bin/bash
#SBATCH --job-name=abc_backfill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=720
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
set -euo pipefail

# Backfill embodiment / intrinsics / extrinsics onto the ABC episodes in R2.
#
#   sbatch scripts/abc_backfill_metadata.sbatch "<task substring>" <manifest> [--apply]
#
# Defaults to a DRY RUN: it reports what would change and writes nothing. Pass
# --apply as the third argument to commit the rewrite.
#
#   # dry run over the fold-and-stack family
#   sbatch scripts/abc_backfill_metadata.sbatch "fold and stack" /path/manifest.json
#   # commit it
#   sbatch scripts/abc_backfill_metadata.sbatch "fold and stack" /path/manifest.json --apply
#
# CPU only and network-bound: each episode is a ~1.2MB ranged GET of the source
# MCAP (the calibration lives in its first chunk) plus a small R2 get/put, run
# across a thread pool. No GPU, little memory.

# First arg is either a task substring, or ALL / --all for every lab='abc' episode.
TASK_CONTAINS="${1:-fold and stack}"
MANIFEST="${2:-/coc/flash7/scratch/acheluva3/abc_data/backfill_manifest.json}"
APPLY="${3:-}"

cd /coc/flash7/acheluva3/egoverse-arc
export HF_HOME="${HF_HOME:-/coc/flash7/scratch/acheluva3/.cache}"

echo "task_contains='$TASK_CONTAINS'  manifest=$MANIFEST  mode=${APPLY:-DRY-RUN}"
echo "cpus=${SLURM_CPUS_PER_TASK:-?}  node=$(hostname)"

case "$TASK_CONTAINS" in
  ALL|all|--all) SELECTOR=(--all) ;;
  *)             SELECTOR=(--task-contains "$TASK_CONTAINS") ;;
esac

srun .venv/bin/python -m egomimic.scripts.abc_process.backfill_abc_metadata \
    "${SELECTOR[@]}" \
    --manifest "$MANIFEST" \
    ${APPLY:+$APPLY}
