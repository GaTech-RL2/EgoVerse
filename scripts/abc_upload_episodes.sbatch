#!/bin/bash
#SBATCH --job-name=abc_upload
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
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

# Upload locally converted ABC episodes to R2 and register them in app.episodes.
#
#   sbatch scripts/abc_upload_episodes.sbatch <zarr_dir> <manifest> [--apply]
#
# Defaults to a DRY RUN. Pass --apply as the third argument to commit.
# Idempotent: episodes already in R2 and already in app.episodes are skipped,
# so a requeue resumes rather than re-uploading.
#
# CPU only and network-bound. Zarr sharding keeps each episode to ~35 objects,
# so this is a few thousand multipart PUTs rather than a small-file storm.

ZARR_DIR="${1:-/coc/flash7/scratch/acheluva3/abc_data/zarr}"
MANIFEST="${2:-/coc/flash7/scratch/acheluva3/abc_data/upload_manifest.json}"
APPLY="${3:-}"

cd /coc/flash7/acheluva3/egoverse-arc
export HF_HOME="${HF_HOME:-/coc/flash7/scratch/acheluva3/.cache}"

echo "zarr_dir=$ZARR_DIR  manifest=$MANIFEST  mode=${APPLY:-DRY-RUN}"
echo "cpus=${SLURM_CPUS_PER_TASK:-?}  node=$(hostname)"

srun .venv/bin/python -m egomimic.scripts.abc_process.upload_abc_episodes \
    --zarr-dir "$ZARR_DIR" \
    --manifest "$MANIFEST" \
    ${APPLY:+$APPLY}
