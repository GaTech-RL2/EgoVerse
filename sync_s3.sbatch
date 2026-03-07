#!/bin/bash
#SBATCH --job-name=sync_s3
#SBATCH --partition=CPU-3TB-AMD          # Keep your lab partition (CPU jobs work here)
#SBATCH --account=gts-dxu345-rl2
#SBATCH --mem=100G
#SBATCH --ntasks=1                  
#SBATCH --nodes=1                    # One node only
#SBATCH --output=sync_s3_%j.out
#SBATCH --error=sync_s3_%j.err
#SBATCH --qos=inferno

# Usage:
#   sbatch sync_s3.sbatch -- [sync_s3.py args...]
#
# Everything after `--` is passed directly to `sync_s3.py`.

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
python "$SUBMIT_DIR/sync_s3.py" "$@"
