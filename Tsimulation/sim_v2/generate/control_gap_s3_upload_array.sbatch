#!/bin/bash
# Parallel, resumable upload for the 6 x 13 PushShapes control-gap dataset.
# One task owns one mode/embodiment cell. Re-running uses s5cmd sync, so already
# uploaded objects with matching sizes are skipped.
#SBATCH --job-name=ps_gap_s3
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-77%16
#SBATCH --open-mode=append
#SBATCH --output=/coc/flash7/paphiwetsa3/experiments/pushshapes_gapgen_all6x13_1000_20260826/logs_s3/%x_%A_%a.out

set -euo pipefail

EXP=/coc/flash7/paphiwetsa3/experiments/pushshapes_gapgen_all6x13_1000_20260826
DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/control_gap_all6x13_1000_per_cell_simv2_20260826
S5=/coc/flash7/paphiwetsa3/projects/EgoVerse-refactor/.venv/bin/s5cmd
REMOTE_ROOT=s3://rldb/processed_v3/pushshapes_sim/control_gap_all6x13_1000_per_cell_simv2_20260826

modes=(ideal tight loose laggy sticky jittery)
agents=(circle circle_small stick L u_socket gripper chain_gripper suction triangle umi scoop flipper spring)

task=${SLURM_ARRAY_TASK_ID:?array task id is required}
mode=${modes[$((task / ${#agents[@]}))]}
agent=${agents[$((task % ${#agents[@]}))]}
source_dir="$DATA/$mode/$agent/T"
destination="$REMOTE_ROOT/$mode/$agent/T/"

if [[ ! -d "$source_dir" ]]; then
  echo "FATAL: source directory does not exist: $source_dir" >&2
  exit 2
fi
if [[ ! -f "$HOME/.egoverse_env" ]]; then
  echo "FATAL: missing $HOME/.egoverse_env" >&2
  exit 2
fi

# s5cmd expects AWS-compatible names; the project env stores the same R2
# credentials under R2_* names.
source "$HOME/.egoverse_env"
: "${R2_ACCESS_KEY_ID:?missing R2_ACCESS_KEY_ID}"
: "${R2_SECRET_ACCESS_KEY:?missing R2_SECRET_ACCESS_KEY}"
: "${R2_ENDPOINT_URL:?missing R2_ENDPOINT_URL}"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION=auto
export AWS_REGION=auto
unset AWS_SESSION_TOKEN AWS_SECURITY_TOKEN || true

local_episodes=$(find "$source_dir" -maxdepth 1 -type d -name '*.zarr' | wc -l)
echo "START task=$task mode=$mode agent=$agent local_episodes=$local_episodes"
echo "SOURCE=$source_dir"
echo "DESTINATION=$destination"
echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"

"$S5" --endpoint-url "$R2_ENDPOINT_URL" --numworkers 32 \
  --retry-count 20 --log error --stat \
  sync "$source_dir/*" "$destination"

# Every episode store has a root zarr.json. Counting those roots verifies that
# the remote cell contains every local episode after the resumable sync.
remote_episodes=$(
  "$S5" --endpoint-url "$R2_ENDPOINT_URL" --numworkers 16 \
    ls "${destination}*/zarr.json" \
    | awk '$NF ~ /(^|\/)episode_[^\/]+\.zarr\/zarr\.json$/ {count++} END {print count+0}'
)
if (( remote_episodes != local_episodes )); then
  echo "FATAL: remote episode count $remote_episodes != local $local_episodes" >&2
  exit 3
fi

echo "UPLOAD_COMPLETE task=$task mode=$mode agent=$agent episodes=$remote_episodes DATE=$(date --iso-8601=seconds)"
