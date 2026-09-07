#!/bin/bash
#SBATCH --job-name=usocket-codec-rerank
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --exclude=bishop
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --array=0-0%1

set -euo pipefail
: "${TASK_ROOT:?set TASK_ROOT to an immutable output bundle}"
: "${SOURCE_ROOT:?set SOURCE_ROOT to the exact clean source checkout}"
: "${SIM_ROOT:?set SIM_ROOT to the exact simulator checkout}"
: "${RUNTIME_PYTHON:?set RUNTIME_PYTHON to the scheduled runtime}"
: "${DATASET:?set DATASET to the exact dataset root}"
: "${SPLIT_MANIFEST:?set SPLIT_MANIFEST to the exact episode manifest}"
: "${CANDIDATE_MANIFEST:?set CANDIDATE_MANIFEST to the exact candidate manifest}"
: "${EXPECTED_EPISODES:?set EXPECTED_EPISODES explicitly}"
: "${SOURCE_COMMIT:?set SOURCE_COMMIT explicitly}"
: "${SIMULATOR_COMMIT:?set SIMULATOR_COMMIT explicitly}"
: "${PROTOCOL_SHA256:?set PROTOCOL_SHA256 explicitly}"
: "${CANONICAL_LAUNCHER_SHA256:?set CANONICAL_LAUNCHER_SHA256 explicitly}"

mkdir -p "${TASK_ROOT}/results/shards" "${TASK_ROOT}/logs" "${TASK_ROOT}/sim_import"
if [[ ! -e "${TASK_ROOT}/sim_import/Tsimulation" ]]; then
  ln -s "${SIM_ROOT}/Tsimulation" "${TASK_ROOT}/sim_import/Tsimulation"
fi
export PYTHONPATH="${TASK_ROOT}/sim_import:${SOURCE_ROOT}:$(dirname "$0"):${PYTHONPATH:-}"
export SDL_VIDEODRIVER=dummy
export TSIM_VERSION=sim_v2
[[ "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" == "${SOURCE_COMMIT}" ]]
[[ -z "$(git -C "${SOURCE_ROOT}" status --porcelain --untracked-files=all)" ]]

args=(
  --index "${SLURM_ARRAY_TASK_ID}"
  --dataset "${DATASET}"
  --split-manifest "${SPLIT_MANIFEST}"
  --candidate-manifest "${CANDIDATE_MANIFEST}"
  --expected-episodes "${EXPECTED_EPISODES}"
  --output-root "${TASK_ROOT}/results/shards"
)
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then args+=(--preflight-only); fi
"${RUNTIME_PYTHON}" "$(dirname "$0")/codec_grid_shard.py" "${args[@]}"
