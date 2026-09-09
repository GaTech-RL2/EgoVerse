#!/bin/bash
#SBATCH --job-name=usocket-codec-aggregate
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail
: "${TASK_ROOT:?set TASK_ROOT to the immutable output bundle}"
: "${RUNTIME_PYTHON:?set RUNTIME_PYTHON to the scheduled runtime}"
: "${CANDIDATE_MANIFEST:?set CANDIDATE_MANIFEST explicitly}"
: "${EXPECTED_CANDIDATES:?set EXPECTED_CANDIDATES explicitly}"
export PYTHONPATH="$(dirname "$0"):${PYTHONPATH:-}"
"${RUNTIME_PYTHON}" "$(dirname "$0")/aggregate_codec_grid.py" \
  --shard-root "${TASK_ROOT}/results/shards" \
  --output-dir "${TASK_ROOT}/results/aggregate" \
  --candidate-manifest "${CANDIDATE_MANIFEST}" \
  --expected-candidates "${EXPECTED_CANDIDATES}"
