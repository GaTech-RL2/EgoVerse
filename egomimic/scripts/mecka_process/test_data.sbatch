#!/bin/bash
#SBATCH --job-name=test_data
#SBATCH --partition=cpu-medium          # Keep your lab partition (CPU jobs work here)
#SBATCH --account=gts-dxu345-rl2
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=22
#SBATCH --time=1880
#SBATCH --nodes=1                    # One node only
#SBATCH --output=test_data_%j.out
#SBATCH --error=test_data_%j.err
#SBATCH --qos=inferno

# Same Slurm resource idea as sync_s3.sbatch.
#
# IMPORTANT: Slurm runs a *copy* of this script under /var/lib/slurm/..., so the
# repo root cannot be taken from the script file path. We walk upward from
# SLURM_SUBMIT_DIR (cwd when you ran sbatch) until we find emimic/bin/activate,
# so submitting from mecka_process/, egomimic/, or the repo root all work.
#
# Optional: export EGOVERSE_ROOT=/path/to/EgoVerse if that directory contains
# emimic/bin/activate (skips the walk).
#
# Everything after `--` is passed to test_data.py.

set -euo pipefail

resolve_egoverse_root() {
  if [[ -n "${EGOVERSE_ROOT:-}" ]]; then
    local r
    r="$(cd "$EGOVERSE_ROOT" && pwd)"
    if [[ -f "$r/emimic/bin/activate" ]]; then
      echo "$r"
      return 0
    fi
    echo "error: EGOVERSE_ROOT=$r has no emimic/bin/activate" >&2
    return 1
  fi
  local d="${SLURM_SUBMIT_DIR:-$PWD}"
  d="$(cd "$d" && pwd)"
  while [[ "$d" != "/" ]]; do
    if [[ -f "$d/emimic/bin/activate" ]]; then
      echo "$d"
      return 0
    fi
    d="$(dirname "$d")"
  done
  echo "error: could not find EgoVerse repo (no emimic/bin/activate) above ${SLURM_SUBMIT_DIR:-$PWD}" >&2
  return 1
}

EGOVERSE_ROOT="$(resolve_egoverse_root)" || exit 1
TEST_DATA_PY="$EGOVERSE_ROOT/egomimic/scripts/mecka_process/test_data.py"

if [[ ! -f "$TEST_DATA_PY" ]]; then
  echo "error: test_data.py not found at $TEST_DATA_PY" >&2
  echo "  Repo root resolved to: $EGOVERSE_ROOT" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$EGOVERSE_ROOT/emimic/bin/activate"
export PYTHONPATH="$EGOVERSE_ROOT"

exec python "$TEST_DATA_PY" "$@"
