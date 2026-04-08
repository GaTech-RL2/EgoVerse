#!/usr/bin/env bash
# Submit a norm-stats-only job to PACE via the Hydra submitit launcher.
#
# Mirrors the trainHydra invocation:
#   python egomimic/trainHydra.py --config-name train_zarr_cartesian_pi \
#     data=mecka_pi model=pi0.5_bc_mecka visualization=pi_cartesian_lang \
#     name=mecka_pi description=mecka_pi -m
#
# but routes through compute_norm_stats_only.py (no model/trainer/GPU) and
# overrides the launcher to the CPU-only PACE launcher.
#
# Usage:
#   bash egomimic/scripts/launch_norm_stats_pace.sh \
#        [extra hydra overrides...]
#
# Example:
#   bash egomimic/scripts/launch_norm_stats_pace.sh data=mecka_pi \
#        model=pi0.5_bc_mecka name=mecka_pi description=mecka_pi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LAUNCHER_NAME="submitit_pace_cpu"
LAUNCHER_YAML="${REPO_ROOT}/egomimic/hydra_configs/hydra/launcher/${LAUNCHER_NAME}.yaml"

# Derive num_workers from the launcher's cpus_per_task (leave 1 core for the
# main process). Falls back to 8 if the field can't be parsed.
CPUS_PER_TASK="$(awk '/^cpus_per_task:/ {print $2; exit}' "${LAUNCHER_YAML}" 2>/dev/null || true)"
if [[ "${CPUS_PER_TASK}" =~ ^[0-9]+$ ]] && (( CPUS_PER_TASK > 1 )); then
  NUM_WORKERS=$(( CPUS_PER_TASK - 1 ))
else
  echo "warn: could not parse cpus_per_task from ${LAUNCHER_YAML}, defaulting num_workers=8" >&2
  NUM_WORKERS=8
fi
echo "info: using norm_stats.num_workers=${NUM_WORKERS} (cpus_per_task=${CPUS_PER_TASK:-?})"

# Default overrides — match the user's example invocation.
DEFAULT_OVERRIDES=(
  --config-name train_zarr_cartesian_pi
  data=mecka_pi
  model=pi0.5_bc_mecka
  visualization=pi_cartesian_lang
  name=mecka_pi_norm_stats
  description=mecka_pi_norm_stats
  "hydra/launcher=${LAUNCHER_NAME}"
  norm_stats.sample_frac=1.0
  norm_stats.precomputed_norm_path=null
  norm_stats.checkpoint_every_frac=0.2
  "norm_stats.num_workers=${NUM_WORKERS}"
)

cd "${REPO_ROOT}"

python egomimic/compute_norm_stats_only.py \
  "${DEFAULT_OVERRIDES[@]}" \
  "$@" \
  -m
