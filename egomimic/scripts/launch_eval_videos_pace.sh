#!/usr/bin/env bash
# Submit an eval-mode validation-video job to PACE via the Hydra submitit
# launcher.
#
# Reloads the exact training config from a previous run's saved `.hydra/`,
# loads a checkpoint, and runs `trainer.validate()` so the PIEvalVideo
# evaluator writes fresh validation videos.
#
# The validation dataloader is NOT shuffled, so each output video plays back
# coherent per-episode trajectories (consecutive frames = consecutive
# timesteps). NOTE: this means the episodes shown match the run's own
# training-time validation videos (both iterate the val set from the start).
#
# Usage:
#   bash egomimic/scripts/launch_eval_videos_pace.sh <RUN_DIR> [extra hydra overrides...]
#
#   RUN_DIR   per-task run directory containing .hydra/ and checkpoints/,
#             e.g. logs/mecka_pi/mecka_pi_2026-05-16_17-46-27/0
#
# Env knobs (all optional):
#   EVAL_SEED          global RNG seed (default 1234)
#   LIMIT_VAL_BATCHES  number of validation batches => more videos (default 2000)
#   CKPT               checkpoint file under RUN_DIR/checkpoints (default last.ckpt)
#
# Example:
#   bash egomimic/scripts/launch_eval_videos_pace.sh \
#        logs/mecka_pi/mecka_pi_2026-05-16_17-46-27/0
#
#   EVAL_SEED=777 LIMIT_VAL_BATCHES=4000 \
#     bash egomimic/scripts/launch_eval_videos_pace.sh \
#        logs/mecka_pi/mecka_pi_2026-05-16_17-46-27/0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_DIR_ARG="${1:?usage: launch_eval_videos_pace.sh <RUN_DIR> [hydra overrides...]}"
shift

# Resolve RUN_DIR to an absolute path.
RUN_DIR="$(cd "${RUN_DIR_ARG}" && pwd)"

CKPT_NAME="${CKPT:-last.ckpt}"
CKPT_PATH="${RUN_DIR}/checkpoints/${CKPT_NAME}"
HYDRA_DIR="${RUN_DIR}/.hydra"

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "error: checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${HYDRA_DIR}/config.yaml" ]]; then
  echo "error: saved hydra config not found: ${HYDRA_DIR}/config.yaml" >&2
  exit 1
fi

# Hydra's --config-path is resolved relative to trainHydra.py's directory
# (egomimic/), so express the saved .hydra dir relative to that.
CONFIG_PATH="$(realpath --relative-to="${REPO_ROOT}/egomimic" "${HYDRA_DIR}")"

# Timestamped output dir, nested under the run so the new videos sit next to
# the run but never clobber its original .hydra/ or training-time videos/.
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
SWEEP_DIR="${RUN_DIR}/eval_videos/${STAMP}"

# Global RNG seed (seed_everything). With the val loader unshuffled this does
# not change which episodes are shown; kept for reproducibility.
EVAL_SEED="${EVAL_SEED:-1234}"

# More validation batches => more episodes => more videos (a video is flushed
# every 1000 buffered frames). Lightning caps this at the val set size.
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-2000}"

echo "info: run dir            ${RUN_DIR}"
echo "info: checkpoint         ${CKPT_PATH}"
echo "info: eval seed          ${EVAL_SEED} (training used 42)"
echo "info: limit_val_batches  ${LIMIT_VAL_BATCHES}"
echo "info: videos will land   ${SWEEP_DIR}/0/videos/epoch_0/<EMBODIMENT>/"

cd "${REPO_ROOT}"

python egomimic/trainHydra.py -m \
  --config-path="${CONFIG_PATH}" \
  --config-name=config \
  hydra.searchpath="[file://egomimic/hydra_configs]" \
  hydra/launcher=submitit_pace_eval \
  hydra.sweep.dir="${SWEEP_DIR}" \
  hydra.sweep.subdir=0 \
  ++mode=eval \
  ++seed="${EVAL_SEED}" \
  ++ckpt_path="'${CKPT_PATH}'" \
  ++evaluator.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  ++data.valid_dataloader_params.mecka_bimanual.num_workers=3 \
  "$@"
