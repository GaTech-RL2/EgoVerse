#!/bin/bash
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=3-00:00:00
#SBATCH --job-name=egomimic
#SBATCH --output=slurm-%j.out

# Usage:
#   sbatch train.sbatch <hydra-overrides...>
#   sbatch --gres=gpu:h200:4 --ntasks=4 train.sbatch <hydra-overrides...>
#
# Defaults to a single-GPU h100 pi-style allocation (per CLAUDE.md: pi runs use
# h100/h200, everything else l40s — override --gres for l40s jobs).

set -euo pipefail

REPO_DIR="/storage/project/r-dxu345-0/agao81/EgoVerse"
VENV_PYTHON="${REPO_DIR}/emimic/bin/python"
APIKEY_FILE="${REPO_DIR}/apikey.txt"

cd "${REPO_DIR}"

# Source API keys (SCALE_API_KEY, OPENAI_API_KEY, ...). The Scale filter and
# annotation utilities read these from os.environ at instantiate time.
if [[ -f "${APIKEY_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${APIKEY_FILE}"
    set +a
else
    echo "WARNING: ${APIKEY_FILE} not found — runs using Scale/OpenAI will fail." >&2
fi

echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Python: ${VENV_PYTHON}"
echo "Overrides: $*"

# Wandb fix: spawn its service in /tmp instead of the (slow) shared FS, and
# extend the service-startup poll timeout 30s → 180s.
export WANDB_DIR="/tmp/wandb_${SLURM_JOB_ID:-$$}"
export WANDB_CACHE_DIR="/tmp/wandb_cache_${SLURM_JOB_ID:-$$}"
export WANDB_CONFIG_DIR="/tmp/wandb_config_${SLURM_JOB_ID:-$$}"
export WANDB_SERVICE_WAIT=180
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
# Reduce GPU memory fragmentation under heavy allocs (paligemma + action expert).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Convert NCCL collective hangs into exceptions (with a flight-recorder dump)
# rather than silent freezes. Pairs with the heartbeat watchdog in pl_model.
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1

# Silence the tokenizer fork warning (and its known-buggy auto-disable path).
export TOKENIZERS_PARALLELISM=false

# Optional: stage a dataset folder to node-local /tmp SSD before training.
# Removes shared-NFS reads from the hot path; combined with the worker
# timeout in dataloader_params this turns "silent NFS stall" into "clean
# requeue". Triggered when STAGE_FROM is exported.
#
#   sbatch --export=ALL,STAGE_FROM=/path/to/zarr[,STAGE_EMBODIMENTS=eva_bimanual,aria_bimanual] \
#     train.sbatch <hydra-overrides...>
#
# rsyncs $STAGE_FROM -> /tmp/$SLURM_JOB_ID/$(basename $STAGE_FROM) once and
# appends `data.train_datasets.<emb>.resolver.folder_path=<staged>` to the
# hydra overrides for every comma-separated embodiment in STAGE_EMBODIMENTS.
STAGE_OVERRIDES=()
if [[ -n "${STAGE_FROM:-}" ]]; then
    STAGE_DST="/tmp/${SLURM_JOB_ID:-$$}/$(basename "${STAGE_FROM}")"
    echo "Staging dataset ${STAGE_FROM} -> ${STAGE_DST}"
    mkdir -p "$(dirname "${STAGE_DST}")"
    time rsync -a "${STAGE_FROM%/}/" "${STAGE_DST}/"
    EMBS="${STAGE_EMBODIMENTS:-eva_bimanual,aria_bimanual}"
    IFS=',' read -r -a EMB_ARR <<< "${EMBS}"
    for emb in "${EMB_ARR[@]}"; do
        STAGE_OVERRIDES+=("data.train_datasets.${emb}.resolver.folder_path=${STAGE_DST}")
    done
fi

if [[ ${#STAGE_OVERRIDES[@]} -gt 0 ]]; then
    srun "${VENV_PYTHON}" egomimic/trainHydra.py "$@" "${STAGE_OVERRIDES[@]}"
else
    srun "${VENV_PYTHON}" egomimic/trainHydra.py "$@"
fi
