#!/bin/bash
#SBATCH --job-name=seed_eval
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/seed_eval_%j.out
# ---------------------------------------------------------------------------
# OFFLINE SEED-PROTOCOL sim-eval launcher  (GOAL verdict: >0.5 val-SEED coverage)
#
# Parametrized template. Submit with RUN_NAME + CKPT_EPOCH exported, e.g.
#
#   sbatch --export=ALL,RUN_NAME=hnet_big_circle3000,CKPT_EPOCH=1499 \
#          scripts/gmm_launch/seed_eval_TEMPLATE.sbatch
#
# Optional overrides (export to change defaults):
#   N_EPISODES   (default 20)   reproducible seed-suite size
#   INIT_SEED_BASE (default 1000) episode i uses env seed = base + i
#   MAX_STEPS    (default 400)
#   ONLY_EMB     (default 15)    big circle / emb15
#   PUSHER       (default circle) env pusher_shape (use circle_small for emb17)
#   OUT_TAG      (default ${RUN_NAME}_e${ONLY_EMB}_ep${CKPT_EPOCH})
#
# What this does:
#   1. Resolves the run's newest non-smoke logdir under logs/<RUN_NAME>*.
#   2. Snapshots the ckpt epoch_epoch=N.ckpt -> a clean '='-free path
#      (defensive vs any '='-sensitive parsing; argparse itself is fine).
#   3. Runs egomimic.eval.core.ckpt_loading with init_mode=seeds, reusing the
#      run's OWN norm_stats (loaded from the ckpt's hyper_parameters.
#      norm_stats_state -- NO recompute, no precomputed_norm_path needed) and
#      the run's OWN data/.hydra/config.yaml (packed loader for hnet/txar,
#      per-frame for hpt). With init_mode=seeds the batch CONTENT is unused
#      (obs come live from the env) but a batch must still be built to enter
#      the rollout loop -- the run's own data config provides it.
#   4. Per-family model/eval-class/obs-stride are selected by RUN_NAME below.
#
# NOTE on the verdict interface: the *working* offline seed eval in this repo
# is the argparse CLI egomimic.eval.core.ckpt_loading (NOT a Hydra
# `trainHydra mode=eval init_mode=seeds` invocation). All verified seed-eval
# launchers (scripts/gmm_launch/fixsw_*.sbatch) use this CLI with
# --init-mode seeds. evaluator.init_seeds / mode=eval are NOT how this codebase
# runs the seed verdict.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
cd "$REPO"
export PYTHONPATH=.
export MUJOCO_GL=egl

# ---- required params -------------------------------------------------------
: "${RUN_NAME:?export RUN_NAME=<one of the 7 runs>}"
: "${CKPT_EPOCH:?export CKPT_EPOCH=<epoch int, e.g. 1499>}"

# ---- tunable defaults ------------------------------------------------------
N_EPISODES="${N_EPISODES:-20}"          # ~20-seed reproducible suite
INIT_SEED_BASE="${INIT_SEED_BASE:-1000}"
MAX_STEPS="${MAX_STEPS:-400}"
ONLY_EMB="${ONLY_EMB:-15}"              # emb15 = big circle
PUSHER="${PUSHER:-circle}"             # circle = big circle env pusher
SEED="${SEED:-0}"
OUT_TAG="${OUT_TAG:-${RUN_NAME}_e${ONLY_EMB}_ep${CKPT_EPOCH}}"

# ---- per-family wiring: eval-class + obs-stride ----------------------------
# hnet_*  : packed loader, H-Net algo, re-plan cadence obs_stride=8.
# txar_*  : packed loader, windowed-BC (tx) algo, obs_stride=8.
# hpt_*   : per-frame loader, HPT algo (eval-class hpt). HPT has no obs_stride
#           re-plan knob (it uses replan_every / chunk_size internally), so we
#           DO NOT pass --obs-stride for hpt.
case "$RUN_NAME" in
  hnet_*|txar_*)
    EVAL_CLASS="packed"
    OBS_STRIDE_ARG="--obs-stride 8"
    ;;
  hpt_*)
    EVAL_CLASS="hpt"
    OBS_STRIDE_ARG=""
    ;;
  *)
    echo "ERROR: unknown RUN_NAME family '$RUN_NAME' (expected hnet_*/txar_*/hpt_*)" >&2
    exit 2
    ;;
esac

# ---- resolve the run's newest NON-smoke logdir -----------------------------
# Run dirs look like: logs/<RUN_NAME>/<RUN_NAME>_<desc>_<timestamp>/  OR
#                     logs/<RUN_NAME>_<desc>_<timestamp>/
# We search both, exclude *_smoke, and take the newest that actually has the
# requested ckpt.
CKPT_REL=""
RUN_DIR=""
# Candidate run dirs (newest last), excluding smoke dirs.
mapfile -t CANDS < <(find "$REPO/logs" -maxdepth 2 -type d -name "*${RUN_NAME}*" \
                       ! -path "*_smoke*" -printf '%T@ %p\n' 2>/dev/null \
                     | sort -n | awk '{print $2}')
for d in "${CANDS[@]}"; do
  c="$d/checkpoints/epoch_epoch=${CKPT_EPOCH}.ckpt"
  if [[ -f "$c" ]]; then
    RUN_DIR="$d"
    CKPT_REL="$c"
  fi
done

if [[ -z "$CKPT_REL" ]]; then
  echo "ERROR: no ckpt 'epoch_epoch=${CKPT_EPOCH}.ckpt' found for RUN_NAME=${RUN_NAME}" >&2
  echo "Searched (non-smoke) run dirs:" >&2
  printf '  %s\n' "${CANDS[@]}" >&2
  exit 3
fi

CONFIG_PATH="$RUN_DIR/.hydra/config.yaml"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: missing $CONFIG_PATH" >&2
  exit 4
fi

echo "[seed_eval] RUN_NAME   = $RUN_NAME"
echo "[seed_eval] RUN_DIR    = $RUN_DIR"
echo "[seed_eval] EVAL_CLASS = $EVAL_CLASS"
echo "[seed_eval] CKPT (orig)= $CKPT_REL"
echo "[seed_eval] CONFIG     = $CONFIG_PATH"
echo "[seed_eval] emb=$ONLY_EMB pusher=$PUSHER n_episodes=$N_EPISODES seed_base=$INIT_SEED_BASE max_steps=$MAX_STEPS"

# ---- snapshot ckpt to a clean '='-free path --------------------------------
# The original filename contains '=' (epoch_epoch=N.ckpt). argparse handles it
# fine when quoted, but we snapshot to a stable '='-free name so nothing
# downstream (or any future Hydra wrapper) trips on it.
SNAP_DIR="$REPO/fixsweep/_seed_eval_ckpts"
mkdir -p "$SNAP_DIR"
SNAP_CKPT="$SNAP_DIR/${RUN_NAME}_ep${CKPT_EPOCH}.ckpt"
if [[ ! -f "$SNAP_CKPT" ]]; then
  cp -f "$CKPT_REL" "$SNAP_CKPT"
fi
echo "[seed_eval] CKPT (snap)= $SNAP_CKPT"

OUT_DIR="$REPO/fixsweep/seed_eval/${OUT_TAG}"
mkdir -p "$OUT_DIR"

# ---- run the seed-protocol offline sim eval --------------------------------
# init_mode=seeds builds init_seeds=[base, base+1, ..., base+N-1] internally.
# --rng-pairing pairs stochastic-head sampling noise across runs per episode.
# norm_stats come from the ckpt itself; data comes from --config-path's data
# subtree (packed for hnet/txar, per-frame for hpt).
set -x
srun .venv/bin/python -m egomimic.eval.core.ckpt_loading \
  --ckpt "$SNAP_CKPT" \
  --config-path "$CONFIG_PATH" \
  --eval-class "$EVAL_CLASS" \
  --n-episodes "$N_EPISODES" \
  --max-steps "$MAX_STEPS" \
  $OBS_STRIDE_ARG \
  --seed "$SEED" \
  --embodiment-name pushshapes_sim \
  --only-emb "$ONLY_EMB" \
  --pusher "$PUSHER" \
  --init-mode seeds \
  --init-seed-base "$INIT_SEED_BASE" \
  --rng-pairing \
  --out-dir "$OUT_DIR"
EVAL_EXIT=$?
set +x

echo "SEED_EVAL_EXIT_${RUN_NAME}_ep${CKPT_EPOCH}=${EVAL_EXIT}"
echo "[seed_eval] verdict source: grep '=== METRICS ===' / 'Valid/emb${ONLY_EMB}_sim_coverage' and"
echo "[seed_eval]                 '[sim] emb${ONLY_EMB} ep_coverages:' lines in this log + $OUT_DIR"
echo "DONE_seed_eval_${OUT_TAG}"
exit $EVAL_EXIT
