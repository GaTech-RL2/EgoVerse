#!/usr/bin/env bash
# run_rollout_pcd_policy_bgobs.sh — copy of run_rollout_pcd_policy.sh (working tree, 2026-11-20) that runs
# rollout_rby1_policy_real_bgobs.py with --bg-obs (BG_OBS=1 default). Same env knobs otherwise.
set -euo pipefail

# Closed-loop hardware rollout for the POINT-CLOUD / DEPTH policies (DP3 / Adapt3R).
# Same skeleton as run_rollout_aria_egoposer.sh, but the policy obs comes from
# live Fast-FoundationStereo depth on the Aria SLAM front stereo pair:
#   OBS_MODE=pcd            -> DP3      front_pcd_1 (N,3) float32, metres, glass frame
#   OBS_MODE=depth          -> Adapt3R  rect-gray front_img_1 (224²x3) + aria_depth (224²)
#   OBS_MODE=rgb_lut        -> h_rect    LUT colour front_img_1, no depth      (port 8000)
#   OBS_MODE=rgb_lut_depth  -> a3r_human LUT colour front_img_1 + aria_depth   (port 8002)
#
# The two rgb_lut modes take the image from the RAW RGB fisheye through
# robot_rect224_lut.npz, which warps camera-rgb INTO the rectified-left-SLAM
# frame the depth lives in — that is why image and depth are pixel-aligned
# despite coming from two different cameras. See
# docs/ai/guides/a3r_human_hardware_eval.md.
#
# Pair with the EgoVerse server (emimic venv), e.g.:
#   python egomimic/scripts/serve_policy.py \
#     --checkpoint checkpoints/RBY1_dp3_pcd1024/dp3_pcd1024_glass_2k/checkpoints/epoch_epoch=1299.ckpt \
#     --port 8000
#
# Usage:
#   bash run_rollout_pcd_policy.sh                          # LIVE DP3 (pcd), port 8000
#   OBS_MODE=depth bash run_rollout_pcd_policy.sh          # LIVE Adapt3R
#   OBS_MODE=rgb_lut_depth bash run_rollout_pcd_policy.sh  # LIVE a3r_human (:8002)
#   OBS_MODE=rgb_lut_depth PORT=8031 bash run_rollout_pcd_policy.sh  # LIVE a3r_full (0827, invalid-depth masking; same obs contract)
#   OBS_MODE=rgb_lut       bash run_rollout_pcd_policy.sh  # LIVE h_rect    (:8000)
#   PCD_RECIPE=tight PORT=8003 bash run_rollout_pcd_policy.sh  # dp3_tight1024
#   DRY_RUN=1 bash run_rollout_pcd_policy.sh               # open-loop preview, no robot motion
#   ZOOM_CAPTURE=1 RECORD_RAW_DIR=... bash run_rollout_pcd_policy.sh  # dense figure dump
#   ABL=GZ GAZE_ENS=1 bash run_rollout_pcd_policy.sh       # GZ on the 6-member robust gaze ensemble
#   GT_MODE=gt_proprio DEMO_NAME=0 bash run_rollout_pcd_policy.sh   # dataset GT obs (plumbing)
#   GT_MODE=gt_proprio_live_image bash run_rollout_pcd_policy.sh    # GT proprio + LIVE cloud
#   ZERO_OBS=1 bash run_rollout_pcd_policy.sh              # zeros cloud probe (reliance check)
#
# NOTE: N (cloud size) is baked into the DP3 checkpoint. Default auto-resolves from
# server metadata / dataset; force with PCD_NUM_POINTS=1024 or 2048.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # SEW-Geometric-Teleop root
cd "$HERE"

CONDA_BASE="${CONDA_BASE:-/home/aloha/anaconda3}"
CONDA_ENV="${CONDA_ENV:-rby1}"
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PATH=":${PATH}:"; PATH="${PATH//:${VIRTUAL_ENV}\/bin:/:}"; PATH="${PATH#:}"; PATH="${PATH%:}"
  unset VIRTUAL_ENV
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PY="${PY:-${CONDA_PREFIX}/bin/python}"
echo "[run] python: ${PY}  (conda env: ${CONDA_DEFAULT_ENV:-?})"
"${PY}" -c "import cv2, mujoco, rby1_sdk, torch" 2>/dev/null || {
  echo "[run] ERROR: ${PY} is missing cv2/mujoco/rby1_sdk/torch. Set PY= or CONDA_ENV." >&2
  exit 1
}

HOST="${HOST:-127.0.0.1}"

# ----- abl0921 ablation fleet (2026-09-21) ---------------------------------
# ABL=<variant> selects the obs-pipeline preset (python-side ABL0921_PRESETS
# — the single config authority; a wrong pipeline degrades QUIETLY), the
# serving port (serve with EgoVerse serve_abl_0921.sh <variant>), the basket
# reset dataset, and EXEC_STEPS=32 (64-step chunks). Extra knobs:
#   ASYNC=1            chunk-overlap inference (spec 4b; skips the settle)
#   ASYNC_LOOKAHEAD=8  prefetch when this many actions remain
#   LAYOUT_TAG=...     start-pose/object-layout tag -> rollouts.jsonl (spec 7)
ABL="${ABL:-}"
if [[ -n "$ABL" ]]; then
  declare -A ABL_PORTS=(
    [V0]=8100 [A1]=8101 [A2]=8102 [A3]=8103 [A4]=8104 [A5]=8105
    [B2]=8106 [C3]=8107 [C4]=8108 [D1]=8109 [D3]=8110 [E4]=8111
    [G1]=8112 [G4]=8113 [G6]=8114 [G7]=8115 [H3]=8116 [H4]=8117 [H5]=8118
    [J1]=8119 [J2]=8120 [J3]=8121 [J3b]=8122 [J4]=8123 [J4b]=8124 [J5]=8125
    [G2]=8126 [G3]=8127 [G8]=8128 [S1]=8129 [S2]=8130 [S3]=8131 [S4]=8132 [I2]=8133 [I3]=8134
    [CHUYE]=8140 [YW]=8141 [MIX]=8142
    [W1]=8143 [W2]=8144 [W3]=8145 [GZ]=8146 [TGZ]=8147 [TGZ12]=8149 [TGZ12HD]=8150 [TGZ10HD2]=8156 [TGZ10HD2Z5]=8159 [TGZ10HD2NG]=8157 [TGZ10HD2NC]=8158 [TGZ1M]=8148
    [R1]=8190   # 2026-10-14: ycho420 basket60_r1 (reproduction of basket60_v2, V0 pipeline); 8147 -> 8148 (10-28) -> 8190 (10-29): the team's TGZ presets take 8147+
  )
  [[ -z "${ABL_PORTS[$ABL]:-}" ]] && { echo "[run] ERROR: unknown ABL='$ABL' (V0 A1-A5 B2 C3 C4 D1 D3 E4 G1-G4 G6-G8 H3-H5 I2 I3 J1-J5/J3b/J4b S1-S4 W1-W3 GZ TGZ TGZ12 TGZ12HD TGZ1M R1)" >&2; exit 1; }
  PORT="${PORT:-${ABL_PORTS[$ABL]}}"
  OBS_MODE="${OBS_MODE:-pcd}"
  PCD_RECIPE="${PCD_RECIPE:-dual_rgb}"
  EXEC_STEPS="${EXEC_STEPS:-32}"
  echo "[run] ABL=${ABL} -> port ${PORT}, EXEC_STEPS=${EXEC_STEPS}"
fi

# Port defaults follow the 0806 session's one-port-per-policy convention, so
# two servers can stay up for a back-to-back A/B without re-pointing anything.
case "${OBS_MODE:-pcd}" in
  rgb_lut_depth_eef) PORT="${PORT:-8006}" ;;   # a3r_eef
  rgb_lut_depth)     PORT="${PORT:-8002}" ;;   # a3r_human (0827 a3r_full: PORT=8031)
  rgb_lut)           PORT="${PORT:-8000}" ;;   # h_rect
  pcd)
    case "${PCD_RECIPE:-glass}" in
      eefball)       PORT="${PORT:-8005}" ;;   # dp3_eefball
      tight)         PORT="${PORT:-8003}" ;;   # dp3_tight1024
      dual)          PORT="${PORT:-8011}" ;;   # dp3_dual (transplant: PORT=8012)
      dual_rgb)      PORT="${PORT:-8010}" ;;   # dp3c_dual
      eefframe)      PORT="${PORT:-8014}" ;;   # dp3_eefframe
      full_eefframe) PORT="${PORT:-8013}" ;;   # dp3_full_eefframe
      *)             PORT="${PORT:-8000}" ;;   # legacy glass; hglass = PORT=8004
    esac ;;
  *)                 PORT="${PORT:-8000}" ;;
esac
OBS_MODE="${OBS_MODE:-pcd}"            # pcd | depth | rgb_lut | rgb_lut_depth
FREQ="${FREQ:-20}"
EXEC_STEPS="${EXEC_STEPS:-6}"
# ZOOM_CAPTURE=1: dense raw dump for neutralization-zoom figures (~1 infer / 0.2 s
# at EXEC_STEPS=2). Override EXEC_STEPS explicitly if you want a different rate.
if [[ "${ZOOM_CAPTURE:-0}" == "1" && "${EXEC_STEPS}" == "6" ]]; then
  EXEC_STEPS=2
  echo "[run] ZOOM_CAPTURE: default EXEC_STEPS=2 (override with EXEC_STEPS=1 for max density)"
fi
IMG_SIZE="${IMG_SIZE:-224}"
MAX_STEPS="${MAX_STEPS:-0}"
SAFE_MODE="${SAFE_MODE:-1}"
SHOW_CAMERA="${SHOW_CAMERA:-1}"
FREEZE_HEAD="${FREEZE_HEAD:-0}"
ZERO_OBS="${ZERO_OBS:-0}"
PCD_NUM_POINTS="${PCD_NUM_POINTS:-0}"  # 0 = auto (metadata/dataset), else 1024/2048
# Which DP3 cloud build to reproduce. MUST match the checkpoint:
#   glass -> dp3_pcd1024_glass / dp3_pcd2048_glass  (Z-slab, random subsample)
#   tight -> dp3_tight1024 / dp3_tight2048          (range crop, stride, fallback)
# A mismatch is silent — both produce a valid (N,3) cloud the server accepts.
PCD_RECIPE="${PCD_RECIPE:-glass}"
# dual_rgb local-ball radius (m): basket fleet 1.5, pick-place v8–v11 use 1.0
PCD_BALL_RADIUS="${PCD_BALL_RADIUS:-1.5}"
# pp_v9_left only: emit front_pcd_3 around the left eef
PCD_LEFT_STREAM="${PCD_LEFT_STREAM:-0}"
# 1 = soft reset ALSO drives the base to the dataset start (default 0: base never moves on y — 2026-09-16)
RESET_BASE="${RESET_BASE:-0}"
# Right XHand USB-serial (FTDI). Falls back to ttyUSB1 if ttyUSB0 is absent
# (common after a hub re-enumeration / second FTDI disconnect).
HAND_SERIAL_RIGHT="${HAND_SERIAL_RIGHT:-/dev/ttyUSB0}"
if [[ ! -e "$HAND_SERIAL_RIGHT" && -e /dev/ttyUSB1 ]]; then
  echo "[run] WARNING: ${HAND_SERIAL_RIGHT} missing; using /dev/ttyUSB1 for right hand"
  HAND_SERIAL_RIGHT=/dev/ttyUSB1
fi
# Left XHand EtherCAT interface (2026-11-07). Default unset = the client's default (enx4cd717ad4a22, the usual adapter).
# Another USB-Ethernet adapter gets another name: HAND_ETH_LEFT=enx0050b6e40b2d (ASIX AX88179). Check with: ip -br link | grep enx
HAND_ETH_LEFT="${HAND_ETH_LEFT:-}"
if [[ -n "$HAND_ETH_LEFT" ]]; then
  ip link show "$HAND_ETH_LEFT" >/dev/null 2>&1 || echo "[run] WARNING: left hand interface '${HAND_ETH_LEFT}' not found (ip -br link | grep enx)"
  echo "[run] left hand EtherCAT on ${HAND_ETH_LEFT}"
fi
GT_MODE="${GT_MODE:-}"                 # ""=live | gt_image | gt_proprio | gt_proprio_live_image | gt_action

# Dataset for GT modes + dataset_avg soft reset. Default matches the recommended
# DP3 1024 checkpoint's training set; switch for Adapt3R (slamrect_rgbd).
EGOVERSE_DS="/home/aloha/RB_Y1_workspace/EgoVerse/datasets"
case "$OBS_MODE" in
  # h_rect / a3r_human are HUMAN-corpus policies: the reset pose must come
  # from the corpus they were trained on, not from a teleop set.
  rgb_lut_depth_eef)     DATASET_DEFAULT="${EGOVERSE_DS}/human_fullpp_rgbd_eef" ;;
  rgb_lut|rgb_lut_depth) DATASET_DEFAULT="${EGOVERSE_DS}/human_fullpp_rgbd" ;;
  depth)                 DATASET_DEFAULT="${EGOVERSE_DS}/rby1_teleop_slamrect_rgbd" ;;
  pcd)
    # The pcd dataset must match the recipe: its clouds set the reset pose and
    # are what --gt-image replays, so a glass dataset with a tight checkpoint
    # would validate the wrong contract.
    case "$PCD_RECIPE" in
      tight)         DATASET_DEFAULT="${EGOVERSE_DS}/rby1_teleop_pcd1024_tight" ;;
      eefball)       DATASET_DEFAULT="${EGOVERSE_DS}/human_dp3_eefball" ;;
      dual)          DATASET_DEFAULT="${EGOVERSE_DS}/human_dp3_dual" ;;
      dual_rgb)      DATASET_DEFAULT="${EGOVERSE_DS}/human_dp3c_dual" ;;
      eefframe)      DATASET_DEFAULT="${EGOVERSE_DS}/human_dp3_eefframe" ;;
      full_eefframe) DATASET_DEFAULT="${EGOVERSE_DS}/human_dp3_full_eefframe" ;;
      *)             DATASET_DEFAULT="${EGOVERSE_DS}/rby1_teleop_pcd1024_glass" ;;
    esac ;;
  *)                     DATASET_DEFAULT="${EGOVERSE_DS}/rby1_teleop_pcd1024_glass" ;;
esac
# abl0921: all 26 are basket-task dp3c policies — reset from the basket set.
[[ -n "$ABL" ]] && DATASET_DEFAULT="${EGOVERSE_DS}/dp3c_basket60_v2_slim"
DATASET="${DATASET:-$DATASET_DEFAULT}"
DEMO_NAME="${DEMO_NAME:-0}"
RESET_CACHE="${RESET_CACHE:-${DATASET:+${DATASET%/}/reset_pose_avg_cache.npz}}"
if [[ -n "$DATASET" && -d "$DATASET" ]]; then
  RESET_MODE="${RESET_MODE:-dataset_avg}"
else
  echo "[run] WARNING: dataset '$DATASET' not found — pull it first (pull_dp3_adapt3r.sh)."
  DATASET=""
  RESET_MODE="${RESET_MODE:-demo}"
fi

COMMON=(
  "${PY}" projects/rby1_teleop/rollout_rby1_policy_real_bgobs.py
  --host "$HOST" --port "$PORT" --frequency "$FREQ"
  --action-map-preset joint_base_torso_head_arm_hand
  --action-key joint_base_torso_head_arm_hand
  --robot0-joint-pos-len 22
  --policy-image-size "$IMG_SIZE"
  --execute-steps-per-infer "$EXEC_STEPS"
  --max-steps "$MAX_STEPS"
  --reset-pose-mode "$RESET_MODE"
  --obs-mode "$OBS_MODE"
  --pcd-num-points "$PCD_NUM_POINTS"
  --pcd-recipe "$PCD_RECIPE"
  --pcd-ball-radius "$PCD_BALL_RADIUS"
  --hand_serial_right "$HAND_SERIAL_RIGHT"
)
[[ -n "$HAND_ETH_LEFT" ]] && COMMON+=(--hand_eth_left "$HAND_ETH_LEFT")
# 2026-10-29: GT_MODE=gt_proprio / gt_action replay the dataset (the client reads nothing from the glasses), so skip the
# Aria publisher, its sudo modprobe and the glasses bring-up there. Every other mode streams as before; ARIA_STREAMING=1 forces it.
if [[ "${ARIA_STREAMING:-0}" == "1" || ( "$GT_MODE" != "gt_proprio" && "$GT_MODE" != "gt_action" ) ]]; then
  COMMON+=(--aria-streaming)
fi
[[ "$PCD_LEFT_STREAM" == "1" ]] && COMMON+=(--pcd-left-stream)
[[ "$RESET_BASE" == "1" ]] && COMMON+=(--reset-base)
[[ -n "$ABL" ]] && COMMON+=(--abl-variant "$ABL")
[[ "${ASYNC:-0}" == "1" ]] && COMMON+=(--async-infer --async-lookahead "${ASYNC_LOOKAHEAD:-8}")
# RTC=1: Real-Time Chunking (needs RTC=1 serve_abl_0921.sh on the server side
# and SAFE_MODE=0). Knobs: RTC_FREEZE (d, default 3), RTC_PERIOD (K, 10),
# RTC_SOFT (E, 19), RTC_SOFT_POWER (gamma, 1.0).
if [[ "${RTC:-0}" == "1" ]]; then
  COMMON+=(--rtc --rtc-freeze-steps "${RTC_FREEZE:-3}" --rtc-period "${RTC_PERIOD:-10}"
           --rtc-soft-horizon "${RTC_SOFT:-19}" --rtc-soft-power "${RTC_SOFT_POWER:-1.0}")
  if [[ "${SAFE_MODE}" == "1" ]]; then
    echo "[run] RTC=1 forces SAFE_MODE=0 (continuous replanning)"; SAFE_MODE=0
  fi
fi
[[ -n "${LAYOUT_TAG:-}" ]] && COMMON+=(--layout-tag "$LAYOUT_TAG")
# GZ only — GAZE_ENS=1: average the 6-member robust gaze ensemble instead of the
# single pulled predictor (gaze_lab 2026-10-04: 4.99 deg / 9.8 cm vs 5.4 / 11,
# and 5.2 vs 7.3 deg under 5 px blur), with --gaze-median 3 (the 5-frame default
# costs 0.9 deg through lag). Weights: GAZE_ENS_DIR. Default off = unchanged.
GAZE_ENS="${GAZE_ENS:-0}"
GAZE_ENS_DIR="${GAZE_ENS_DIR:-/home/aloha/RB_Y1_workspace/EgoVerse/checkpoints/abl0921/gaze/ens}"
GAZE_ENS_NAME="${GAZE_ENS_NAME:-robust}"   # TGZ: GAZE_ENS_NAME=fridge -> gaze_heat7_fridge_m{0..5}.pt
if [[ "$GAZE_ENS" == "1" ]]; then
  GAZE_LIST=""
  for i in 0 1 2 3 4 5; do
    w="${GAZE_ENS_DIR}/gaze_heat7_${GAZE_ENS_NAME}_m${i}.pt"
    [[ -f "$w" ]] || { echo "[run] ERROR: GAZE_ENS=1 but '$w' is missing" >&2; exit 1; }
    GAZE_LIST="${GAZE_LIST:+${GAZE_LIST},}${w}"
  done
  COMMON+=(--gaze-predictor "$GAZE_LIST" --gaze-median "${GAZE_MEDIAN:-3}")
  echo "[run] GAZE_ENS=1: 6-member ${GAZE_ENS_NAME} gaze ensemble, --gaze-median ${GAZE_MEDIAN:-3}"
elif [[ -n "${GAZE_MEDIAN:-}" ]]; then
  COMMON+=(--gaze-median "$GAZE_MEDIAN")
fi
# TGZ (2026-10-27) — VV_MODEL=<virtual_cam_fridge.npz>: predict the human camera from the robot's joints each frame
# and keep only the cloud points it can see (utils/virtual_view.py). Use with ABL=TGZ GAZE_ENS=1 GAZE_ENS_NAME=fridge
# (GAZE_ENS_DIR = the folder of the fridge gaze ensemble). Default unset = unchanged.
[[ -n "${VV_MODEL:-}" ]] && { [[ -f "$VV_MODEL" ]] || { echo "[run] ERROR: VV_MODEL '$VV_MODEL' is missing" >&2; exit 1; }; COMMON+=(--virtual-view-model "$VV_MODEL"); echo "[run] VV_MODEL=${VV_MODEL}: virtual human camera ON"; }
# Progress policies (2026-10-27) — PROGRESS_MAX_STEP=<limit per executed row of the task-progress memory>. Default unset = the
# client's 0.004 (fridge); the cart demos are shorter: PROGRESS_MAX_STEP=0.0144. Only used when the policy has a `progress` input.
[[ -n "${PROGRESS_MAX_STEP:-}" ]] && { COMMON+=(--progress-max-step "$PROGRESS_MAX_STEP"); echo "[run] PROGRESS_MAX_STEP=${PROGRESS_MAX_STEP}"; }
# LIGHTING="afternoon sun, blinds open" -> rollouts.jsonl (0929 lighting experiment)
[[ -n "${LIGHTING:-}" ]] && COMMON+=(--lighting-tag "$LIGHTING")
# CHUNK_FILTER=1: zero-lag within-chunk smoothing (the within-motion jerk fix;
# any mode). 2 = extra pass (smoother than demos).
[[ "${CHUNK_FILTER:-0}" != "0" ]] && COMMON+=(--chunk-filter "${CHUNK_FILTER}")
# STRAY_FILTER=1 (2026-11-01): drop isolated stereo depth points (streaks off the robot's hands) before the
# dual_rgb clouds and the gaze hit; the sim-trained cart policy drove forward ~3x less with them. STRAY_NEAR
# (m, default 2.0) limits it to close points; STRAY_NEAR=0 = whole grid (egoverse-29's rule). ~60 ms per frame.
[[ "${STRAY_FILTER:-0}" == "1" ]] && { COMMON+=(--depth-stray-filter --depth-stray-near "${STRAY_NEAR:-2.0}"); echo "[run] STRAY_FILTER near ${STRAY_NEAR:-2.0} m"; }
# BASE_DELTA_FRAME=odom restores the pre-2026-10-13 behaviour (policy base deltas added in the
# raw odometry frame). Default (start) rotates them by the start heading, matching the dataset
# convention and the base-pose observation — see rollout --base-delta-frame.
[[ -n "${BASE_DELTA_FRAME:-}" ]] && COMMON+=(--base-delta-frame "$BASE_DELTA_FRAME")
# 2026-11-04 base/arm sync, both off by default. The arm stops while each obs is built (~0.8 s, no rows sent) but
# the base keeps chasing its goal, 2-7 cm behind. BASE_PAUSE_AT_OBS=1 stops the base too, keeping the distance owed;
# BASE_FF=<gain> (try 1.0) adds velocity feed-forward so the base trails less. Standard client only.
[[ "${BASE_PAUSE_AT_OBS:-0}" == "1" ]] && COMMON+=(--base-pause-at-obs)
[[ -n "${BASE_FF:-}" ]] && COMMON+=(--base-ff-gain "$BASE_FF")
# BASE_KP_SCALE=<x> multiplies the follower's x/y KP (yaw untouched); KP sets the settling time, the speed and accel limits don't.
[[ -n "${BASE_KP_SCALE:-}" ]] && COMMON+=(--base-kp-scale "$BASE_KP_SCALE")
# FAST_OBS=1 (2026-11-16): build the dual_rgb obs ~3x faster with the SAME clouds bit for bit (batched hand-dense FPS,
# no thrown-away plain FPS, 4-thread stray filter). No rows go out while the obs is built, so this shortens the pause
# at every query. Default off = unchanged.
[[ "${FAST_OBS:-0}" == "1" ]] && { COMMON+=(--fast-obs); echo "[run] FAST_OBS=1: batched hand-dense FPS + 4-thread stray filter"; }
# BG_OBS=1 (default in this launcher): build the RTC/async prefetch obs in the worker thread, so rows keep
# executing while it is built (rollout_rby1_policy_real_bgobs.py --bg-obs). BG_OBS=0 = original behaviour.
[[ "${BG_OBS:-1}" == "1" ]] && { COMMON+=(--bg-obs); echo "[run] BG_OBS=1: prefetch obs built in the worker thread (rows keep running)"; }
# RECORD_SLAM_PAIR=1 (2026-11-17): with RECORD_RAW_DIR, also dump the rectified grey SLAM pair Fast-FS saw (~0.5 MB/query).
if [[ "${RECORD_SLAM_PAIR:-0}" == "1" ]]; then
  [[ -n "${RECORD_RAW_DIR:-}" ]] || { echo "[run] ERROR: RECORD_SLAM_PAIR=1 needs RECORD_RAW_DIR" >&2; exit 1; }
  COMMON+=(--record-slam-pair); echo "[run] RECORD_SLAM_PAIR=1: rectified SLAM pair saved in the raw dump"
fi
[[ "$OBS_MODE" == "pcd" ]] && echo "[run] cloud recipe: ${PCD_RECIPE}"
[[ "$SHOW_CAMERA" == "1" ]] && COMMON+=(--show-camera)
# VIZ_CHUNK=0 drops the per-splice full-chunk FK preview + viewer bundle draw
# (~0.1 s of main-thread work per replan at 20 Hz RTC). Default on.
[[ "${VIZ_CHUNK:-1}" == "1" ]] && COMMON+=(--visualize-eef-chunk)
[[ "$SAFE_MODE"   == "1" ]] && COMMON+=(--safe-mode)
[[ "$FREEZE_HEAD" == "1" ]] && COMMON+=(--freeze-head)
[[ "$ZERO_OBS"    == "1" ]] && COMMON+=(--zero-obs)
REC_COMPARE_DIR="${REC_COMPARE_DIR:-}"
[[ -n "$REC_COMPARE_DIR" ]] && COMMON+=(--record-image-compare-dir "$REC_COMPARE_DIR")
# RECORD_RAW_DIR=<dir> dumps depth512 + raw RGB + pose + policy tensors per
# inference (figure captures). ~2.5 MB per inference — short runs only.
RECORD_RAW_DIR="${RECORD_RAW_DIR:-}"
[[ -n "$RECORD_RAW_DIR" ]] && COMMON+=(--record-raw-dir "$RECORD_RAW_DIR")
# BASE_DIAG=1 logs base provenance + SDK velocity + odometry without drawing in control.
if [[ "${BASE_DIAG:-0}" == "1" || -n "${BASE_DIAG_DIR:-}" ]]; then
  BASE_DIAG_DIR="${BASE_DIAG_DIR:-${RECORD_RAW_DIR:-/home/aloha/RB_Y1_workspace/rollout_logs/base_$(date +%Y%m%d_%H%M%S)}/base_diag}"
  COMMON+=(--base-diag-dir "$BASE_DIAG_DIR")
  echo "[base-diag] separate terminal: $PY projects/rby1_teleop/utils/plot_base_diagnostics.py '$BASE_DIAG_DIR' --watch"
fi
if [[ -n "$DATASET" ]]; then
  COMMON+=(--dataset "$DATASET" --demo-name "$DEMO_NAME")
  [[ "$RESET_MODE" == "dataset_avg" ]] && COMMON+=(--reset-pose-cache "$RESET_CACHE")
  echo "[run] reset: ${RESET_MODE} from dataset ${DATASET}"
else
  echo "[run] reset: ${RESET_MODE} (no dataset; starts from current robot pose)"
fi

case "$GT_MODE" in
  gt_image)
    [[ -z "$DATASET" ]] && { echo "[run] ERROR: GT_MODE=gt_image needs DATASET" >&2; exit 1; }
    COMMON+=(--gt-image)
    echo "[run] GT mode: LIVE proprio + dataset GT ${OBS_MODE} obs (server ON)"
    ;;
  gt_proprio)
    [[ -z "$DATASET" ]] && { echo "[run] ERROR: GT_MODE=gt_proprio needs DATASET" >&2; exit 1; }
    COMMON+=(--use-gt-proprio)
    echo "[run] GT mode: dataset GT ${OBS_MODE} obs + proprio (server ON)"
    ;;
  gt_proprio_live_image)
    [[ -z "$DATASET" ]] && { echo "[run] ERROR: GT_MODE=gt_proprio_live_image needs DATASET" >&2; exit 1; }
    COMMON+=(--use-gt-proprio --gt-proprio-live-image)
    echo "[run] GT mode: dataset GT proprio + LIVE ${OBS_MODE} perception (server ON)"
    ;;
  gt_action)
    [[ -z "$DATASET" ]] && { echo "[run] ERROR: GT_MODE=gt_action needs DATASET" >&2; exit 1; }
    COMMON+=(--use-gt-action)
    echo "[run] GT mode: replay dataset actions (no policy/server)"
    ;;
  "" )
    echo "[run] LIVE mode: Fast-FS ${OBS_MODE} obs + hardware proprio (server must be running)"
    ;;
  *) echo "[run] ERROR: unknown GT_MODE='$GT_MODE'" >&2; exit 1;;
esac

# Fail fast: check the policy server BEFORE the ~40 s robot/Aria/hand bring-up
# (a refused connect after that leaves hands/shm to clean up). Waits up to
# WAIT_SERVER_S for a server that is still loading its checkpoint.
if [[ "$GT_MODE" != "gt_action" && "$PY" != "/bin/echo" ]]; then
  WAIT_SERVER_S="${WAIT_SERVER_S:-90}"
  waited=0
  # Local server: look for the LISTENING socket (no connection is made — a bare
  # TCP probe made the websockets server log "did not receive a valid HTTP
  # request" on every launch). Remote host: fall back to a TCP connect.
  _server_up() {
    if [[ "$HOST" == "127.0.0.1" || "$HOST" == "localhost" ]]; then
      ss -ltnH 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p"$" {f=1} END {exit !f}'
    else
      (exec 3<>"/dev/tcp/${HOST}/${PORT}") 2>/dev/null
    fi
  }
  until _server_up; do
    if (( waited == 0 )); then
      echo "[run] policy server ws://${HOST}:${PORT} not up yet — waiting up to ${WAIT_SERVER_S}s"
      echo "[run]   (start it: cd ~/RB_Y1_workspace/EgoVerse && $([[ "${RTC:-0}" == "1" ]] && echo 'RTC=1 ')bash serve_abl_0921.sh ${ABL:-<variant>})"
    fi
    if (( waited >= WAIT_SERVER_S )); then
      echo "[run] ERROR: no policy server on ${HOST}:${PORT} after ${WAIT_SERVER_S}s — aborting BEFORE hardware bring-up." >&2
      exit 1
    fi
    sleep 3; waited=$((waited + 3))
  done
  (( waited > 0 )) && echo "[run] policy server is up (after ${waited}s)."
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[run] OPEN-LOOP preview (no hardware motion). MuJoCo overlay + perception window."
  exec "${COMMON[@]}" --not_use_hardware
else
  echo "[run] CLOSED-LOOP on hardware. obs=${OBS_MODE} ws://${HOST}:${PORT} safe-mode=${SAFE_MODE}. Ctrl-C / e-stop ready."
  exec "${COMMON[@]}"
fi
