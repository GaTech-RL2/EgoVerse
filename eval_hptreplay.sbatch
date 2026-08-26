#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=ig-88,megazord,deebot,cyborg,hk47,spd-13,sonny,kitt,baymax
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/%x_%j.out
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
COMBO=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowComboEma/hptFlowComboEma_2026-07-18_14-29-22
BASE=/coc/flash7/scratch/paphiwetsa3/archive/EgoVerse2_logs/hptFlowC3000/hpt_flow_circle_3000_2026-06-12_18-39-34
case $RUN in
  base) SRC=$BASE; MODEL=hpt_pushshapes_circle; BAKE=raw;;
  ema)  SRC=$COMBO; MODEL=hpt_bc_flow_pushshapes_ema; BAKE=ema;;
  raw)  SRC=$COMBO; MODEL=hpt_bc_flow_pushshapes_ema; BAKE=raw;;
esac
CKPT_IN=${CKPT_IN:-last.ckpt}; EPTAG=${EPTAG:-}
SNAP=${SRC}/checkpoints/snap_${RUN}${EPTAG}_${MODE}.ckpt
python - <<PY
import torch
ck = torch.load("${SRC}/checkpoints/${CKPT_IN}", map_location="cpu", weights_only=False)
print("TRUE_EPOCH", ck["epoch"])
if "${BAKE}" == "ema":
    sd = ck["state_dict"]
    for k, v in ck["ema_state_dict"].items():
        assert k in sd, k
        sd[k] = v
    print("EMA_BAKED")
ck.pop("ema_state_dict", None)
torch.save(ck, "${SNAP}")
PY
case $MODE in
  ol)  MODEARGS="+evaluator.rollout_mode=chunk_openloop +evaluator.chunk_k=32 +evaluator.temporal_ensemble=false";;
  ol8) MODEARGS="+evaluator.rollout_mode=chunk_openloop +evaluator.chunk_k=8 +evaluator.temporal_ensemble=false";;
  *)   MODEARGS="";;
esac
export ROLLOUT_VIDEO_DIR=${SRC}/eval/video_${RUN}${EPTAG}_${MODE}_40s
mkdir -p "$ROLLOUT_VIDEO_DIR" logs/sbatch
echo GRID_DONE_${RUN}${EPTAG}_${MODE}
