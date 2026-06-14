#!/bin/bash
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
UP=/coc/flash7/paphiwetsa3/datasets/pusht_upstream
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
OUTBASE=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/upstream_eval
mkdir -p "$OUTBASE"

run_one () {
  local TAG=$1 MODEL=$2 RUNDIR=$3 CKPTSRC=$4
  local CKPT="${RUNDIR}/checkpoints/${TAG}_evalckpt.ckpt"
  cp -n "$CKPTSRC" "$CKPT"
  export ROLLOUT_VIDEO_DIR="${OUTBASE}/${TAG}_videos"
  mkdir -p "$ROLLOUT_VIDEO_DIR"
  echo "##### RUNNING $TAG #####"
  python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
    name=upstream_${TAG} description=upstream_replay_cov mode=eval data=tsimulation \
    ckpt_path=${CKPT} \
    model=${MODEL} \
    evaluator=eval_hnet_sim evaluator.rollout_mode=ar \
    evaluator.init_mode=replay evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
    +evaluator.max_videos=2 \
    evaluator.limit_val_batches=7 trainer.limit_val_batches=7 \
    data.valid_dataloader_params.pushshapes_sim.batch_size=4 \
    trainer=debug trainer.precision=32 trainer.profiler=null logger=wandb \
    data.train_datasets.pushshapes_sim.resolver.folder_path=$UP \
    data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
    data.valid_datasets.pushshapes_sim.resolver.folder_path=$UP \
    data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
    norm_stats.norm_mode=minmax \
    norm_stats.precomputed_norm_path=${RUNDIR}/norm_stats/norm_stats.json \
    2>&1 | tee "${OUTBASE}/${TAG}_stdout.log"
  echo "${TAG}_EXIT=${PIPESTATUS[0]}"
}

run_one hnet bc_rnn_pushshapes_paperexact_hnet_chunk8 \
  /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetC8_nc3/bc_rnn_hnet_chunk8_2026-06-06_19-38-16 \
  "/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetC8_nc3/bc_rnn_hnet_chunk8_2026-06-06_19-38-16/checkpoints/epoch_epoch=799.ckpt"

run_one tx bc_rnn_pushshapes_paperexact_tx_chunk8 \
  /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnTxC8_nc3/bc_rnn_tx_chunk8_2026-06-05_23-40-22 \
  "/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnTxC8_nc3/bc_rnn_tx_chunk8_2026-06-05_23-40-22/checkpoints/epoch_epoch=899.ckpt"

echo "===== PARSING COVERAGE ====="
for TAG in hnet tx; do
  L="${OUTBASE}/${TAG}_stdout.log"
  echo "----- $TAG -----"
  python - "$L" <<'PY'
import sys,re
L=sys.argv[1]
covs=[]
for line in open(L,errors="ignore"):
    m=re.search(r"final_cov=([0-9.]+)",line)
    if m: covs.append(float(m.group(1)))
import statistics as st
if covs:
    n=len(covs); mean=sum(covs)/n
    sr=sum(1 for c in covs if c>=0.8)/n
    print(f"n_episodes={n} mean_coverage={mean:.4f} success_rate@0.8={sr:.4f}")
    print("min=%.3f max=%.3f"%(min(covs),max(covs)))
else:
    print("NO final_cov lines found")
# also dump logged metric line if present
for line in open(L,errors="ignore"):
    if "sim_coverage" in line or "sim_success" in line:
        print("METRIC:",line.strip())
PY
done
