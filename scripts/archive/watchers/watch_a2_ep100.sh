#!/bin/bash
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
JOB=3308048
D=""
for i in $(seq 1 40); do
  D=$(ls -td logs/hnet_baseline_nc3/bcchunk_gmm_maxstd1_* 2>/dev/null | head -1)
  [ -n "$D" ] && break
  if ! squeue -j $JOB -h 2>/dev/null | grep -q .; then echo "JOB_GONE_NO_DIR"; exit 0; fi
  sleep 20
done
CKDIR=$D/checkpoints
echo "A2_DIR=$D"
while [ ! -e "$CKDIR/epoch_epoch=99.ckpt" ]; do
  if ! squeue -j $JOB -h 2>/dev/null | grep -q .; then echo "JOB_ENDED_BEFORE_EP100"; break; fi
  sleep 60
done
echo "EP100_READY"; ls "$CKDIR" 2>/dev/null
JID=$(sbatch --parsable --export=ALL,TF_DUMP_DIR=logs/tf_dump_a2_ep100,CKPT=$CKDIR/last.ckpt scripts/eval_bcgmm_tfdump.sbatch)
echo "DUMP_JID=$JID"
while squeue -j $JID -h 2>/dev/null | grep -q .; do sleep 20; done
echo "DUMP_DONE"
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
python scripts/plot_bcgmm_tfdump.py logs/tf_dump_a2_ep100 a2-ep100-maxstd1 2>&1 | tail -12
