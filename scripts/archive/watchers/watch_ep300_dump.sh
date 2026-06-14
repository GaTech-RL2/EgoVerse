#!/bin/bash
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
CKDIR=logs/hnet_baseline_nc3/bcchunk_gmm_resnet_2026-06-04_12-50-56/checkpoints
JOB=3307644
# wait for the ep-300 checkpoint (epoch index 299) or the job to end
while [ ! -e "$CKDIR/epoch_epoch=299.ckpt" ]; do
  if ! squeue -j $JOB -h 2>/dev/null | grep -q .; then echo "JOB_ENDED_BEFORE_EP300"; break; fi
  sleep 60
done
echo "EP300_CHECK_DONE; ckpts present:"; ls "$CKDIR" 2>/dev/null
# dump on the latest checkpoint (last.ckpt avoids the epoch= Hydra bug)
JID=$(sbatch --parsable --export=ALL,TF_DUMP_DIR=logs/tf_dump_bcgmm_ep300,CKPT=$CKDIR/last.ckpt scripts/eval_bcgmm_tfdump.sbatch)
echo "DUMP_JID=$JID"
while squeue -j $JID -h 2>/dev/null | grep -q .; do sleep 20; done
echo "DUMP_DONE"
of=$(ls -t logs/sbatch/*${JID}*.out 2>/dev/null | head -1)
grep -nE "EVAL_EXIT|Error|Traceback|size mismatch" "$of" 2>/dev/null | tail -8
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
python scripts/plot_bcgmm_tfdump.py logs/tf_dump_bcgmm_ep300 epoch-300 2>&1 | tail -12
