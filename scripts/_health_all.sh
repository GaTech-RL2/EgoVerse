#!/bin/bash
# Health probe for all active training runs.
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2 || exit 1
NOW=$(date +%s)
echo "NOW $(date '+%m-%d %H:%M:%S')"
echo "--- my jobs in queue ---"
squeue --me -o "%.10i %.28j %.9T %.10M %R" 2>/dev/null | grep -iE "JOBID|CotrainBigSmall|PlusGen|hptCotrain"
echo
for run in \
  "3340782:bcrnnTxC8FHCotrainBigSmall:bcrnn_bcrnnTxC8FHCotrainBigSmall:TX cotrain big+small" \
  "3340617:hptFlowC3000Act8PlusGen:hptgen_hptFlowC3000Act8PlusGen:HPT +gen" \
  "3340611:bcrnnTxC8FHPlusGen:bcrnn_bcrnnTxC8FHPlusGen:TX +gen" ; do
  J=${run%%:*}; rest=${run#*:}; NAME=${rest%%:*}; rest2=${rest#*:}; LP=${rest2%%:*}; DESC=${rest2#*:}
  echo "=== $DESC ($J) ==="
  ST=$(squeue -j "$J" -h -o "%T %M %R" 2>/dev/null)
  echo "  queue: ${ST:-NOT-IN-QUEUE}"
  EP=$(ls logs/$NAME/*/checkpoints/epoch*.ckpt 2>/dev/null | grep -oE "epoch=[0-9]+" | grep -oE "[0-9]+" | sort -n | tail -1)
  echo "  ckptEp: ${EP:-none}"
  LAST=$(ls -t logs/$NAME/*/checkpoints/last.ckpt 2>/dev/null | head -1)
  if [ -n "$LAST" ]; then AGE=$(( (NOW - $(stat -c %Y "$LAST")) / 60 )); echo "  last.ckpt age: ${AGE} min"; else echo "  last.ckpt: none yet"; fi
  L=$(ls -t logs/sbatch/${LP}_${J}.out 2>/dev/null | head -1)
  TS=$(grep -c "Starting training" "$L" 2>/dev/null)
  ER=$(grep -cE "Traceback|Error|oom-kill|OutOfMemory|CANCELLED" "$L" 2>/dev/null)
  echo "  trainStarted=$TS  errorHits=$ER"
done
echo
echo "--- big+small HPT 3339901 (should be DONE/converged) ---"
sacct -j 3339901 -X -o State,Elapsed 2>/dev/null | head -2
echo "--- disk ---"; df -h /coc/flash7 | tail -1
