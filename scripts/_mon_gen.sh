#!/bin/bash
# Monitor the two +gen runs (HPT 3340288, TX 3340289): queue state, training-start, errors, epoch.
BASE=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch
H=3340617; T=3340611
HLOG=$BASE/hptgen_hptFlowC3000Act8PlusGen_$H.out
TLOG=$BASE/bcrnn_bcrnnTxC8FHPlusGen_$T.out
hs=$(squeue -j $H -h -o %T 2>/dev/null); ts=$(squeue -j $T -h -o %T 2>/dev/null)
ht=$(grep -c "Starting training" "$HLOG" 2>/dev/null); tt=$(grep -c "Starting training" "$TLOG" 2>/dev/null)
he=$(grep -cE "Traceback|Error" "$HLOG" 2>/dev/null); te=$(grep -cE "Traceback|Error" "$TLOG" 2>/dev/null)
df=$(df -P /coc/flash7 | tail -1 | tr -s ' ' | cut -d' ' -f4)
echo "HPT[${hs:-NOQ} start=${ht:-0} err=${he:-0}] TX[${ts:-NOQ} start=${tt:-0} err=${te:-0}] free=${df}"
