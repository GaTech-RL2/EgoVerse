#!/bin/bash
# Remote monitor helper for the HPT big+small act8 cotrain run.
# Liveness via squeue (reliable); progress via newest epoch ckpt + last.ckpt age.
JOBID=3339901
BASE=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptCotrainBigSmallAct8
QST=$(squeue -j $JOBID -h -o '%T' 2>/dev/null | head -1)
EP=$(ls $BASE/*/checkpoints/epoch*epoch=*.ckpt 2>/dev/null | grep -oE 'epoch=[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)
LASTMT=$(stat -c %Y $BASE/*/checkpoints/last.ckpt 2>/dev/null | sort -n | tail -1)
NOW=$(date +%s)
if [ -n "$LASTMT" ]; then AGE=$(( (NOW - LASTMT) / 60 )); else AGE=-1; fi
DF=$(df -P /coc/flash7 | tail -1 | tr -s ' ' | cut -d' ' -f4)
echo "queue=${QST:-NOTINQ} ckptEp=${EP:-none} lastCkptAgeMin=${AGE} free=${DF}"
