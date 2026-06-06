#!/bin/bash
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
SALLOC=/opt/slurm/Ubuntu-20.04/current/bin/salloc
JID=$($SALLOC -A rl2-lab -p rl2-lab --time=1:00:00 --gres=gpu:a40:1 --cpus-per-task=8 --mem=48G --no-shell 2>&1 | grep -oE "allocation [0-9]+" | grep -oE "[0-9]+")
echo "ENDGAME_ALLOC=$JID"
echo "=== CLEAN obs->action MSE + train loss ==="
bash reeval_dec.sh "$JID"
echo "=== SIM coverage ==="
bash eval_DEC.sh "$JID"
echo ENDGAME_DONE
