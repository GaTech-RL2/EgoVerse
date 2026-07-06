#!/bin/bash
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
sed -i 's#DEC_v1\*#DEC_*#g' reeval_dec.sh eval_DEC_avg.sh
SALLOC=/opt/slurm/Ubuntu-20.04/current/bin/salloc
JID=$($SALLOC -A rl2-lab -p rl2-lab --time=1:00:00 --gres=gpu:a40:1 --cpus-per-task=8 --mem=48G --no-shell 2>&1 | grep -oE "allocation [0-9]+" | grep -oE "[0-9]+")
echo "MEVAL_ALLOC=$JID"
echo "=== DEC_v2 (~ep250) CLEAN obs->action MSE (1 sample) ==="
bash reeval_dec.sh "$JID"
echo "=== DEC_v2 SIM coverage (batched 8-avg, 300 steps, 2 seeds) ==="
bash eval_DEC_avg.sh "$JID"
echo MEVAL_DONE
