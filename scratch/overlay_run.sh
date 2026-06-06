#!/bin/bash
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
export PYTHONPATH=. SDL_VIDEODRIVER=dummy
SALLOC=/opt/slurm/Ubuntu-20.04/current/bin/salloc
JID=$($SALLOC -A rl2-lab -p rl2-lab --time=1:00:00 --gres=gpu:a40:1 --cpus-per-task=8 --mem=48G --no-shell 2>&1 | grep -oE "allocation [0-9]+" | grep -oE "[0-9]+")
echo "OVL_ALLOC=$JID"
CKPT=$(ls -t logs/pixeldec/DEC_*/checkpoints/last.ckpt 2>/dev/null | head -1)
CFG=$(dirname $(dirname "$CKPT"))/.hydra/config.yaml
echo "CKPT=$CKPT"
export OVL_VID=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/dec_overlay.mp4
/opt/slurm/Ubuntu-20.04/current/bin/srun --jobid=$JID --overlap .venv/bin/python -u scripts/tf_dec_overlay.py \
  --ckpt "$CKPT" --config-path "$CFG" --k 8 --n-samples 8 --n-steps 25 --max-eps 8 --max-frames 20
echo "OVL_RC=$?"
echo OVL_DONE
