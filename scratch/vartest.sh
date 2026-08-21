#!/bin/bash
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
export PYTHONPATH=. SDL_VIDEODRIVER=dummy
CKPT=logs/pixeldec/DEC_v1_2026-06-01_00-11-19/checkpoints/last.ckpt
CFG=logs/pixeldec/DEC_v1_2026-06-01_00-11-19/.hydra/config.yaml
JID=${1:-3283665}
for K in 1 8; do
  echo "NSAMP=$K"
  /opt/slurm/Ubuntu-20.04/current/bin/srun --jobid=$JID --overlap .venv/bin/python scripts/tf_decoupled_eval.py \
    --ckpt $CKPT --config-path $CFG --n-ctx 4 --max-eps 3 --max-frames 60 --n-steps 50 --n-samples $K 2>&1 \
    | grep -aE "clean_obs2action|Error|Traceback" | tail -2
done
echo VARTEST_DONE
