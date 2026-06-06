#!/bin/bash
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
export PYTHONPATH=. SDL_VIDEODRIVER=dummy
RUN=$(ls -dt logs/pixeldec/DEC_* 2>/dev/null | head -1)
CKPT=$RUN/checkpoints/last.ckpt
CFG=$RUN/.hydra/config.yaml
M=$(find "$RUN" -name metrics.csv 2>/dev/null | head -1)
echo "=== latest train action_loss ==="
.venv/bin/python show_loss.py "$M" 2>/dev/null | tail -2
echo "=== clean obs->action MSE (n_ctx=4) ==="
JID=${1:-3283426}
/opt/slurm/Ubuntu-20.04/current/bin/srun --jobid=$JID --overlap .venv/bin/python scripts/tf_decoupled_eval.py \
  --ckpt "$CKPT" --config-path "$CFG" --n-ctx 4 --max-eps 3 --max-frames 60 2>&1 \
  | grep -aE "clean_obs2action|Error|Traceback" | tail -3
