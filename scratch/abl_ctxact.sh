#!/bin/bash
set -u
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
export PYTHONPATH=. SDL_VIDEODRIVER=dummy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SRUN=/opt/slurm/Ubuntu-20.04/current/bin/srun
CKPT=logs/pixelpol_A/A_diffuse_v2_2026-05-31_17-27-49/checkpoints/last.ckpt
CFG=logs/pixelpol_A/A_diffuse_v2_2026-05-31_17-27-49/.hydra/config.yaml
JID=$1
for MODE in gt prev zero; do
  $SRUN --jobid=$JID --overlap .venv/bin/python scripts/tf_chunk_eval.py \
    --ckpt $CKPT --config-path $CFG --n-ctx 1 --k 8 --max-eps 2 --max-frames 40 \
    --ctx-act-mode $MODE > /tmp/o_abl_$MODE.txt 2>&1
  echo "ABL_${MODE}_DONE rc=$? :: $(grep -aE 'chunk_mse_OVERALL|chunk_mse_step_00' /tmp/o_abl_$MODE.txt | tr '\n' ' ')"
done
echo ALL_ABL_DONE
