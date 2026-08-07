#!/bin/bash
set -u
export COLUMNS=250 PYTHONPATH=. SDL_VIDEODRIVER=dummy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
DATA=/coc/flash7/paphiwetsa3/datasets/new_circle_3_clean
NORM=logs/hpt_ref/hpt_regression_v1_2026-05-31_12-18-06/norm_stats/norm_stats.json
JID=$(salloc -A rl2-lab -p rl2-lab --time=2:00:00 --gres=gpu:a40:1 --cpus-per-task=8 --mem=48G --no-shell 2>&1 | grep -oP 'allocation \K[0-9]+')
echo alloc=$JID
srun --jobid=$JID --export=ALL .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian mode=eval \
  name=hpt_eval description=sim ckpt_path=external_ckpts/hpt_ref_v1_last.ckpt \
  data=tsimulation_hpt \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_dataloader_params.pushshapes_sim.batch_size=32 \
  model=hpt_pushshapes_circle_regression \
  evaluator=eval_hpt_standard evaluator.limit_val_batches=20 \
  trainer=debug trainer.limit_val_batches=1 \
  norm_stats.precomputed_norm_path=$NORM logger=csv > /tmp/o_hpteval.txt 2>&1
echo "HPTEVAL rc=$?"
grep -oE "sim_(coverage|success_rate)[^0-9]+[0-9.eE+-]+" /tmp/o_hpteval.txt | tail -6
grep -iE "error|exception|Traceback|not in struct|NotImplemented|KeyError|inference_step" /tmp/o_hpteval.txt | head -6
scancel $JID 2>/dev/null
