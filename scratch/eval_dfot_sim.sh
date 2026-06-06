#!/bin/bash
# DFoT spatial_decoupled sim eval. OUT + CKPT + NORM + TAG parameterized (no clobber).
set -u
export COLUMNS=250 PYTHONPATH=. SDL_VIDEODRIVER=dummy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
CKPT=${CKPT:?set CKPT}; NORM=${NORM:?set NORM}; TAG=${TAG:-x}
OUT=/tmp/o_eval_${TAG}.txt
DATA=/coc/flash7/paphiwetsa3/datasets/new_circle_3_clean
JID=$(salloc -A rl2-lab -p rl2-lab --time=2:00:00 --gres=gpu:a40:1 --cpus-per-task=8 --mem=48G --no-shell 2>&1 | grep -oP 'allocation \K[0-9]+')
echo alloc=$JID tag=$TAG
srun --jobid=$JID --export=ALL .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian mode=eval \
  name=cutaction_eval description=$TAG ckpt_path=$CKPT \
  data=tsimulation_full \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_dataloader_params.pushshapes_sim.batch_size=1 \
  model=dfot_pushshapes_image_spatial_policy \
  +model.robomimic_model.outer_stage.decouple_action_noise=true \
  model.robomimic_model.inference_mode=spatial_decoupled \
  model.robomimic_model.sampler_n_steps=25 \
  evaluator=eval_dfot_sim evaluator.max_steps=600 evaluator.max_videos=12 \
  trainer=debug trainer.limit_val_batches=1 \
  norm_stats.precomputed_norm_path=$NORM logger=csv > $OUT 2>&1
echo "EVAL_${TAG} rc=$?"
grep -oE "sim_(coverage|success_rate)[^0-9]+[0-9.eE+-]+" $OUT | tail -6
grep -iE "error|Traceback|NotImplemented|KeyError" $OUT | head -4
scancel $JID 2>/dev/null
