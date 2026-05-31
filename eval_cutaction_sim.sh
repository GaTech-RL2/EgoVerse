#!/bin/bash
set -u
export COLUMNS=250 PYTHONPATH=. SDL_VIDEODRIVER=dummy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
CKPT=${CKPT:-external_ckpts/cutaction_v1_last.ckpt}
NORM=${NORM:?set NORM}
DATA=/coc/flash7/paphiwetsa3/datasets/new_circle_3_clean
JID=$(salloc -A rl2-lab -p rl2-lab --time=2:00:00 --gres=gpu:a40:1 --cpus-per-task=8 --mem=48G --no-shell 2>&1 | grep -oP 'allocation \K[0-9]+')
echo alloc=$JID
srun --jobid=$JID --export=ALL .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian mode=eval \
  name=cutaction_eval description=sim ckpt_path=$CKPT \
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
  norm_stats.precomputed_norm_path=$NORM logger=csv > /tmp/o_cuteval.txt 2>&1
echo "EVAL rc=$?"
grep -oE "sim_(coverage|success_rate) +[\xe2\x94\x82|] +[0-9.eE+-]+" /tmp/o_cuteval.txt | sed -E "s/[|]/ /g; s/  +/ /g" | sort -u
grep -iE "error|exception|Traceback|not in struct|NotImplemented" /tmp/o_cuteval.txt | head -5
scancel $JID 2>/dev/null
