#!/bin/bash
# Wait for the first Design-A checkpoint, then run a bounded closed-loop eval.
set -u
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
export COLUMNS=250 PYTHONPATH=. SDL_VIDEODRIVER=dummy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA=/coc/flash7/paphiwetsa3/datasets/new_circle_3_clean

for i in $(seq 1 45); do
  C=$(ls -t logs/pixelpol_A/*/checkpoints/*.ckpt 2>/dev/null | head -1)
  [ -n "$C" ] && break
  sleep 60
done
CKPT=$(ls -t logs/pixelpol_A/*/checkpoints/*.ckpt 2>/dev/null | head -1)
NORM=$(ls -t logs/pixelpol_A/*/norm_stats/norm_stats.json 2>/dev/null | head -1)
echo "ckpt=$CKPT"
echo "norm=$NORM"
if [ -z "$CKPT" ]; then echo "NO_CKPT_AFTER_45MIN"; exit 1; fi

/opt/slurm/Ubuntu-20.04/current/bin/srun --jobid=3283065 --overlap .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian mode=eval name=pixelpol_A_eval description=ep50 \
  ckpt_path="$CKPT" data=tsimulation_full \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_dataloader_params.pushshapes_sim.batch_size=1 \
  model=dfot/pixel_policy model.robomimic_model.inference_mode=pixel_policy \
  model.robomimic_model.sampler_n_steps=25 \
  ++model.robomimic_model.sp_n_context=4 ++model.robomimic_model.sp_commit=1 \
  evaluator=eval_dfot_sim evaluator.max_steps=300 evaluator.max_videos=2 \
  trainer=debug trainer.limit_val_batches=1 \
  norm_stats.precomputed_norm_path="$NORM" logger=csv > /tmp/o_pixelpol_A_ep50.txt 2>&1
echo "A_EP50_EVAL rc=$?"
grep -iE "emb15_sim_coverage|sim_success" /tmp/o_pixelpol_A_ep50.txt | tail -3
grep -iE "Traceback|Error:" /tmp/o_pixelpol_A_ep50.txt | head -3
