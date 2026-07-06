#!/bin/bash
# Retrain the 2D policy with action_loss_weight=5 (was 1.0). The verification
# showed the action token nails the immediate 1-2 steps but the mid-chunk
# diverges -> the single action token's loss is swamped by T*P latent tokens at
# weight 1.0. Upweighting should pull more gradient onto the action head.
# Self-contained: train -> copy ckpt -> verify action MSE.
set -u
export COLUMNS=250
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
JOB=3281903
DATA=/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/external_ckpts/pushshapes_circle_750_norm_stats.json

srun --jobid=$JOB --overlap .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=oa_policy2d_alw5 description=alw5 mode=train \
  data=tsimulation_full \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  model=dfot_pushshapes_image_spatial_policy \
  model.robomimic_model.outer_stage.action_loss_weight=5.0 \
  model.scheduler.max_steps=19200 \
  evaluator=eval_dfot_obs_action_image \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=30 \
  trainer=debug trainer.max_epochs=120 trainer.min_epochs=120 \
  trainer.limit_train_batches=160 \
  data.train_dataloader_params.pushshapes_sim.batch_size=2 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=2 \
  trainer.limit_val_batches=1 trainer.check_val_every_n_epoch=200 \
  trainer.precision=16-mixed +trainer.gradient_clip_val=1.0 \
  norm_stats.precomputed_norm_path=$NORM \
  logger=csv_wandb > /tmp/o_train_alw5.txt 2>&1
echo "TRAIN_DONE rc=$?  $(tail -2 /tmp/o_train_alw5.txt)"
RUN=$(ls -dt logs/oa_policy2d_alw5/* | head -1)
cp $RUN/checkpoints/last.ckpt external_ckpts/policy2d_alw5_last.ckpt && echo COPIED
ls -la external_ckpts/policy2d_alw5_last.ckpt

srun --jobid=$JOB --overlap .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian mode=eval \
  name=v2 description=verify_alw5 ckpt_path=external_ckpts/policy2d_alw5_last.ckpt \
  data=tsimulation_full \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  model=dfot_pushshapes_image_spatial_policy evaluator=eval_dfot_image_spatial_policy \
  trainer=debug trainer.limit_val_batches=1 \
  norm_stats.precomputed_norm_path=$NORM logger=csv > /tmp/o_verify_alw5.txt 2>&1
echo "############ alw5 action MSE (vs baseline first20=0.892, step00=0.046, step01=0.0016) ############"
grep -oE "action_mse_(first20|step_0[0-9]) +[│|] +[0-9.eE+-]+" /tmp/o_verify_alw5.txt | sed -E 's/[│|]/ /g; s/  +/ /g' | sort -u
