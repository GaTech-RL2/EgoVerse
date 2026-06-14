#!/bin/bash
#SBATCH --job-name=chunkA_currentframe
#SBATCH --account=overcap
#SBATCH --partition=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --requeue
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/chunkA_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/chunkA_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=online
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json
EPOCHS=${EPOCHS:-600}
VALEVERY=${VALEVERY:-200}
DESC=${DESC:-chunkA_currentframe}
# chunk-A CURRENT-FRAME: reactive HNetFused chunk model (ResNet), chunk_k=32.
# reactive=true makes BOTH the packed training forward AND policy.generate use
# the current frame only -- each timestep is an independent [c_t, BOS] pair, head
# at pos 1 -> a K=32 chunk. This makes training CONSISTENT with the chunk_openloop
# rollout (which calls generate's chunk_k>1 branch: same [c_t,BOS]@0/1, head@1).
srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hnet_baseline_nc3 description=${DESC} mode=train data=tsimulation \
  model=hnet_pushshapes_fused_pusher_resnet \
  ++model.robomimic_model.reactive=true \
  ++model.robomimic_model.chunk_k=32 \
  ++model.robomimic_model.action_horizon=1024 \
  ++model.robomimic_model.cond_encoder.img_encoders.front_img_1.spatial=true \
  evaluator=eval_hnet_sim \
  evaluator.rollout_mode=chunk_openloop evaluator.chunk_k=32 \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.devices=1 trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=8 trainer.limit_val_batches=4 trainer.check_val_every_n_epoch=${VALEVERY} \
  trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
  norm_stats.precomputed_norm_path=$NORM
echo "TRAIN_EXIT=$?"
