#!/bin/bash
#SBATCH --job-name=chunk-var
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/chunkvar_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/chunkvar_%x_%j.err
set -euxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json
# Knobs.
BACKBONE=${BACKBONE:-hnet_chunked}
READOUT=${READOUT:-mean_pool}
RATIO=${RATIO:-4.0}
EPOCHS=${EPOCHS:-500}
VALEVERY=${VALEVERY:-100}
DESC=${DESC:-chunked_variant}
CKPT=${CKPT:-null}          # set to a last.ckpt path to RESUME (continue training)
MAXSTEPS=$(( EPOCHS * 8 ))
srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hnet_baseline_nc3 description=${DESC} \
  mode=train data=tsimulation \
  ckpt_path=${CKPT} \
  model=hnet_pushshapes_chunktoken_hptfull \
  +model.robomimic_model.token_dropout_p=0.0 \
  ++model.robomimic_model.backbone=${BACKBONE} \
  ++model.robomimic_model.stem=none \
  ++model.robomimic_model.readout=${READOUT} \
  ++model.robomimic_model.chunk_compress_ratio=${RATIO} \
  model.scheduler.eta_min=4e-4 model.scheduler.max_steps=${MAXSTEPS} \
  evaluator=eval_hnet_sim \
  evaluator.rollout_mode=chunk_te evaluator.temporal_ensemble=true \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=8 trainer.limit_val_batches=4 trainer.check_val_every_n_epoch=${VALEVERY} \
  trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  evaluator.max_steps=null evaluator.coverage_threshold=0.8 \
  norm_stats.precomputed_norm_path=$NORM
echo "TRAIN_EXIT=$?"
