#!/bin/bash
#SBATCH --job-name=smkE_ar_ca
#SBATCH --time=00:30:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/smkE_ar_ca_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/smkE_ar_ca_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json
MODEL=hnet_pushshapes_fused_ar_crossattn_resnet
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=hnet_baseline_nc3 description=smkE_ar_crossattn mode=train data=tsimulation \
  model=${MODEL} \
  +model.robomimic_model.token_dropout_p=0.0 \
  +model.robomimic_model.train_mode=ar \
  +model.robomimic_model.ss_prob=0.5 \
  ++model.robomimic_model.action_horizon=1024 \
  evaluator=eval_hnet_sim \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=1 \
  trainer=debug trainer.max_epochs=2 trainer.min_epochs=2 \
  trainer.limit_train_batches=4 trainer.limit_val_batches=2 trainer.check_val_every_n_epoch=1 \
  trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  evaluator.max_steps=200 evaluator.coverage_threshold=0.8 \
  norm_stats.precomputed_norm_path=$NORM
echo "SMOKE_EXIT=$?"
