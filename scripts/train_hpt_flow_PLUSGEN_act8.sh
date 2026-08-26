#!/bin/bash
#SBATCH --job-name=hptFlowC3000Act8PlusGen
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclude=ig-88
#SBATCH --requeue
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/hptgen_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/hptgen_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export WANDB_ENTITY=rl2-group

# +gen arm: HPT-flow chunk-8, SAME recipe as the converged -gen run (hptFlowC3000CausalAct8),
# only the dataset differs -> circle_3000_plus_cotrain240 (circle_3000 + 240 gen episodes).
NAME=hptFlowC3000Act8PlusGen
DS=/coc/flash7/paphiwetsa3/datasets/circle_3000_plus_cotrain240

RESUME=""
LAST=$(ls -t /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/${NAME}/*/checkpoints/last.ckpt 2>/dev/null | head -1)
[ -n "$LAST" ] && RESUME="ckpt_path=$LAST"

srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=hpt_flow_circle_3000_PLUSGEN_causal_obs1_act8 mode=train \
  data=tsimulation_hpt_causal model=hpt_pushshapes_circle \
  ++model.robomimic_model.head_specs.pushshapes_sim.action_horizon=8 \
  ++model.robomimic_model.head_specs.pushshapes_sim.model.act_seq=8 \
  model.optimizer.lr=4e-5 ~model.scheduler \
  +model.scheduler._target_=egomimic.utils.schedulers.warmup_cosine_scheduler \
  +model.scheduler._partial_=true +model.scheduler.max_steps=90000 \
  +model.scheduler.warmup_steps=500 +model.scheduler.warmup_start_factor=0.1 +model.scheduler.eta_min=4.0e-6 \
  evaluator=eval_hpt_standard \
  +evaluator.rollout_mode=chunk_openloop +evaluator.chunk_k=8 +evaluator.temporal_ensemble=false \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=50 \
  trainer=debug trainer.max_epochs=1800 trainer.limit_train_batches=50 \
  trainer.check_val_every_n_epoch=100 trainer.limit_val_batches=4 \
  trainer.profiler=null logger=csv_wandb logger.wandb.project=zarr_test \
  norm_stats.sample_frac=0.1 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal \
  data.train_datasets.pushshapes_sim.resolver.key_map.action_horizon=8 \
  data.valid_datasets.pushshapes_sim.resolver.key_map.action_horizon=8 \
  data.train_dataloader_params.pushshapes_sim.batch_size=256 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=256 \
  data.train_dataloader_params.pushshapes_sim.num_workers=16 \
  data.valid_dataloader_params.pushshapes_sim.num_workers=16 \
  ++data.train_dataloader_params.pushshapes_sim.pin_memory=true \
  ++data.valid_dataloader_params.pushshapes_sim.pin_memory=true \
  ++data.train_dataloader_params.pushshapes_sim.persistent_workers=true \
  ++data.valid_dataloader_params.pushshapes_sim.persistent_workers=true \
  ++data.train_dataloader_params.pushshapes_sim.prefetch_factor=4 \
  ++data.valid_dataloader_params.pushshapes_sim.prefetch_factor=4 \
  ${RESUME}
echo "TRAIN_EXIT=$?"
