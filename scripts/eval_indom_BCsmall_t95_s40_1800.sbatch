#!/bin/bash
#SBATCH --job-name=indom_BCsmall
#SBATCH --time=01:30:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=ig-88
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/indom_BCsmall_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/indom_BCsmall_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
DS=/coc/flash7/paphiwetsa3/datasets/small_circle_3000
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
RUNDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnTxSmallCircle3000C8FH/bc_rnn_tx_chunk8_fullhist_smallcircle3000_2026-06-19_18-16-23
CK=${RUNDIR}/checkpoints/last.ckpt
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=indom_BCsmall description=indom_BCsmall_1min20seed mode=eval data=tsimulation \
  ckpt_path=${CK} \
  model=bc_rnn_pushshapes_paperexact_tx_chunk8_fullhist ++model.robomimic_model.action_horizon=1800 \
  evaluator=eval_hnet_sim evaluator.rollout_mode=ar \
  ++evaluator.env_kwargs.pusher_shape=circle_small \
  evaluator.init_mode=seeds ++evaluator.init_seeds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39] \
  evaluator.max_steps=1800 evaluator.coverage_threshold=0.95 evaluator.video_fps=30 \
  evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \
  trainer=debug trainer.precision=32 trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax \
  norm_stats.precomputed_norm_path=${RUNDIR}/norm_stats/norm_stats.json
echo "indom_BCsmall_EXIT=$?"
