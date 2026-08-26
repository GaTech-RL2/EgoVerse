#!/bin/bash
#SBATCH --job-name=evalHnetC8FHCo_BIG
#SBATCH --time=01:10:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=ig-88
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalHnetC8FHCo_BIG_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalHnetC8FHCo_BIG_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/circle_3000
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
RUNDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetC8FHCotrainBigSmall/bc_rnn_hnet_chunk8_fullhist_cotrain_big_small_2026-06-18_14-19-47
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=evalHnetC8FHCo_BIG description=cotrain_at_circle_1min20seed mode=eval data=tsimulation \
  ckpt_path=${RUNDIR}/checkpoints/last.ckpt \
  model=bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist \
  evaluator=eval_hnet_sim evaluator.rollout_mode=ar \
  evaluator.init_mode=seeds ++evaluator.init_seeds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39] evaluator.max_steps=1024 evaluator.coverage_threshold=0.95 \
  ++evaluator.env_kwargs.pusher_shape=circle ++evaluator.env_kwargs.obstacle_level=0 \
  ++data.valid_dataloader_params.pushshapes_sim.batch_size=20 \
  evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \
  trainer=debug trainer.precision=32 trainer.profiler=null logger=csv \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax \
  norm_stats.precomputed_norm_path=${RUNDIR}/norm_stats/norm_stats.json
echo "EVAL_EXIT=$?"
