#!/bin/bash
#SBATCH --job-name=bcrnnHnetCo_vid
#SBATCH --time=01:10:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/bcrnnHnetCo_vid_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/bcrnnHnetCo_vid_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
export ROLLOUT_VIDEO_DIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetCo_ep1200_videos
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
RUNDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetCo_nc3/bc_rnn_hnet_cotrain_2026-06-06_02-55-53
cp -n "${RUNDIR}/checkpoints/epoch_epoch=1199.ckpt" ${RUNDIR}/checkpoints/ep1200.ckpt
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=bcrnnHnetCo_ep1200_vid description=bcrnnHnetCo_rollout mode=eval data=tsimulation \
  ckpt_path=${RUNDIR}/checkpoints/ep1200.ckpt \
  model=bc_rnn_pushshapes_paperexact_hnet \
  evaluator=eval_hnet_sim evaluator.rollout_mode=ar \
  evaluator.init_mode=seeds evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
  +evaluator.max_videos=6 \
  evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \
  trainer=debug trainer.precision=32 trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax \
  norm_stats.precomputed_norm_path=${RUNDIR}/norm_stats/norm_stats.json
echo "BCRNNHNETCO_EXIT=$?"
ls $ROLLOUT_VIDEO_DIR
