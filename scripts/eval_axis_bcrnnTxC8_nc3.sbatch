#!/bin/bash
#SBATCH --job-name=ev_bcrnnTxC8_nc3
#SBATCH --time=01:10:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/ev_bcrnnTxC8_nc3_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/ev_bcrnnTxC8_nc3_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
RUNDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnTxC8_nc3/bc_rnn_tx_chunk8_2026-06-05_23-40-22/
cp -n "${RUNDIR}checkpoints/epoch_epoch=399.ckpt" "/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnTxC8_nc3/bc_rnn_tx_chunk8_2026-06-05_23-40-22/checkpoints/bcrnnTxC8_nc3_ep400.ckpt"
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=bcrnnTxC8_nc3_ep399_eval description=bcrnnTxC8_nc3_axis_rollout mode=eval data=tsimulation \
  ckpt_path=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnTxC8_nc3/bc_rnn_tx_chunk8_2026-06-05_23-40-22/checkpoints/bcrnnTxC8_nc3_ep400.ckpt \
  model=bc_rnn_pushshapes_paperexact_tx_chunk8 \
  evaluator=eval_hnet_sim evaluator.rollout_mode=ar \
  evaluator.init_mode=seeds evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
  evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \
  trainer=debug trainer.precision=32 trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax \
  norm_stats.precomputed_norm_path=${RUNDIR}norm_stats/norm_stats.json
echo "bcrnnTxC8_nc3_EXIT=$?"
