#!/bin/bash
#SBATCH --job-name=eval_chunktraj_A
#SBATCH --account=overcap
#SBATCH --partition=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=00:25:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/eval_chunktraj_A_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/eval_chunktraj_A_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
export CHUNK_TRAJ_DIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/chunktraj_A
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json
CKPT=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_baseline_nc3/chunkA_tf_resnet_2026-06-03_21-17-03/checkpoints/last.ckpt
MODEL=hnet_pushshapes_fused_pusher_resnet
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=hnet_baseline_nc3 description=chunktraj_A mode=eval data=tsimulation \
  ckpt_path=${CKPT} model=${MODEL} \
  ++model.robomimic_model.chunk_k=32 \
  ++model.robomimic_model.action_horizon=1024 \
  evaluator=eval_hnet_sim evaluator.rollout_mode=chunk_openloop evaluator.chunk_k=32 \
  evaluator.init_mode=replay evaluator.max_steps=400 \
  trainer=debug trainer.limit_val_batches=1 trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.precomputed_norm_path=$NORM
echo "EVAL_EXIT=$?"
