#!/bin/bash
#SBATCH --job-name=evalC200Causal
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalC200Causal_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalC200Causal_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled

RUNDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowC200Causal_pushshapesPaper/hpt_flow_pushshapes_paper_causal_obs1_act32_2026-06-13_06-03-25
CKPT=${RUNDIR}/checkpoints/eval_snapshot_c200_last.ckpt
NORM=${RUNDIR}/norm_stats/norm_stats.json
DS=/coc/flash7/paphiwetsa3/datasets/pushshapes_paper

export ROLLOUT_VIDEO_DIR=${RUNDIR}/eval/video_chunkol_k32_replay_thr95_ep899
mkdir -p "$ROLLOUT_VIDEO_DIR"

# CLOSED-LOOP chunk_openloop rollout, training-consistent:
#  observe 1 frame -> predict 32 actions -> execute all 32 open-loop -> replan.
#  rollout_mode=chunk_openloop chunk_k=32 temporal_ensemble=false (match training overrides).
#  init_mode=replay (fixed-goal [256,256,pi/4] dataset -> replay reproduces recorded fixed-goal scenes; training-consistent).
#  20 episodes via batch_size=20 + limit_val_batches=1.
#  causal keymap to match the run's own data config.
#  max_steps=400 (covers pushshapes_paper max ~338 frames).
srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=evalC200Causal description=eval_c200_causal_chunkol_k32_replay_ep899 mode=eval \
  data=tsimulation model=hpt_pushshapes_circle evaluator=eval_hpt_standard \
  ckpt_path=${CKPT} \
  evaluator.init_mode=replay \
  '+evaluator.rollout_mode=chunk_openloop' \
  '+evaluator.chunk_k=32' \
  '+evaluator.temporal_ensemble=false' \
  evaluator.max_steps=400 \
  evaluator.coverage_threshold=0.95 \
  evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \
  trainer=debug trainer.profiler=null logger=csv \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal_eval \
  data.valid_dataloader_params.pushshapes_sim.batch_size=20 \
  norm_stats.norm_mode=quantile \
  norm_stats.precomputed_norm_path=$NORM
echo "EVAL_EXIT=$?"
