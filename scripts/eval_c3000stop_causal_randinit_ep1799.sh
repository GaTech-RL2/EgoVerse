#!/bin/bash
#SBATCH --job-name=evalC3000StopRI1799
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalC3000StopRI_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalC3000StopRI_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled

RUNDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowC3000CausalStop/hpt_flow_circle_3000_stop_causal_obs1_act32_2026-06-13_19-15-13
CKPT=${RUNDIR}/checkpoints/eval_snapshot_stop_ep1799.ckpt
NORM=${RUNDIR}/norm_stats/norm_stats.json
DS=/coc/flash7/paphiwetsa3/datasets/circle_3000

export ROLLOUT_VIDEO_DIR=${RUNDIR}/eval/video_randinit_thr95_stop_ep1799
mkdir -p "$ROLLOUT_VIDEO_DIR"

# GENERALIZATION eval: RANDOM-NEW init positions (init_mode=seeds), chunk_openloop k=32,
# threshold 0.95. SAME config as the original hptFlowC3000Causal random-init eval (got 0.497
# @ep1799); only RUNDIR/CKPT/NORM/name/video-dir differ. This is the circle_3000_stop run,
# evaluated at its latest snapshot (~ep599, NOT converged).
srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=evalC3000StopRI1799 description=eval_c3000stop_causal_chunkol_k32_randinit_ep599 mode=eval \
  data=tsimulation model=hpt_pushshapes_circle evaluator=eval_hpt_standard \
  ckpt_path=${CKPT} \
  evaluator.init_mode=seeds \
  '+evaluator.init_seeds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]' \
  '+evaluator.rollout_mode=chunk_openloop' \
  '+evaluator.chunk_k=32' \
  '+evaluator.temporal_ensemble=false' \
  evaluator.max_steps=500 \
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
