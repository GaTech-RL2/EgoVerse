#!/bin/bash
#SBATCH --job-name=evalC950CausalRI
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalC950CausalRI_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalC950CausalRI_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled

# c950 (new_circle_3, ~954 ep, act32) random-init generalization eval. Matched to the
# c3000 (0.497) and c3000_stop (0.513) @ep1799 numbers: SAME protocol (init_mode=seeds,
# chunk_openloop k=32, thr0.95), only dataset/ckpt/norm differ. Data-quantity comparison.
RUNDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowC950Causal/hpt_flow_new_circle_3_causal_obs1_act32_2026-06-14_07-56-49
CKPT=${RUNDIR}/checkpoints/last.ckpt
NORM=${RUNDIR}/norm_stats/norm_stats.json
DS=/coc/flash7/paphiwetsa3/datasets/new_circle_3

export ROLLOUT_VIDEO_DIR=${RUNDIR}/eval/video_chunkol_k32_randinit_thr95_ep1799
mkdir -p "$ROLLOUT_VIDEO_DIR"

srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=evalC950CausalRI description=eval_c950_causal_chunkol_k32_randinit_ep1799 mode=eval \
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
