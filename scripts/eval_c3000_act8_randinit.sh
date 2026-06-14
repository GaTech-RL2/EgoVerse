#!/bin/bash
#SBATCH --job-name=evalC3000Act8RI
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalC3000Act8RI_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalC3000Act8RI_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled

RUNDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowC3000CausalAct8/hpt_flow_circle_3000_causal_obs1_act8_2026-06-14_01-39-26
CKPT=${RUNDIR}/checkpoints/last.ckpt
NORM=${RUNDIR}/norm_stats/norm_stats.json
DS=/coc/flash7/paphiwetsa3/datasets/circle_3000

export ROLLOUT_VIDEO_DIR=${RUNDIR}/eval/video_chunkol_k8_randinit_thr95_earlypeek
mkdir -p "$ROLLOUT_VIDEO_DIR"

# EARLY PEEK (~ep300, NOT converged): random-init generalization eval, chunk_openloop k=8,
# threshold 0.95. Same protocol as the act32 c3000 random-init eval; only k=8 + act8 model
# overrides + this run's ckpt/norm differ. c3000 is goal-conditioned -> env samples a random
# in-distribution goal per seed.
srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=evalC3000Act8RI description=eval_c3000_act8_causal_chunkol_k8_randinit_earlypeek mode=eval \
  data=tsimulation model=hpt_pushshapes_circle evaluator=eval_hpt_standard \
  ++model.robomimic_model.head_specs.pushshapes_sim.action_horizon=8 \
  ++model.robomimic_model.head_specs.pushshapes_sim.model.act_seq=8 \
  ckpt_path=${CKPT} \
  evaluator.init_mode=seeds \
  '+evaluator.init_seeds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]' \
  '+evaluator.rollout_mode=chunk_openloop' \
  '+evaluator.chunk_k=8' \
  '+evaluator.temporal_ensemble=false' \
  evaluator.max_steps=500 \
  evaluator.coverage_threshold=0.95 \
  evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \
  trainer=debug trainer.profiler=null logger=csv \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal_eval \
  data.train_datasets.pushshapes_sim.resolver.key_map.action_horizon=8 \
  data.valid_datasets.pushshapes_sim.resolver.key_map.action_horizon=8 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=20 \
  norm_stats.norm_mode=quantile \
  norm_stats.precomputed_norm_path=$NORM
echo "EVAL_EXIT=$?"
