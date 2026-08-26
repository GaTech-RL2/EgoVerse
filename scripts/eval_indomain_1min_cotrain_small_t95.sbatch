#!/bin/bash
#SBATCH --job-name=evalID_cotrain_small
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalID_cotrain_small_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalID_cotrain_small_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
which python
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled PACK_COLLATE_MAX_TOTAL_FRAMES=3200

RUNDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptCotrainBigSmallAct8/hpt_flow_cotrain_big_small_3000_causal_obs1_act8_2026-06-16_17-41-59
CKPT=${RUNDIR}/checkpoints/last.ckpt
NORM=${RUNDIR}/norm_stats/norm_stats.json
DS=/coc/flash7/paphiwetsa3/datasets/small_circle_3000

export ROLLOUT_VIDEO_DIR=${RUNDIR}/eval/indomain_1min_cotrain_small_20seeds_thr70
mkdir -p "$ROLLOUT_VIDEO_DIR"

# Replicates the working BC eval (slurm 3342890, emb15_sim_coverage 0.4204) verbatim,
# changing ONLY ckpt/norm/dataset to the COTRAIN run + adding the act_horizon=8 / act_seq=8
# model overrides that the FAILED cotrain eval (3342527) was missing -> that omission caused
# the FMPolicy head state_dict size mismatch. Cotrain was TRAINED with model=hpt_pushshapes_circle
# + those exact act8 overrides (verified from its .hydra/overrides.yaml), so the architecture
# matches and the state_dict loads. data=tsimulation, get_keymap_eval, 20 seeds, max_steps=1800,
# evaluator defaults (chunk_openloop, chunk_k=8, coverage_threshold=0.95, init_mode=seeds).
srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=evalID_cotrain_small description=indomain_1min_cotrain_small mode=eval \
  data=tsimulation \
  ckpt_path=${CKPT} \
  model=hpt_pushshapes_circle \
  model.robomimic_model.head_specs.pushshapes_sim.action_horizon=8 \
  model.robomimic_model.head_specs.pushshapes_sim.model.act_seq=8 \
  evaluator=eval_hpt_standard \
  evaluator.init_mode=seeds \
  '++evaluator.init_seeds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]' \
  evaluator.max_steps=1800 \
  '+evaluator.rollout_mode=chunk_openloop' \
  '+evaluator.chunk_k=8' \
  '+evaluator.temporal_ensemble=false' \
  evaluator.env_kwargs.pusher_shape=circle_small \
  evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \
  trainer=debug trainer.profiler=null logger=csv \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_eval \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_eval \
  norm_stats.norm_mode=quantile \
  norm_stats.precomputed_norm_path=$NORM
echo "UECOTRAIN_EXIT=$?"
echo "########## DONE ##########"
