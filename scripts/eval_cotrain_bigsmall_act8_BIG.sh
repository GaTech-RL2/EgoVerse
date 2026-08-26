#!/bin/bash
#SBATCH --job-name=evalCotrainBigSmallAct8_BIG
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalCotrainBigSmallAct8_BIG_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalCotrainBigSmallAct8_BIG_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled

# auto-discover the cotrain run dir + its own norm_stats
RUNDIR=$(ls -td /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptCotrainBigSmallAct8/*/ 2>/dev/null | head -1)
CKPT=${RUNDIR}/checkpoints/last.ckpt
NORM=${RUNDIR}/norm_stats/norm_stats.json
DS=/coc/flash7/paphiwetsa3/datasets/circle_3000
echo "RUNDIR=$RUNDIR"; echo "CKPT=$CKPT"; echo "NORM=$NORM"
export ROLLOUT_VIDEO_DIR=${RUNDIR}/eval/video_BIGcircle_chunkol_k8_randinit_thr95
mkdir -p "$ROLLOUT_VIDEO_DIR"

# BIG circle (default pusher_shape=circle, r=15). Goal-conditioned -> env samples a
# random in-distribution goal per seed. random-init generalization, chunk_openloop k=8, thr 0.95, 20 seeds.
srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=evalCotrainBigSmallAct8_BIG description=eval_cotrain_bigsmall_act8_BIGcircle_chunkol_k8_randinit mode=eval \
  data=tsimulation model=hpt_pushshapes_circle evaluator=eval_hpt_standard \
  ++model.robomimic_model.head_specs.pushshapes_sim.action_horizon=8 \
  ++model.robomimic_model.head_specs.pushshapes_sim.model.act_seq=8 \
  ckpt_path=${CKPT} \
  evaluator.init_mode=seeds \
  '+evaluator.init_seeds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]' \
  '+evaluator.rollout_mode=chunk_openloop' \
  '+evaluator.chunk_k=8' \
  '+evaluator.temporal_ensemble=false' \
  evaluator.env_kwargs.pusher_shape=circle \
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
