#!/bin/bash
#SBATCH --job-name=genlvl
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/genlvl/%x_%A_%a.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/genlvl/%x_%A_%a.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
# Params via --export: CELL, RUN, DATASET, FAMILY (hpt|tx). LVL = array task id.
LVL=${SLURM_ARRAY_TASK_ID:-0}
RD=$(ls -dt logs/${RUN}/*/ 2>/dev/null | head -1)
CK=${RD}checkpoints/last.ckpt
NORM=${RD}norm_stats/norm_stats.json
SEEDS='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]'
echo "CELL=$CELL FAMILY=$FAMILY LVL=$LVL RD=$RD"

if [ "$FAMILY" = "hpt" ]; then
  srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
    name=genlvl_${CELL}_L${LVL} mode=eval data=tsimulation model=hpt_pushshapes_circle \
    ++model.robomimic_model.head_specs.pushshapes_sim.action_horizon=8 \
    ++model.robomimic_model.head_specs.pushshapes_sim.model.act_seq=8 \
    ckpt_path=${CK} evaluator=eval_hpt_standard \
    evaluator.init_mode=seeds "+evaluator.init_seeds=${SEEDS}" \
    "+evaluator.rollout_mode=chunk_openloop" "+evaluator.chunk_k=8" "+evaluator.temporal_ensemble=false" \
    "++evaluator.env_kwargs.obstacle_level=${LVL}" evaluator.max_steps=900 evaluator.coverage_threshold=0.95 \
    evaluator.limit_val_batches=1 trainer.limit_val_batches=1 trainer=debug trainer.profiler=null logger=csv \
    data.train_datasets.pushshapes_sim.resolver.folder_path=${DATASET} \
    data.valid_datasets.pushshapes_sim.resolver.folder_path=${DATASET} \
    data.train_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal \
    data.valid_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal_eval \
    data.train_datasets.pushshapes_sim.resolver.key_map.action_horizon=8 \
    data.valid_datasets.pushshapes_sim.resolver.key_map.action_horizon=8 \
    norm_stats.norm_mode=quantile norm_stats.precomputed_norm_path=${NORM}
else
  srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
    name=genlvl_${CELL}_L${LVL} mode=eval data=tsimulation model=bc_rnn_pushshapes_paperexact_tx_chunk8_fullhist \
    ckpt_path=${CK} evaluator=eval_hnet_sim evaluator.rollout_mode=ar \
    evaluator.init_mode=seeds "+evaluator.init_seeds=${SEEDS}" \
    "++evaluator.env_kwargs.obstacle_level=${LVL}" evaluator.max_steps=900 evaluator.coverage_threshold=0.95 \
    evaluator.limit_val_batches=1 trainer.limit_val_batches=1 trainer=debug trainer.profiler=null logger=csv \
    data.train_datasets.pushshapes_sim.resolver.folder_path=${DATASET} \
    data.valid_datasets.pushshapes_sim.resolver.folder_path=${DATASET} \
    data.train_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_eval \
    data.valid_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_eval \
    norm_stats.norm_mode=minmax norm_stats.precomputed_norm_path=${NORM}
fi
echo "GENLVL_${CELL}_L${LVL}_EXIT=$?"
