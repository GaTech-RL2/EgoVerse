#!/bin/bash
#SBATCH --job-name=split_diag_chunkB
#SBATCH --time=00:45:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/split_diag_chunkB_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/split_diag_chunkB_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200

NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json
CKPT=logs/hnet_baseline_nc3/chunkB_rollout_resnet_2026-06-03_21-17-03/checkpoints/last.ckpt
OUTBASE=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/split_diag_chunkB

export SPLIT_DIAG_NEPS=2      # first 2 demo-init episodes
export SPLIT_DIAG_CONT=120    # continuation / restart length

# run_mode MODE TE SPLIT OUTSUBDIR
run_mode () {
  local MODE=$1 TE=$2 SPLIT=$3 SUB=$4
  export SPLIT_DIAG_SPLIT=$SPLIT
  export SPLIT_DIAG_DIR=${OUTBASE}/${SUB}
  echo "=========== SPLIT DIAG MODE=${MODE} TE=${TE} SPLIT=${SPLIT} -> ${SPLIT_DIAG_DIR} ==========="
  srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
    name=split_diag_chunkB description=${SUB} mode=eval data=tsimulation \
    ckpt_path=${CKPT} model=hnet_pushshapes_fused_windowed_resnet \
    ++model.robomimic_model.chunk_k=32 ++model.robomimic_model.action_horizon=1024 \
    evaluator=eval_hnet_sim evaluator.rollout_mode=${MODE} \
    evaluator.temporal_ensemble=${TE} evaluator.chunk_k=32 \
    evaluator.init_mode=replay evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
    evaluator.limit_val_batches=1 \
    trainer=debug trainer.profiler=null logger=wandb \
    data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
    data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
    data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
    data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
    norm_stats.precomputed_norm_path=$NORM
  echo "MODE_${SUB}_EXIT=$?"
}

# chunk_openloop at the task-specified SPLIT=120 (mid-chunk) AND at a chunk
# boundary SPLIT=128 (phase-aligned: continuation re-observes at the same step
# as the restart, isolating world-state reproduction from chunk-phase artifacts).
run_mode chunk_openloop false 120 chunk_openloop
run_mode chunk_openloop false 128 chunk_openloop_bdry
# chunk_te (per-step re-observe; SPLIT value irrelevant to phase).
run_mode chunk_te        true  120 chunk_te

echo "ALL_DONE_EXIT=$?"
