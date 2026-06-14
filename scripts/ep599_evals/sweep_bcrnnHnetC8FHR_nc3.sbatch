#!/bin/bash
#SBATCH --job-name=sweep_HnetFull
#SBATCH --time=08:30:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/sweep_HnetFull_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/sweep_HnetFull_%j.err
set -uxo pipefail
export PATH=/opt/slurm/Ubuntu-20.04/current/bin:$PATH
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
RUNDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetC8FHR_nc3/bc_rnn_hnet_chunk8_fullhist_ratio_2026-06-06_22-31-32
CKDIR=${RUNDIR}/checkpoints
MODEL=bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist_ratio

# >12 ckpts available (99..1699 step 100) -> eval every 200 epochs to bound cost
for EP in 199 399 599 799 999 1199 1399 1599; do
  SRC="${CKDIR}/epoch_epoch=${EP}.ckpt"
  CLEAN="${CKDIR}/sweep_ep${EP}.ckpt"
  if [ ! -f "${SRC}" ]; then echo "SWEEP_SKIP epoch=${EP} (missing ${SRC})"; continue; fi
  cp -n "${SRC}" "${CLEAN}"
  echo "================ SWEEP_RUN epoch=${EP} ================"
  srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
    name=sweep_HnetFull_ep${EP} description=bestckpt_sweep mode=eval data=tsimulation \
    ckpt_path=${CLEAN} \
    model=${MODEL} \
    evaluator=eval_hnet_sim evaluator.rollout_mode=ar \
    evaluator.init_mode=seeds evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
    evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \
    trainer=debug trainer.precision=32 trainer.profiler=null logger=wandb \
    data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
    data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
    data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
    data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
    norm_stats.norm_mode=minmax \
    norm_stats.precomputed_norm_path=${RUNDIR}/norm_stats/norm_stats.json \
    2>&1 | sed "s/^/[ep${EP}] /"
  echo "SWEEP_DONE epoch=${EP} rc=${PIPESTATUS[0]}"
done
echo "SWEEP_ALL_DONE_HnetFull"
