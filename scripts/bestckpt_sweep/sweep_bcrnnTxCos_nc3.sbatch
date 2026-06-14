#!/bin/bash
#SBATCH --job-name=sw_NoChunk
#SBATCH --time=04:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/sweep_bcrnnTxCos_nc3_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/sweep_bcrnnTxCos_nc3_%j.err
set -uxo pipefail
export PATH=/opt/slurm/Ubuntu-20.04/current/bin:$PATH
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
RUNDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnTxCos_nc3/bc_rnn_tx_cos_2026-06-05_17-00-36
MODEL=bc_rnn_pushshapes_paperexact_tx_cos
TAG=NoChunk
echo "SWEEP_RUN=bcrnnTxCos_nc3 MODEL=bc_rnn_pushshapes_paperexact_tx_cos EPOCHS=99 299 499 699 899 1099 1299"
for EP in 99 299 499 699 899 1099 1299; do
  SRC="${RUNDIR}/checkpoints/epoch_epoch=${EP}.ckpt"
  CLEAN="${RUNDIR}/checkpoints/sweep_NoChunk_ep${EP}.ckpt"
  if [ ! -f "$SRC" ]; then echo "SWEEP_SKIP epoch=${EP} reason=missing_ckpt"; continue; fi
  cp -n "$SRC" "$CLEAN"
  EPLOG=$(mktemp)
  srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
    name=sweep_NoChunk_ep${EP} description=bestckpt_sweep mode=eval data=tsimulation \
    ckpt_path="$CLEAN" \
    model=$MODEL \
    evaluator=eval_hnet_sim evaluator.rollout_mode=ar \
    evaluator.init_mode=seeds evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
    evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \
    trainer=debug trainer.precision=32 trainer.profiler=null logger=wandb \
    data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
    data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
    data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
    data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
    norm_stats.norm_mode=minmax \
    norm_stats.precomputed_norm_path=${RUNDIR}/norm_stats/norm_stats.json 2>&1 | tee "$EPLOG"
  COV=$(grep -oE "sim_coverage[^0-9]*[0-9]+\.[0-9]+" "$EPLOG" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
  SUC=$(grep -oE "sim_success_rate[^0-9]*[0-9]+\.[0-9]+" "$EPLOG" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
  echo "SWEEP_RESULT epoch=${EP} coverage=${COV:-NA} success_rate=${SUC:-NA}"
  rm -f "$EPLOG" "$CLEAN"
done
echo "SWEEP_DONE_NoChunk"
