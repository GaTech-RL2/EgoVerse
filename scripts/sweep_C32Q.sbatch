#!/bin/bash
#SBATCH --job-name=sweep_C32Q
#SBATCH --time=05:30:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/sweep_C32Q_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/sweep_C32Q_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
RUNDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnTxC32Q_nc3/bc_rnn_tx_chunk32_queries_2026-06-07_17-04-45/
CKPTDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnTxC32Q_nc3/bc_rnn_tx_chunk32_queries_2026-06-07_17-04-45/checkpoints
RESFILE=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sweep_results/C32Q_curve.txt
mkdir -p $(dirname $RESFILE)
echo "# C32Q sweep $(date)" > $RESFILE
for EP in 99 199 299 399 499 599; do
  SRC="$CKPTDIR/epoch_epoch=${EP}.ckpt"
  CLEAN="$CKPTDIR/sweep_C32Q_ep${EP}.ckpt"
  if [ ! -f "$SRC" ]; then echo "$EP MISSING_CKPT" >> $RESFILE; continue; fi
  cp -n "$SRC" "$CLEAN"
  LOG=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/sweep_C32Q_ep${EP}.evlog
  srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
    name=sweep_C32Q_ep${EP} description=sweep_C32Q_ep${EP} mode=eval data=tsimulation \
    ckpt_path=$CLEAN \
    model=bc_rnn_pushshapes_paperexact_tx_chunk32_q \
    evaluator=eval_hnet_sim evaluator.rollout_mode=ar \
    evaluator.init_mode=seeds evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
    evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \
    trainer=debug trainer.precision=32 trainer.profiler=null logger=wandb \
    data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
    data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
    data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
    data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
    norm_stats.norm_mode=minmax \
    norm_stats.precomputed_norm_path=${RUNDIR}norm_stats/norm_stats.json \
    > $LOG 2>&1
  COV=$(grep -oE "emb15_sim_coverage[^0-9]*[0-9]+\.[0-9]+" $LOG | grep -oE "[0-9]+\.[0-9]+" | tail -1)
  SUC=$(grep -oE "emb15_sim_success_rate[^0-9]*[0-9]+\.[0-9]+" $LOG | grep -oE "[0-9]+\.[0-9]+" | tail -1)
  echo "$EP cov=${COV:-NA} succ=${SUC:-NA}" >> $RESFILE
  rm -f "$CLEAN"
done
echo "DONE_C32Q" >> $RESFILE
