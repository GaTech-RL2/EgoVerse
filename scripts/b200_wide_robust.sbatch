#!/bin/bash
#SBATCH --job-name=b200_wide_robust
#SBATCH --time=03:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/b200_wide_robust_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/b200_wide_robust_%j.err
set -uxo pipefail
export PATH=/opt/slurm/Ubuntu-20.04/current/bin:$PATH
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
RUNDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnTxCosB200_nc3/bc_rnn_tx_cos_bal200_2026-06-06_16-38-11
MODEL=bc_rnn_pushshapes_paperexact_tx_cos
# Wider 50-seed suite [0..49] to test robustness of the 0.262 spike at ep1799.
SEEDS="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]"
echo "B200_WIDE_SWEEP seeds=0..49 (n=50) EPOCHS=1799 1699 1499"
for EP in 1799 1699 1499; do
  SRC="${RUNDIR}/checkpoints/epoch_epoch=${EP}.ckpt"
  CLEAN="${RUNDIR}/checkpoints/b200wide_ep${EP}.ckpt"
  if [ ! -f "$SRC" ]; then echo "B200_WIDE_SKIP epoch=${EP} reason=missing_ckpt"; continue; fi
  cp -n "$SRC" "$CLEAN"
  EPLOG=$(mktemp)
  srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
    name=b200wide_ep${EP} description=b200_wide50seed_robust mode=eval data=tsimulation \
    ckpt_path="$CLEAN" \
    model=$MODEL \
    evaluator=eval_hnet_sim evaluator.rollout_mode=ar \
    evaluator.init_mode=seeds "evaluator.init_seeds=${SEEDS}" \
    evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
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
  echo "B200_WIDE_RESULT epoch=${EP} seeds=50 coverage=${COV:-NA} success_rate=${SUC:-NA}"
  rm -f "$EPLOG" "$CLEAN"
done
echo "B200_WIDE_DONE"
