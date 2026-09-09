#!/bin/bash
#SBATCH --job-name=e1foldam
#SBATCH --partition=coe-gpu
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=4:00:00
#SBATCH --output=/home/hice1/agao81/scratch/logs/e1foldam_%x_%j.log
# Re-score ONE finished mecka fold E1 cell (runs/e1_fold/<variant>_<spread>_s<seed>) under the
# current evaluator, which computes the lab's arc-matched paired MSE alongside the E_time-era
# metrics. These cells predate the picker: each holds exactly ONE checkpoint (epoch_epoch=299 =
# step 30000, the final), so there is nothing to select and no selection leak -- every row is
# scored at the same budget at its only checkpoint. Results land in eval_final_<set>/ so the
# standard harvest (eval_(best|final)_<set>) picks them up.
#   sbatch --job-name=amf_<run> --export=ALL,RUN=<run dir>,VARIANT=arcmean,SPREAD=full,SEED=42 fold_eval_only.sbatch
set -euo pipefail
WT=${WT:-$HOME/scratch/EgoVerse/.claude/worktrees/arc-e1}; PY=${PY:-$HOME/scratch/EgoVerse/.venv/bin/python}
DATA=${DATA:-$HOME/scratch/data/e1_fold}; EXP=${EXP:-fold}
# train_root is unused in mode=eval but the datamodule still builds it, and the fold TRAIN splits
# are symlinks into episodes that were released from ICE. TRAIN_ROOT points it at a set that is
# actually on disk; it never reaches the model.
TRAIN_ROOT=${TRAIN_ROOT:-$DATA/val_e1}
cd $WT; export PYTHONPATH=$WT HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_$SLURM_JOB_ID TRITON_CACHE_DIR=/tmp/tr_$SLURM_JOB_ID
H_MATCH=${H_MATCH:-$(cat $DATA/h_match_frames)}
CKPT=$(ls $RUN/train/checkpoints/*.ckpt | head -1)
NORM_DIR=$(dirname $(find $RUN/train -name "norm_stats*.json" | head -1))
echo "=== $RUN variant=$VARIANT spread=$SPREAD seed=$SEED ckpt=$CKPT norm=$NORM_DIR ($(date)) ==="
$PY - "$RUN" "$CKPT" <<'PYEOF'
import json, os, sys
run, ck = sys.argv[1], sys.argv[2]
p = f"{run}/best.json"
d = json.load(open(p)) if os.path.exists(p) else {}
d.update({"best_ckpt": ck, "final_ckpt": ck, "single_checkpoint": True})
json.dump(d, open(p, "w"), indent=2)
PYEOF
for TS in ${TEST_SETS:-test_low test_mid test_high}; do
  OUTD=$RUN/eval_final_$TS
  if $PY -c "import json,sys;r=json.load(open('$OUTD/eval_metrics.json'))['results'][0];sys.exit(0 if any('arcmatch_paired_mse' in k for k in r) else 1)" 2>/dev/null; then
    echo "have $TS (arcmatch)"; continue
  fi
  echo "=== EVAL $TS ($(date)) ==="
  $PY egomimic/trainHydra.py --config-name=train_zarr_cartesian +experiment=e1/${EXP}_${VARIANT} mode=eval "ckpt_path=\"$CKPT\"" \
    e1.spread=$SPREAD e1.train_root=$TRAIN_ROOT e1.valid_root=$DATA/$TS e1.h_match_frames=$H_MATCH e1.limit_val_batches=1.0 \
    seed=$SEED norm_stats.precomputed_norm_path=$NORM_DIR \
    trainer.devices=1 trainer.num_nodes=1 launch_params.gpus_per_node=1 launch_params.nodes=1 \
    ${EVAL_EXTRA:-} hydra.run.dir=$OUTD || echo "EVAL FAILED $TS"
done
$PY - "$RUN" <<'PYEOF'
import json, sys
run = sys.argv[1]; d = json.load(open(f"{run}/best.json"))
d["rescore_arcmatch_paired_mse"] = {"val": None, "metric": "arcmatch_paired_mse",
                                    "val_e1": {"best": None, "final": None}, "selected": "final",
                                    "note": "single checkpoint (step 30000); no selection"}
json.dump(d, open(f"{run}/best.json", "w"), indent=2)
PYEOF
echo "=== DONE ($(date)) ==="
