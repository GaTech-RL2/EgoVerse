#!/usr/bin/env bash
# Accuracy A/B for the vectorized augmentations: same seed, same data, same
# config, only the augmentation path differs.
#   bench/ab_augs.sh <arm: batched|loop> <out.json> [extra hydra overrides...]
# Train and held-out episodes are disjoint, so the held-out loss the ABMetrics
# callback records is a generalization measure, not a fit measure.
set -euo pipefail
cd "$(dirname "$0")/.."
source ./wt-env.sh
export EGOVERSE_DATASET_DIR="${EGOVERSE_DATASET_DIR:-/workspace/users/copierceryan/datasets}"
export TOKENIZERS_PARALLELISM=false
ALL=($(ls "$EGOVERSE_DATASET_DIR"))
TRAIN=$(printf '%s\n' "${ALL[@]:0:32}" | paste -sd, -)
VALID=$(printf '%s\n' "${ALL[@]:32}" | paste -sd, -)

ARM="$1"; OUT="$2"; shift 2
case "$ARM" in
  batched) ENTRY=egomimic/trainHydra.py ;;
  loop)    ENTRY=bench/train_loop_augs.py ;;
  *) echo "arm must be batched or loop" >&2; exit 2 ;;
esac

EPOCHS="${AB_EPOCHS:-20}"
python "$ENTRY" --config-name=train_zarr_cartesian \
  data=mecka_fold_freeform_opscale1_6d model=hpt_bc_mecka_6d_300M \
  "seed=${AB_SEED:-42}" \
  "data.train_datasets.human_bimanual.resolver._target_=egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver" \
  "data.train_datasets.human_bimanual.filters=null" \
  "++data.train_datasets.human_bimanual.filters={_target_: egomimic.rldb.filters.DatasetFilter, episode_hashes: [${TRAIN}]}" \
  "data.valid_datasets.human_bimanual.filters=null" \
  "++data.valid_datasets.human_bimanual.filters={_target_: egomimic.rldb.filters.DatasetFilter, episode_hashes: [${VALID}]}" \
  "~data.unseen_op_valid_datasets" "~data.unseen_op_valid_dataloader_params" \
  "~data.video_episodes" \
  trainer=default trainer.accelerator=gpu trainer.devices=1 \
  "trainer.max_epochs=${EPOCHS}" "trainer.min_epochs=${EPOCHS}" \
  trainer.limit_train_batches=100 +trainer.enable_checkpointing=false \
  trainer.limit_val_batches=0 trainer.check_val_every_n_epoch=100000 \
  +trainer.enable_progress_bar=false logger=debug '~evaluator' '~callbacks' train_viz=false \
  "+callbacks={ab: {_target_: bench.ab_metrics.ABMetrics, out_path: '${OUT}', eval_batches: 12}}" \
  norm_stats.sample_frac=0.02 name=ab_augs "description=$ARM" "$@"
