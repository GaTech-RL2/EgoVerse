#!/usr/bin/env bash
# End-to-end HPT throughput: the real trainHydra path on locally cached mecka
# episodes, so the dataloader is in the measurement too.
#   bench/e2e_hpt.sh <run-name> [extra hydra overrides...]
set -euo pipefail
cd "$(dirname "$0")/.."
source ./wt-env.sh
export EGOVERSE_DATASET_DIR="${EGOVERSE_DATASET_DIR:-/workspace/users/copierceryan/datasets}"
export TOKENIZERS_PARALLELISM=false
HASHES=$(ls "$EGOVERSE_DATASET_DIR" | head -20 | paste -sd, -)
RUN="$1"; shift
python egomimic/trainHydra.py --config-name=train_zarr_cartesian \
  data=mecka_fold_freeform_opscale1_6d model=hpt_bc_mecka_6d_300M \
  "data.train_datasets.human_bimanual.resolver._target_=egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver" \
  "data.train_datasets.human_bimanual.filters=null" \
  "++data.train_datasets.human_bimanual.filters={_target_: egomimic.rldb.filters.DatasetFilter, episode_hashes: [${HASHES}]}" \
  "data.valid_datasets.human_bimanual.filters=null" \
  "++data.valid_datasets.human_bimanual.filters={_target_: egomimic.rldb.filters.DatasetFilter, episode_hashes: [${HASHES}]}" \
  "~data.unseen_op_valid_datasets" "~data.unseen_op_valid_dataloader_params" \
  "~data.video_episodes" \
  trainer=default trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=5 trainer.min_epochs=5 \
  trainer.limit_train_batches=40 +trainer.enable_checkpointing=false trainer.limit_val_batches=0 trainer.check_val_every_n_epoch=1000 \
  +trainer.enable_progress_bar=false logger=debug '~evaluator' '~callbacks' train_viz=false \
  '+callbacks={step_timer: {_target_: bench.step_timer.StepTimer, warmup_steps: 40}}' \
  norm_stats.sample_frac=0.01 +trainer.profiler=simple name=perf_e2e "description=$RUN" "$@"
