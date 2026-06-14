#!/bin/bash
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
for SD in /opt/slurm/Ubuntu-*/current/bin /opt/slurm/Ubuntu-*/24.11.0/bin; do [ -x "$SD/srun" ] && export PATH="$SD:$PATH" && break; done
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
export ROLLOUT_VIDEO_DIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/upstream_eval/hnet_videos
mkdir -p "$ROLLOUT_VIDEO_DIR"
UP=/coc/flash7/paphiwetsa3/datasets/pusht_upstream
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
RUNDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetC8_nc3/bc_rnn_hnet_chunk8_2026-06-06_19-38-16
cp -n "${RUNDIR}/checkpoints/epoch_epoch=799.ckpt" "${RUNDIR}/checkpoints/up_ep799.ckpt"
python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=upstream_hnet_c8 description=upstream_replay_coverage mode=eval data=tsimulation \
  ckpt_path=${RUNDIR}/checkpoints/up_ep799.ckpt \
  model=bc_rnn_pushshapes_paperexact_hnet_chunk8 \
  evaluator=eval_hnet_sim evaluator.rollout_mode=ar \
  evaluator.init_mode=replay evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
  +evaluator.max_videos=2 \
  evaluator.limit_val_batches=7 trainer.limit_val_batches=7 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=4 \
  trainer=debug trainer.precision=32 trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$UP \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$UP \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax \
  norm_stats.precomputed_norm_path=${RUNDIR}/norm_stats/norm_stats.json
echo "HNET_EXIT=$?"
ls -la "$ROLLOUT_VIDEO_DIR"
