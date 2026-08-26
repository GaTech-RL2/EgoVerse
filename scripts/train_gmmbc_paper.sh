#!/bin/bash
#SBATCH --job-name=gmmBCpaper
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/gmmbc_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/gmmbc_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
export WANDB_MODE=online
PAPER=/coc/flash7/paphiwetsa3/datasets/pushshapes_paper
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

# GMM-BC (paper-exact transformer, obs_stride 8, chunk_len 8) trained FRESH on
# pushshapes_paper (the SAME fixed-target boundary task the flow-RL bootstrap
# ckpt hptC200Retrain used). minmax norm (REQUIRED for GMM tanh means).
# Default model scheduler = warmup->cosine peak 1e-4 (max_steps 90000 = 1800x50).
# This model -> exact GMM log-probs -> standard PPO on top for >0.9 coverage.
NAME=gmmBCpaper
DESC=gmm_bc_paper
# pushshapes_paper = 200 episodes -> ONLY 13 packed batches/epoch (limit_train_batches=50
# is capped by the data). So real optimizer steps = EPOCHS*13. The tx_cos default
# scheduler (max_steps=90000, warmup=4500) assumed 50 batches/epoch -> WRONG horizon
# here (346-epoch warmup, cosine never anneals). Set the cosine to the REAL step count.
EPOCHS=1800; VALEVERY=100; LTB=50
BPE=13                       # measured packed batches/epoch on pushshapes_paper
MAXSTEPS=$((EPOCHS*BPE))     # 23400
WARMUP=$((MAXSTEPS/20))      # 5% = 1170

# ---- requeue-safe resume hook (scavenger preemption restarts the script) ----
LOGDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/${NAME}
RESUME=""
LAST=$(ls -t ${LOGDIR}/*/checkpoints/last.ckpt 2>/dev/null | head -1)
if [ -n "$LAST" ]; then RESUME="ckpt_path='$LAST'"; echo "RESUMING from $LAST"; fi

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train data=tsimulation \
  model=bc_rnn_pushshapes_paperexact_tx_chunk8 \
  evaluator=eval_hnet_sim \
  evaluator.rollout_mode=ar \
  evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.precision=32 trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=${LTB} trainer.limit_val_batches=4 trainer.check_val_every_n_epoch=${VALEVERY} \
  +trainer.gradient_clip_val=150 \
  ++model.scheduler.max_steps=${MAXSTEPS} \
  ++model.scheduler.warmup_steps=${WARMUP} \
  trainer.profiler=null logger=wandb \
  data.train_dataloader_params.pushshapes_sim.batch_size=16 \
  data.train_dataloader_params.pushshapes_sim.num_workers=12 \
  +data.train_dataloader_params.pushshapes_sim.persistent_workers=true \
  +data.train_dataloader_params.pushshapes_sim.prefetch_factor=4 \
  +data.train_dataloader_params.pushshapes_sim.pin_memory=true \
  data.valid_dataloader_params.pushshapes_sim.batch_size=16 \
  data.valid_dataloader_params.pushshapes_sim.num_workers=4 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$PAPER \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$PAPER \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax \
  $RESUME
echo "TRAIN_EXIT=$?"
