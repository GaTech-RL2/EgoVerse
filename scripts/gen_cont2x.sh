#!/bin/bash
# Generator for the 8 cont2x_<run>.sh launchers.
# Mechanism (LOGGED): weights-only +init_ckpt of each run's furthest checkpoint
# + FRESH warmup_cosine_scheduler over 1800 new epochs (max_steps=90000=1800x50,
# peak 1e-4 -> eta_min 1e-6, warmup 4500). Epoch counter restarts 0->1800.
# Chosen because full-resume (ckpt_path) restores the OLD cosine T_max=85500 from
# the checkpoint's scheduler state and IGNORES any max_steps config override, which
# would bottom the LR at ep1800 then CYCLE it back up (the documented cycling trap).
# On a SLURM requeue, the launcher switches to full ckpt_path resume of THIS run's
# own newest last.ckpt (which carries the fresh cosine), so requeue continues cleanly.
set -euo pipefail
ROOT=/coc/flash7/paphiwetsa3/projects/EgoVerse2
OUT=$ROOT/scripts
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

# Fields per run (pipe-separated):
# tag | jobname | name | desc | model | rnn_horizon | every_n | initckpt | norm | extra
read -r -d '' RUNS <<'EOF' || true
HnetC4|bcrnnHnetC4FHx2|bcrnnHnetC4FH_nc3|bc_rnn_hnet_chunk4_fullhist_cont2x|bc_rnn_pushshapes_paperexact_hnet_chunk4|160|100|logs/bcrnnHnetC4FH_nc3/bc_rnn_hnet_chunk4_fullhist_2026-06-10_00-57-36/checkpoints/epoch_epoch=399.ckpt|logs/bcrnnHnetC4FH_nc3/bc_rnn_hnet_chunk4_fullhist_2026-06-10_00-57-36/norm_stats/norm_stats.json|
HnetC8|bcrnnHnetC8FHx2|bcrnnHnetC8FH_nc3|bc_rnn_hnet_chunk8_fullhist_cont2x|bc_rnn_pushshapes_paperexact_hnet_chunk8|80|100|logs/bcrnnHnetC8FH_nc3/bc_rnn_hnet_chunk8_fullhist_2026-06-10_00-57-36/checkpoints/epoch_epoch=399.ckpt|logs/bcrnnHnetC8FH_nc3/bc_rnn_hnet_chunk8_fullhist_2026-06-10_00-57-36/norm_stats/norm_stats.json|
HnetC16|bcrnnHnetC16FHx2|bcrnnHnetC16FH_nc3|bc_rnn_hnet_chunk16_fullhist_cont2x|bc_rnn_pushshapes_paperexact_hnet_chunk16|40|100|logs/bcrnnHnetC16FH_nc3/bc_rnn_hnet_chunk16_fullhist_2026-06-10_00-57-36/checkpoints/epoch_epoch=399.ckpt|logs/bcrnnHnetC16FH_nc3/bc_rnn_hnet_chunk16_fullhist_2026-06-10_00-57-36/norm_stats/norm_stats.json|
HnetC32|bcrnnHnetC32FHx2|bcrnnHnetC32FH_nc3|bc_rnn_hnet_chunk32_fullhist_cont2x|bc_rnn_pushshapes_paperexact_hnet_chunk32|20|100|logs/bcrnnHnetC32FH_nc3/bc_rnn_hnet_chunk32_fullhist_2026-06-10_01-44-49/checkpoints/epoch_epoch=599.ckpt|logs/bcrnnHnetC32FH_nc3/bc_rnn_hnet_chunk32_fullhist_2026-06-10_01-44-49/norm_stats/norm_stats.json|
TxC4|bcrnnTxC4FHx2|bcrnnTxC4FH_nc3|bc_rnn_tx_chunk4_fullhist_cont2x|bc_rnn_pushshapes_paperexact_tx_chunk4|160|25|logs/bcrnnTxC4FH_nc3/bc_rnn_tx_chunk4_fullhist_relaunch_2026-06-10_02-54-38/checkpoints/epoch_epoch=599.ckpt|logs/bcrnnTxC4FH_nc3/bc_rnn_tx_chunk4_fullhist_relaunch_2026-06-10_02-54-38/norm_stats/norm_stats.json|+model.robomimic_model.window_anchor=start
TxC8|bcrnnTxC8FHx2|bcrnnTxC8FH_nc3|bc_rnn_tx_chunk8_fullhist_cont2x|bc_rnn_pushshapes_paperexact_tx_chunk8_fullhist|0|100|logs/bcrnnTxC8FH_nc3/bc_rnn_tx_chunk8_fullhist_2026-06-07_15-39-53/checkpoints/epoch_epoch=1699.ckpt|logs/bcrnnTxC8FH_nc3/bc_rnn_tx_chunk8_fullhist_2026-06-07_15-39-53/norm_stats/norm_stats.json|
TxC16|bcrnnTxC16FHx2|bcrnnTxC16FH_nc3|bc_rnn_tx_chunk16_fullhist_cont2x|bc_rnn_pushshapes_paperexact_tx_chunk16|40|25|logs/bcrnnTxC16FH_nc3/bc_rnn_tx_chunk16_fullhist_relaunch_2026-06-10_02-55-13/checkpoints/epoch_epoch=299.ckpt|logs/bcrnnTxC16FH_nc3/bc_rnn_tx_chunk16_fullhist_relaunch_2026-06-10_02-55-13/norm_stats/norm_stats.json|+model.robomimic_model.window_anchor=start
TxC32|bcrnnTxC32FHx2|bcrnnTxC32FH_nc3|bc_rnn_tx_chunk32_fullhist_cont2x|bc_rnn_pushshapes_paperexact_tx_chunk32|20|100|logs/bcrnnTxC32FH_nc3/bc_rnn_tx_chunk32_fullhist_2026-06-10_00-57-36/checkpoints/epoch_epoch=299.ckpt|logs/bcrnnTxC32FH_nc3/bc_rnn_tx_chunk32_fullhist_2026-06-10_00-57-36/norm_stats/norm_stats.json|+model.robomimic_model.window_anchor=start
EOF

while IFS='|' read -r tag jobname name desc model rnnh everyn initckpt norm extra; do
  [ -z "${tag:-}" ] && continue
  f="$OUT/cont2x_${tag}.sh"
  # Optional per-run hydra overrides, assembled into ONE shell var (no dangling
  # backslash-continued blank lines). TxC8 fullhist bakes rnn_horizon in -> rnnh=0 skips it.
  EXTRA_ARGS=""
  if [ "$rnnh" != "0" ]; then EXTRA_ARGS="model.robomimic_model.rnn_horizon=${rnnh} model.robomimic_model.core_net.max_window=${rnnh}"; fi
  [ -n "${extra:-}" ] && EXTRA_ARGS="${EXTRA_ARGS} ${extra}"
  cat > "$f" <<SCRIPT
#!/bin/bash
#SBATCH --job-name=${jobname}
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=ig-88
#SBATCH --output=${ROOT}/logs/sbatch/bcrnn_%x_%j.out
#SBATCH --error=${ROOT}/logs/sbatch/bcrnn_%x_%j.err
set -uxo pipefail
cd ${ROOT}
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=\${WANDB_MODE:-online}
NC3=${NC3}
KM=${KM}

# ---- CONT-2x of ${name} (weights-only init + fresh productive warmup-cosine) ----
INIT_CKPT=${ROOT}/${initckpt}
NORM=${ROOT}/${norm}
NAME=${name}_cont2x
DESC=${desc}
EPOCHS=\${EPOCHS:-1800}; VALEVERY=${everyn}; LTB=50; MAXSTEPS=400
MAXSTEPS_SCHED=\$(( EPOCHS * LTB ))   # fresh cosine horizon = new epochs x LTB
EXTRA_ARGS="${EXTRA_ARGS}"   # per-run hydra overrides (rnn_horizon / window_anchor)

# ---- REQUEUE-SAFE ckpt selection ----
# First launch: weights-only +init_ckpt (installs fresh cosine).
# After a requeue: full ckpt_path resume of THIS run's own newest last.ckpt
# (carries the fresh cosine state), so we never re-init from epoch 0.
LAST=\$(ls -t ${ROOT}/logs/\${NAME}/*/checkpoints/last.ckpt 2>/dev/null | head -1)
if [ -n "\$LAST" ]; then
  CKPT_MODE="ckpt_path='\$LAST'"
  echo "REQUEUE: full-resume from \$LAST"
else
  CKPT_MODE="+init_ckpt='\$INIT_CKPT'"
  echo "FIRST LAUNCH: weights-only init from \$INIT_CKPT"
fi

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \\
  --config-name=train_zarr_cartesian \\
  name=\${NAME} description=\${DESC} mode=train data=tsimulation \\
  model=${model} \\
  evaluator=eval_hnet_sim \\
  evaluator.rollout_mode=ar \\
  evaluator.max_steps=\${MAXSTEPS} evaluator.coverage_threshold=0.8 \\
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=\${VALEVERY} \\
  trainer=debug trainer.precision=32 trainer.max_epochs=\${EPOCHS} trainer.min_epochs=\${EPOCHS} \\
  trainer.limit_train_batches=\${LTB} trainer.limit_val_batches=4 trainer.check_val_every_n_epoch=\${VALEVERY} \\
  trainer.profiler=null logger=wandb \\
  data.train_dataloader_params.pushshapes_sim.batch_size=16 \\
  data.valid_dataloader_params.pushshapes_sim.batch_size=16 \\
  data.train_datasets.pushshapes_sim.resolver.folder_path=\$NC3 \\
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=\$KM \\
  data.valid_datasets.pushshapes_sim.resolver.folder_path=\$NC3 \\
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=\$KM \\
  norm_stats.norm_mode=minmax \\
  norm_stats.precomputed_norm_path=\$NORM \\
  model.scheduler._target_=egomimic.utils.schedulers.warmup_cosine_scheduler \\
  model.scheduler.max_steps=\${MAXSTEPS_SCHED} \\
  model.scheduler.warmup_steps=4500 \\
  model.scheduler.warmup_start_factor=0.1 \\
  model.scheduler.eta_min=1.0e-6 \\
  \$EXTRA_ARGS \\
  \$CKPT_MODE
echo "TRAIN_EXIT=\$?"
SCRIPT
  chmod +x "$f"
  echo "wrote $f"
done <<< "$RUNS"
