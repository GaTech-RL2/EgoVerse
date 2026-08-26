#!/bin/bash
#SBATCH --job-name=bcrnnTx10_c3000pg
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --qos=long
#SBATCH --time=6-23:59:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse-dfot-stack/logs/sbatch/bcrnn_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse-dfot-stack/logs/sbatch/bcrnn_%x_%j.err
set -uxo pipefail
REPO=/coc/flash7/paphiwetsa3/projects/EgoVerse-dfot-stack
cd $REPO
# NOTE: this repo's own .venv symlink (-> ../EgoVerse-pact/.venv) is BROKEN
# (EgoVerse-pact was removed). Use EgoVerse7's venv — the SAME venv the proven
# EgoVerse2 c3000v2 launchers sourced. Its egomimic editable-install points at
# EgoVerse7, so PYTHONPATH MUST be this repo (absolute) to shadow it.
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=$REPO
export MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
# Guard: verify we run THIS repo's egomimic, not the venv's editable install.
python -c "import egomimic, sys; p=egomimic.__file__; assert 'EgoVerse-dfot-stack' in p, f'WRONG egomimic: {p}'; print('egomimic OK:', p)" || exit 1
DATA=/coc/flash7/paphiwetsa3/datasets/circle_3000_plus_gen
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

# bcrnnTx10_c3000pg: the WINDOW-10 CLASSIC BC-RNN-Transformer (robomimic-style
# 10-step window / hidden-reinit, per-step GMM) on circle_3000_plus_gen —
# controlled comparison arm vs the packed full-episode Tx (~0.4 coverage
# plateau). Recipe cloned from the proven train_bc_rnn_tx.sh (model=
# bc_rnn_pushshapes_paperexact_tx, constant LR 1e-4, EPOCHS=1800 x LTB=50,
# batch 16, fp32, minmax fresh full-data stats) with EXACTLY ONE deviation:
#   pad_mode=repeat_pusher_unmasked (USER DIRECTIVE 2026-07): window tails
#   past the episode end pad the ACTION target with the episode's LAST PUSHER
#   POSITION (state_agent_obj[:2] at the last real frame, unnormalized with
#   the proprio stats and re-normalized with the ACTION minmax stats) instead
#   of repeating the last recorded CURSOR action (which sits ~5-10px off the
#   pusher and would teach a persistent offset push). Obs tail + unmasked NLL
#   are unchanged from repeat_unmasked.

SMOKE=${SMOKE:-0}
DESC=${DESC:-bc_rnn_tx10_c3000pg_holdpad}
NAME=${NAME:-bcrnnTx10_c3000pg}
if [ "$SMOKE" = "1" ]; then
  NAME=bcrnnTx10_c3000pg_smoke; EPOCHS=2; VALEVERY=2; LTB=6; LOGGER=csv; MAXSTEPS=20; SEEDS="evaluator.init_seeds=[0,1]"; export WANDB_MODE=disabled
else
  EPOCHS=${EPOCHS:-1800}; VALEVERY=${VALEVERY:-100}; LTB=${LTB:-50}; LOGGER=csv_wandb; MAXSTEPS=400; SEEDS=""; export WANDB_MODE=online
fi

# Requeue-safe resume hook (mirrors the EgoVerse2 c3000v2 launcher): on SLURM
# auto-requeue pick up the newest last.ckpt for THIS run name so it does NOT
# restart at ep0. Gated on a marker file so the FIRST launch is always fresh.

# Precomputed norm stats (vault mandate): pass NORM_JSON=<path/to/norm_stats.json>
# to skip the 30-60 min full-data stats pass (json must be minmax stats computed
# on THIS dataset, e.g. from the validated smoke run). Empty -> fresh compute.
PRECOMP=""
[ -n "${NORM_JSON:-}" ] && PRECOMP="norm_stats.precomputed_norm_path=${NORM_JSON}"

RESUME=""
if [ "$SMOKE" != "1" ]; then
  MARK=logs/${NAME}/.launched
  if [ -f "$MARK" ]; then
    LAST=$(ls -t $REPO/logs/${NAME}/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LAST" ] && RESUME="ckpt_path=$LAST"
  fi
  mkdir -p logs/${NAME} && touch "$MARK"
fi

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train data=tsimulation \
  model=bc_rnn_pushshapes_paperexact_tx \
  model.robomimic_model.pad_mode=repeat_pusher_unmasked \
  evaluator=eval_hnet_sim \
  evaluator.max_steps=${MAXSTEPS} evaluator.coverage_threshold=0.8 ${SEEDS} \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.precision=32 trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=${LTB} trainer.limit_val_batches=4 trainer.check_val_every_n_epoch=${VALEVERY} \
  trainer.profiler=null logger=${LOGGER} \
  data.train_dataloader_params.pushshapes_sim.batch_size=16 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=16 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax ${PRECOMP} ${RESUME}
echo "TRAIN_EXIT=$?"
