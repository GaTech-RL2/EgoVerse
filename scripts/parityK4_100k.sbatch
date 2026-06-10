#!/bin/bash
#SBATCH -A rl2-lab
#SBATCH -p rl2-lab
#SBATCH --qos=short
#SBATCH --time=47:59:59
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -J parityK4_100k
#SBATCH -o /coc/flash7/paphiwetsa3/projects/EgoVerse-pact-2/logs/sbatch/parityK4_100k_%j.out

# K=4 pureDF parity replica — run 1 of the port-parity campaign.
# Recipe: dfot_pushshapes_pixel_policy_parity + tsimulation_k4 (see yaml
# headers for the D-number rationale). Judge by the OVERLAY eval, not sim
# (checklist §4); 100k optimizer steps ≈ 1112 epochs at ~90 packed batches.
# Smoke-verified 2026-06-10 (logs/debug/parity_k4_smoke_2026-06-10_04-32-54).

REPO=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact-2
cd "$REPO"
# CRITICAL: the shared venv's editable egomimic points at EgoVerse-pact
# (parity probe finding 2026-06-10) — PYTHONPATH must pin this repo.
export PYTHONPATH="$REPO"

.venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=dfot_parity description=parityK4_100k mode=train \
  data=tsimulation_k4 \
  model=dfot_pushshapes_pixel_policy_parity \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle \
  trainer.gradient_clip_val=10.0 \
  norm_stats.norm_mode=minmax \
  norm_stats.precomputed_norm_path=$REPO/egomimic/hydra_configs/data/norm_stats_pushshapes_fixed512.json \
  reject_outliers=false \
  evaluator=eval_dfot_pixel_overlay \
  callbacks=checkpoints \
  callbacks.model_checkpoint.every_n_epochs=139 \
  trainer.max_epochs=1112 trainer.min_epochs=1112 \
  trainer.check_val_every_n_epoch=139 \
  trainer.limit_val_batches=4 \
  logger=csv_wandb
