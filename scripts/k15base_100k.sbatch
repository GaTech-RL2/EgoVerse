#!/bin/bash
#SBATCH -A rl2-lab
#SBATCH -p rl2-lab
#SBATCH --qos=short
#SBATCH --time=47:59:59
#SBATCH --gres=gpu:a40:1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -J k15base_100k
#SBATCH -o /coc/flash7/paphiwetsa3/projects/EgoVerse-pact-2/logs/sbatch/k15base_100k_%j.out

# K15nf20 COUPLED baseline — the port-parity canary that should reproduce
# upstream's 0.77 coverage (action_stack=15 is the proven driver; K4=0.00,
# K8=0.14, K15nf20=0.77 in upstream's own eval logs). pixel_mode=policy =
# joint (coupled) obs+action noise = standard DFoT. eta=0.0 (corrected sim
# default baked into the yaml). Smoke-verified 2026-06-15
# (logs/debug/k15_smoke_2026-06-15_16-18-04: finite loss, fits 40G, no OOM).
# In-training monitor = gt-history overlay (fair per-step signal, n_ctx=19).

REPO=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact-2
cd "$REPO"
export PYTHONPATH="$REPO"   # venv editable egomimic points at EgoVerse-pact otherwise

.venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=dfot_parity description=k15base_100k mode=train \
  data=tsimulation_k15 \
  model=dfot_pushshapes_pixel_policy_K15 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle \
  trainer.gradient_clip_val=10.0 \
  norm_stats.norm_mode=minmax \
  norm_stats.precomputed_norm_path=$REPO/egomimic/hydra_configs/data/norm_stats_pushshapes_fixed512.json \
  reject_outliers=false \
  evaluator=eval_dfot_pixel_overlay \
  evaluator.evals.0.n_context_frames=19 \
  evaluator.evals.0.action_history=gt \
  callbacks=checkpoints \
  callbacks.model_checkpoint.every_n_epochs=139 \
  trainer.max_epochs=1112 trainer.min_epochs=1112 \
  trainer.check_val_every_n_epoch=139 \
  trainer.limit_val_batches=4 \
  logger=csv_wandb
