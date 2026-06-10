#!/bin/bash
#SBATCH --job-name=bcrnnHnetCoBigSmall
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact-2/logs/sbatch/bcrnnHnetCoBigSmall_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact-2/logs/sbatch/bcrnnHnetCoBigSmall_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact-2
source /coc/flash7/paphiwetsa3/projects/EgoVerse-pact-2/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 SDL_VIDEODRIVER=dummy
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200

# Cross-embodiment COTRAIN: BC-RNN-style H-Net (obs-as-input, full-episode
# teacher forcing, GMM head) trained jointly on the regular big circle and the
# small circle, both living in /coc/flash7/paphiwetsa3/datasets/circle_co_big_small
# (split by episode-name filter -> pushshapes_sim (big, emb 15) +
# pushshapes_sim_small_circle (small, emb 17)). Per-embodiment OUTER chunker,
# shared INNER chunker + ComputeStage trunk.
#
# EVAL = closed-loop sim coverage on the SMALL circle ONLY (pusher_shape=
# circle_small, emb_id=17), init_mode=seeds, 20 seeds, max_steps=400,
# coverage_threshold=0.8. circle_small support was ported into this checkout's
# Tsimulation (PUSHER_RADII + render path).
#
# norm_stats: minmax, fresh full-data stats (no precomputed_norm_path).

SMOKE=${SMOKE:-0}
DESC=${DESC:-bc_rnn_hnet_cotrain_bigsmall}
NAME=${NAME:-bcrnnHnetCoBigSmall}
SEEDS20="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]"

if [ "$SMOKE" = "1" ]; then
  NAME=bcrnnHnetCoBigSmall_smoke; EPOCHS=2; VALEVERY=2; LTB=6; LOGGER=csv
  MAXSTEPS=20; EVALSEEDS="[0,1]"; NEVAL=2; MAXSTEPSARG=90000; export WANDB_MODE=disabled
else
  EPOCHS=${EPOCHS:-1800}; VALEVERY=${VALEVERY:-100}; LTB=${LTB:-50}; LOGGER=wandb
  MAXSTEPS=400; EVALSEEDS="$SEEDS20"; NEVAL=20; MAXSTEPSARG=90000; export WANDB_MODE=online
fi

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train \
  data=tsimulation_cotrain_small_circle \
  model=hnet_cotrain_gmm_obs \
  model.scheduler.max_steps=${MAXSTEPSARG} \
  evaluator=eval_hnet_sim \
  ++evaluator.embodiment_name=pushshapes_sim_small_circle \
  ++evaluator.eval_embodiment_id=17 \
  ++evaluator.env_kwargs.pusher_shape=circle_small \
  evaluator.init_mode=seeds \
  evaluator.init_seeds=${EVALSEEDS} \
  ++evaluator.n_eval_episodes=${NEVAL} \
  evaluator.max_steps=${MAXSTEPS} evaluator.coverage_threshold=0.8 \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.precision=32 \
  trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=${LTB} trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=${VALEVERY} \
  trainer.profiler=null logger=${LOGGER} \
  data.train_dataloader_params.pushshapes_sim.batch_size=8 \
  data.train_dataloader_params.pushshapes_sim_small_circle.batch_size=8 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=8 \
  data.valid_dataloader_params.pushshapes_sim_small_circle.batch_size=8 \
  norm_stats.norm_mode=minmax
echo "TRAIN_EXIT=$?"
