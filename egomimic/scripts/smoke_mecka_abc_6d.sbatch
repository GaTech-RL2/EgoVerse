#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --job-name=smoke_mecka_abc
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/smoke_mecka_abc_%j.log

# End-to-end smoke for the COMBINED mecka+abc pretrain path: 16-episode
# subset of EACH corpus, trainer=debug (4 epochs, val@2, 2 train/3 val
# batches), viz forced at epoch 3 -> exercises the two-key CombinedLoader
# (1:1 human/eva batches per step), the merged norm stats (ids 3+6), the
# mixed-resolution collate resize on the eva frames, both converters'
# forward/backward, all val metrics (incl. RKL M=4) for both embodiments,
# the train_viz head, and both wristframe reverts + video render. Fresh
# init from the pi05 base weights — no init_weights_ckpt, matching the
# real pretrain.

set -euo pipefail
cd /storage/project/r-dxu345-0/agao81/EgoVerse
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# NORM_PATH knob: point at the debug merged stats (mecka_abc_6d_debug) to
# smoke the foldviz path without waiting for the full norm recompute.
NORM_PATH="${NORM_PATH-/storage/project/r-dxu345-0/agao81/norm_stats/mecka_abc_6d/norm_stats}"

"$PY" -u egomimic/trainHydra.py \
  --config-name=train_zarr_cartesian_pi \
  data=mecka_abc_pi_6d \
  model=pi0.5_bc_mecka_abc_6d \
  evaluator=eval_pi_wristframe_6d \
  +evaluator@train_viz_evaluator=train_viz_pi_wristframe_6d \
  trainer=debug \
  launch_params.gpus_per_node=1 \
  evaluator.viz_every_n_epochs=4 \
  "++data.train_datasets.human_bimanual.resolver.debug=16" \
  "++data.train_datasets.eva_bimanual.resolver.debug=16" \
  "data.train_datasets.human_bimanual.valid_ratio=0.25" \
  "data.train_datasets.eva_bimanual.valid_ratio=0.25" \
  "norm_stats.precomputed_norm_path=$NORM_PATH" \
  model.robomimic_model.config.pytorch_weight_path=/storage/project/r-dxu345-0/agao81/EgoVerse/egomimic/algo/pi_checkpoints/pi05_base_pytorch \
  name=smoke_mecka_abc \
  description=smoke_combined_mecka_abc_pretrain_path
