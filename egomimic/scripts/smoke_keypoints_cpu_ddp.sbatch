#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --job-name=smoke_kp_cpu_ddp
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/smoke_keypoints_cpu_ddp_%j.log

# CPU smoke test (1-process DDP on CPU: the codebase calls torch.distributed
# unconditionally, so trainer=default without a process group dies) of the
# pi0.5 KEYPOINT pipeline (every GPU partition is days
# deep): 2 episodes per split, batch 2, fp32, 2 train steps + 1 val batch
# through eval_pi_keypoints_wristframe (revert + metrics + overlay viz) and the
# train_viz head, from the pi05 base weights with the action projections
# re-initialized at action_dim 140. In-run norm stats (0.05 frac of 2 eps).

set -euo pipefail
WT=/storage/project/r-dxu345-0/agao81/EgoVerse/.claude/worktrees/fold-freeform-kp
REPO=/storage/project/r-dxu345-0/agao81/EgoVerse
cd "$WT"
export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24

echo "[smoke] cwd=$(pwd) commit=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
"$PY" -u egomimic/trainHydra.py \
  --config-name=train_zarr_cartesian_pi \
  data=mecka_fold_freeform_opsplit_pi_keypoints \
  model=pi0.5_bc_mecka_keypoints \
  evaluator=eval_pi_keypoints_wristframe \
  +evaluator@train_viz_evaluator=train_viz_pi_keypoints_wristframe \
  trainer=ddp_pi trainer.accelerator=cpu trainer.devices=1 trainer.num_nodes=1 trainer.strategy=ddp_find_unused_parameters_true trainer.sync_batchnorm=false trainer.precision=32 \
  trainer.max_epochs=1 trainer.min_epochs=1 ++trainer.max_steps=2 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=1 trainer.check_val_every_n_epoch=1 \
  trainer.num_sanity_val_steps=0 \
  model.robomimic_model.config.pytorch_training_precision=float32 \
  model.robomimic_model.config.reverse_kl_samples=1 \
  model.optimizer.lr=5e-5 model.scheduler=null \
  model.robomimic_model.config.pytorch_weight_path=/storage/project/r-dxu345-0/agao81/EgoVerse/egomimic/algo/pi_checkpoints/pi05_base_pytorch \
  +data.train_datasets.human_bimanual.debug=2 +data.valid_datasets.human_bimanual.debug=2 +data.train_viz_datasets.human_bimanual.debug=2 \
  data.train_dataloader_params.human_bimanual.batch_size=2 data.valid_dataloader_params.human_bimanual.batch_size=2 data.train_viz_dataloader_params.human_bimanual.batch_size=2 \
  data.train_dataloader_params.human_bimanual.num_workers=2 data.valid_dataloader_params.human_bimanual.num_workers=2 data.train_viz_dataloader_params.human_bimanual.num_workers=1 \
  norm_stats.sample_frac=0.05 norm_stats.num_workers=4 \
  evaluator.viz_max_batches=1 evaluator.viz_every_n_epochs=1 train_viz_evaluator.limit_val_batches=1 \
  name=smoke_keypoints_cpu description=cpu_2steps_ddp1_fup \
  'hydra.run.dir=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/${name}/${description}_${now:%Y-%m-%d_%H-%M-%S}'
echo "[smoke] exit=$?"
