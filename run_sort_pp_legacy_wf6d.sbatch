#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=257875M
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90
#SBATCH -J sort_pp_legacy_wf6d
#SBATCH -o %x_%j.out

# Training launcher: sort pick-place LEGACY wristframe 6D.
# Self-contained (no submitit / hydra --multirun) so the job does not depend on a
# long-lived parent process. Submit with:
#   sbatch run_sort_pp_legacy_wf6d.sbatch
# Reconstructed from logs/sort_pp_legacy_wf6d/.../.hydra/overrides.yaml (job 10053188).

cd /storage/project/r-dxu345-0/agao81/EgoVerse
set -a; source ./apikey.txt; set +a
# Per-job, node-local inductor cache (NOT a shared NFS dir) to avoid corrupt-JSON /
# stale-file-handle crashes from concurrent writers. SLURM_JOB_ID is stable across
# requeue, so a requeued job reuses its cache.
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
VENV=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
WEIGHTS=/storage/project/r-dxu345-0/agao81/EgoVerse/egomimic/algo/pi_checkpoints/pi05_base_pytorch

srun $VENV egomimic/trainHydra.py --config-name train_zarr_cartesian_pi \
  model=pi0.5_cotrain_eva_aria_6d data=cotrain_sort_pp_lang_6d evaluator=eval_pi_wristframe_6d \
  launch_params.gpus_per_node=4 \
  model.robomimic_model.config.pytorch_weight_path="$WEIGHTS" \
  norm_stats.sample_frac=0.1 \
  +data.train_dataloader_params.eva_bimanual.timeout=600 \
  +data.train_dataloader_params.aria_bimanual.timeout=600 \
  +data.valid_dataloader_params.eva_bimanual.timeout=600 \
  +data.valid_dataloader_params.aria_bimanual.timeout=600 \
  name=sort_pp_legacy_wf6d description="sort legacy wristframe 6d"
