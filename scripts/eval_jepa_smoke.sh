#!/bin/bash
#SBATCH --job-name=hptJepaSmoke
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/jepasmoke_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/jepasmoke_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled

# Reuse the c3000 act8 norm_stats (shape-8 actions, circle_3000) so the smoke
# skips the slow norm pass. Static file (written once at run start) -> safe to read.
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowC3000CausalAct8/hpt_flow_circle_3000_causal_obs1_act8_2026-06-14_01-39-26/norm_stats/norm_stats.json

srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hptJepaSmoke description=jepa_smoke_2ep mode=train \
  data=tsimulation_hpt_causal_jepa model=hpt_pushshapes_circle_jepa \
  trainer=debug trainer.max_epochs=2 trainer.limit_train_batches=4 \
  trainer.check_val_every_n_epoch=100 trainer.limit_val_batches=1 \
  trainer.profiler=null logger=csv \
  data.train_dataloader_params.pushshapes_sim.batch_size=16 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=16 \
  data.train_dataloader_params.pushshapes_sim.num_workers=6 \
  data.valid_dataloader_params.pushshapes_sim.num_workers=6 \
  norm_stats.norm_mode=quantile \
  norm_stats.precomputed_norm_path=$NORM
echo "SMOKE_EXIT=$?"
