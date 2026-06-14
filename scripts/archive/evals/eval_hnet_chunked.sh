#!/bin/bash
#SBATCH --job-name=eval-chunked
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/eval_chunked_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/eval_chunked_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl WANDB_MODE=disabled HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
RUNDIR=${RUNDIR_OVERRIDE:-logs/hnet_baseline_nc3/hnet_chunked_test_80ep_2026-06-01_04-02-08}
CKPT=${CKPT_OVERRIDE:-$RUNDIR/checkpoints/last.ckpt}
NORM=$RUNDIR/norm_stats/norm_stats.json
DESC=${DESC_OVERRIDE:-hnet_chunked}
srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=reeval_corrected description=re_${DESC} \
  mode=eval \
  model=hnet_pushshapes_chunktoken_hptfull \
  ++model.robomimic_model.backbone=${BACKBONE:-hnet_chunked} \
  ++model.robomimic_model.stem=none \
  ++model.robomimic_model.readout=${READOUT:-mean_pool} \
  data=tsimulation \
  evaluator=eval_hnet_sim \
  evaluator.rollout_mode=chunk_te evaluator.temporal_ensemble=true evaluator.chunk_k=32 \
  ckpt_path=$CKPT \
  trainer=debug trainer.devices=1 trainer.limit_val_batches=4 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
  norm_stats.precomputed_norm_path=$NORM
echo "EVAL_EXIT=$?"
