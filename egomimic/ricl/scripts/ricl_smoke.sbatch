#!/bin/bash
#SBATCH --job-name=ricl_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --qos=embers
#SBATCH -A gts-dxu345-rl2
#SBATCH --time=01:00:00
#SBATCH --output=logs/ricl_smoke/%j.out
#SBATCH --error=logs/ricl_smoke/%j.err

set -uo pipefail
mkdir -p logs/ricl_smoke

# EgoVerse env (valid: openpi/flax/torch + pi05 checkpoint), EgoVerse2 ricl code on path
source /storage/project/r-dxu345-0/rco3/EgoVerse/emimic/bin/activate
export PYTHONPATH=/storage/project/r-dxu345-0/rco3/EgoVerse2
cd /storage/project/r-dxu345-0/rco3/EgoVerse2

echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import openpi, egomimic; print('openpi', openpi.__file__); print('egomimic', egomimic.__file__)"

python -m egomimic.trainHydra --config-name train_zarr_cartesian \
  model=pi0.5_ricl data=cotrain_pi_ricl_smoke evaluator=eval_pi_ricl trainer=ddp_pi \
  launch_params.gpus_per_node=1 trainer.devices=1 trainer.num_nodes=1 \
  +trainer.fast_dev_run=true \
  data.retrieval_cache_dir=/storage/project/r-dxu345-0/rco3/EgoVerse2/egomimic/ricl/outputs/ricl_cache_smoke \
  "data.bank_zarr_root='/storage/project/r-dxu345-0/agao81/pick_place/{hash}'" \
  model.robomimic_model.tokenizer_model_name=/storage/project/r-dxu345-0/rco3/EgoVerse2/egomimic/ricl/pg_tokenizer \
  data.train_dataloader_params.eva_bimanual.batch_size=2 \
  data.valid_dataloader_params.eva_bimanual.batch_size=2 \
  data.train_dataloader_params.eva_bimanual.num_workers=2 \
  data.valid_dataloader_params.eva_bimanual.num_workers=2 \
  norm_stats.num_workers=2 norm_stats.sample_frac=0.5 \
  logger=debug name=ricl_smoke description=fastdevrun

echo "=== trainHydra exit: $? ==="
