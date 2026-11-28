#!/bin/bash
#SBATCH --job-name=bgobs-rtc1221-audit
#SBATCH --partition=interactive-cpu
#SBATCH --account=rl2-lab
#SBATCH --qos=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/coc/flash7/cxin34/bgobs-rtc-audit-20261124/model-%j.log
set -euo pipefail
export PYTHONPATH=/coc/flash7/cxin34/egoa180-cart-fix-FT10k-robotpc-20261124:/coc/flash7/cxin34/egoa180-cart-fix-FT10k-robotpc-20261124/source
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
/coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python /coc/flash7/cxin34/bgobs-rtc-audit-20261124/test_model.py
