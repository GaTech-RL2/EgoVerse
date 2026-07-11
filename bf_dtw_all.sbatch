#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --requeue
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=. MUJOCO_GL=egl
PY=.venv/bin/python
DS=/coc/flash7/paphiwetsa3/datasets
mkdir -p eval_bf/dtw
for ARM in bf_prdec_ctrl bf_prdec_dec1 bf_prdec_sig bf_prdec_sig_var4 bf_prdec_var4 bf_symw1_gmm bf_symw1_reg bf_cvae_zs_bcrecipe; do
  D=$(ls -dt logs/indomain_c4/${ARM}_2026-07-10_1[4-9]* 2>/dev/null | head -1)
  CK=$(ls -t $D/checkpoints/last.ckpt 2>/dev/null | head -1)
  [ -z "$CK" ] && { echo "== $ARM: NO CKPT"; continue; }
  CFG=eval_bf/dtw/${ARM}_dtwcfg.yaml
  $PY - "$D/.hydra/config.yaml" "$CFG" "$DS" <<'PYCFG'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
ds = sys.argv[3]
for split in ("train_datasets", "valid_datasets"):
    for k, v in c["data"].get(split, {}).items():
        fp = v["resolver"]["folder_path"]
        v["resolver"]["folder_path"] = ds + ("/probe_sub_small" if "small" in fp else "/probe_sub_circle")
for split in ("train_dataloader_params", "valid_dataloader_params"):
    for k, v in c["data"].get(split, {}).items():
        v["batch_size"] = 20; v["num_workers"] = 2
yaml.safe_dump(c, open(sys.argv[2], "w"))
PYCFG
  echo "== $ARM ($(basename $CK) @ $(basename $D))"
  $PY scripts/bf_dtw_latent.py --ckpt "$CK" --config-path "$CFG" --n-episodes 20 --evr 0.95 --good-eps 3,7,0 2>&1 | grep -E "bfdtw|=|DONE|Error|Traceback" | head -14
done
echo BF_DTW_SWEEP_DONE
