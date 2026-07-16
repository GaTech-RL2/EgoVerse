#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --exclude=ig-88,spd-13,sonny,cyborg,hk47,heistotron,baymax,deebot,kitt,cheetah,megazord,puma,omgwth,megabot
#SBATCH --output=bf_chunkviz_reverdict_%j.out
# Re-probe argmax organization on arms condemned before ep2000 (fence /
# degenerate-chunker / STE-negative verdicts). One export per RUN:EP:TAG.
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=. MUJOCO_GL=egl
PY=.venv/bin/python
SPECS=(
  "bf_symw1_gmm_2026-07-11_23-06-12:2499:symw1_gmm"
  "bf_prdec_var4_2026-07-10_15-23-17:2999:prdec_var4"
  "bf_prdec_sig_var4_2026-07-10_15-23-20:2999:sig_var4"
  "bf_prdec_ste8_2026-07-13_01-59-08:1999:ste8"
  "bf_prdec_ste4_499_2026-07-13_01-11-35:1999:ste4"
  "bf_prdec_regoff999_2026-07-13_01-11-35:1499:regoff999"
  "bf_cvae_zs_do1_2026-07-11_23-50-40:1999:cvae_zs_do1"
  "bf_symw1_gmm_do1_2026-07-10_23-20-53:2999:symw1_gmm_do1"
)
NPZS=""
for SPEC in "${SPECS[@]}"; do
  RUN=${SPEC%%:*}; REST=${SPEC#*:}; EP=${REST%%:*}; TAG=${REST#*:}
  CK="logs/indomain_c4/${RUN}/checkpoints/epoch_epoch=${EP}.ckpt"
  [ -s "$CK" ] || { echo "MISS $TAG ep$EP"; continue; }
  SNAP=eval_bf/reverdict_${TAG}_ep${EP}.ckpt
  [ -s "$SNAP" ] || cp "$CK" "$SNAP"
  TRUE=$($PY -c "import torch;print(torch.load('$SNAP',map_location='cpu',weights_only=False)['epoch'])")
  echo "SNAP ${TAG} ep${EP} true_epoch=${TRUE}"
  OUT=chunkviz_bf/reverdict_${TAG}_ep${EP}.npz
  $PY -m egomimic.eval.explorer.export --ckpt "$SNAP" \
    --config-path "logs/indomain_c4/${RUN}/.hydra/config.yaml" \
    --out "$OUT" --n-episodes 10
  echo "EXPORT_${TAG}_ep${EP}_rc=$?"
  [ -s "$OUT" ] && NPZS="$NPZS $OUT"
done
$PY chunklen_report.py $NPZS
echo DONE
