#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --output=bf_rebuild_orghtml_%j.out
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=.
PY=.venv/bin/python
M=""
for EP in 999 1499 1999 2499 2999; do
  F=chunkviz_bf/nopre_gmm_ep${EP}.npz
  [ -s "$F" ] && M="$M --model nopre_gmm@ep${EP}=$F" || echo "SKIP ep${EP} (no npz)"
done
$PY -m egomimic.eval.explorer.build_html $M --out chunkviz_bf/nopre_gmm_organization.html
echo BUILD_rc=$?
echo DONE
