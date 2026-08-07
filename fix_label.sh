#!/bin/bash
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
E=$(.venv/bin/python - <<'PYEOF'
import torch
print(int(torch.load("eval_bf/bf_cvae_zs_var4_ep_snap.ckpt", map_location="cpu", weights_only=False)["epoch"]))
PYEOF
)
echo "EPOCH=$E"
mv eval_bf/bf_cvae_zs_var4_ep_snap.ckpt "eval_bf/bf_cvae_zs_var4_ep${E}_snap.ckpt"
mv chunkviz_bf/cvae_zs_var4_ep.npz "chunkviz_bf/cvae_zs_var4_ep${E}.npz"
PYTHONPATH=. .venv/bin/python -m egomimic.eval.explorer.build_html \
  --model "cvae_zs_var4@ep${E}=chunkviz_bf/cvae_zs_var4_ep${E}.npz" \
  --model "prdec_var4@ep999=chunkviz_bf/prdec_var4_ep999v2.npz" \
  --model "prdec_var4@ep1499=chunkviz_bf/prdec_var4_ep1499.npz" \
  --out chunkviz_bf/var4_family.html && echo REBUILD_OK
