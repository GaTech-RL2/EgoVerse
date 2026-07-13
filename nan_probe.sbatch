#!/bin/bash
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --qos=short
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
.venv/bin/python - <<'PYEOF'
import torch, torch.nn as nn
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra import compose, initialize_config_dir
import os
cfgdir = os.path.abspath("egomimic/hydra_configs")
with initialize_config_dir(version_base=None, config_dir=cfgdir):
    cfg = compose(config_name="train_zarr_cartesian",
                  overrides=["+experiment=indomain_c4/dualstream_cotrain_2trunk_d16_200m",
                             "model=bf_prdec_sig",
                             "norm_stats.precomputed_norm_path=/coc/flash7/paphiwetsa3/projects/EgoVerse-gmm-dualstream/logs/indomain_c4/dualstream_cotrain_m16_sym_2026-06-24_15-12-49/norm_stats/norm_stats.json"])
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
dm = instantiate(cfg.data); dm.setup(stage="fit")
batch = next(iter(dm.train_dataloader()))
batch = batch[0] if isinstance(batch, tuple) else batch
norm = MultiDataset.from_state(dm.norm_stats_state) if hasattr(dm, "norm_stats_state") else None
algo = instantiate(cfg.model.robomimic_model, norm_stats=dm.norm_stats if hasattr(dm, "norm_stats") else norm)
algo.nets.cuda(); algo.nets.train()
pb = algo.process_batch_for_training(batch)
def cu(x):
    return {k: cu(v) for k, v in x.items()} if isinstance(x, dict) else (x.cuda() if torch.is_tensor(x) else x)
pb = cu(pb)
first_nan = []
def hook(name):
    def f(mod, inp, out):
        def chk(t, tag):
            if torch.is_tensor(t) and t.is_floating_point() and torch.isnan(t).any():
                if len(first_nan) < 8:
                    first_nan.append(f"{name} {tag} shape={tuple(t.shape)}")
        outs = out if isinstance(out, tuple) else (out,)
        for i, o in enumerate(outs):
            chk(o, f"out{i}")
            if hasattr(o, "boundary_prob"): chk(o.boundary_prob, "bprob")
    return f
for n, m in algo.nets.named_modules():
    m.register_forward_hook(hook(n))
with torch.autocast("cuda", dtype=torch.bfloat16):
    try:
        preds = algo.forward_training(pb)
        print("forward ok:", {k: float(v) for k, v in list(preds.items())[:3] if hasattr(v, "__float__")})
    except Exception as e:
        print("forward raised:", type(e).__name__, str(e)[:200])
print("FIRST NAN SITES:")
for x in first_nan: print("  ", x)
if not first_nan: print("   none")
PYEOF
echo NANPROBE_DONE rc=$?
