#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=00:25:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --requeue
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
.venv/bin/python - <<'PYEOF'
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra import compose, initialize_config_dir
import os
import egomimic.models.hnet.routing as R
print("HAS_MAMBA_SCAN =", getattr(R, "_HAS_MAMBA_SCAN", "?"))

cfgdir = os.path.abspath("egomimic/hydra_configs")
def build(model_name):
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="train_zarr_cartesian",
                      overrides=[f"+experiment=indomain_c4/dualstream_cotrain_2trunk_d16_200m",
                                 f"model={model_name}",
                                 "norm_stats.precomputed_norm_path=/coc/flash7/paphiwetsa3/projects/EgoVerse-gmm-dualstream/logs/indomain_c4/dualstream_cotrain_m16_sym_2026-06-24_15-12-49/norm_stats/norm_stats.json"])
    return cfg
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
_ck = torch.load("bf_prdec_mapped624.ckpt", map_location="cpu", weights_only=False)
norm = MultiDataset.from_state(_ck["hyper_parameters"]["norm_stats_state"])
cfg = build("bf_prdec_sig")
dm = instantiate(cfg.data); dm.setup(stage="fit")
batch = next(iter(dm.train_dataloader()))
batch = batch[0] if isinstance(batch, tuple) else batch

orig = R.DeChunkLayer._forward_packed
def spy(self, hidden_states, boundary_mask, boundary_prob, cu_seqlens):
    p_packed = torch.clamp(boundary_prob[..., -1][boundary_mask], min=1e-4, max=1-1e-4)
    dt = torch.log(1 / (1 - p_packed))
    print(f"    [dechunk d={hidden_states.shape[-1]}] M={hidden_states.shape[0]} "
          f"bprob_dtype={boundary_prob.dtype} p_max={p_packed.max().item():.6f} "
          f"dt_max={dt.max().item():.4f} dt_bf16_max={dt.to(torch.bfloat16).max().item():.4f} "
          f"x_nan={torch.isnan(hidden_states).any().item()}")
    out = orig(self, hidden_states, boundary_mask, boundary_prob, cu_seqlens)
    print(f"      -> out_nan={torch.isnan(out).any().item()}")
    return out
R.DeChunkLayer._forward_packed = spy

for model_name in ("bf_prdec_sig", "bf_prdec"):
    print("==== MODEL", model_name)
    cfg = build(model_name)
    algo = instantiate(cfg.model.robomimic_model, norm_stats=norm)
    algo.nets.cuda(); algo.nets.train()
    pb = algo.process_batch_for_training({k: v for k, v in batch.items()})
    def cu(x):
        return {k: cu(v) for k, v in x.items()} if isinstance(x, dict) else (x.cuda() if torch.is_tensor(x) else x)
    pb = cu(pb)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        try:
            algo.forward_training(pb)
            print("  forward OK")
        except Exception as e:
            print("  forward raised:", type(e).__name__, str(e)[:120])
    del algo; torch.cuda.empty_cache()
PYEOF
echo NANPROBE3_DONE rc=$?
