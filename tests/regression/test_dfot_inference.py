"""Smoke: refactored DFoT inference paths (closed-loop AR + chunk) still work
via the property accessors."""
import pytest  # noqa: E402

pytest.skip(
    "manual regression smoke: hardcoded EgoVerse-clone-3 paths + configs "
    "removed from this repo + needs GPU/checkpoints; run directly with "
    "python, skipped under pytest collection",
    allow_module_level=True,
)

import sys, torch, numpy as np
sys.path.insert(0, "/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3")
from omegaconf import OmegaConf
from hydra.utils import instantiate


class MockNormStats:
    def __init__(self):
        self._t = {0: {"action_keys": ["actions"], "proprio_keys": ["state_agent_obj"],
                       "lang_keys": [], "camera_keys": ["front_img_1"]}}
    def keys_of_type(self, kind, emb_id):
        return self._t.get(emb_id, {}).get(kind, [])
    def is_key_with_embodiment(self, k, eid): return True
    def zarr_key_to_keyname(self, k, eid): return k
    def normalize(self, b, e): return b
    def unnormalize(self, b, e): return b


import egomimic.rldb.embodiment.embodiment as _emb
orig_get_id = _emb.get_embodiment_id
orig_get_name = _emb.get_embodiment
_emb.get_embodiment_id = lambda n: 0 if n == "pushshapes_sim" else orig_get_id(n)
_emb.get_embodiment = lambda i: "pushshapes_sim" if i == 0 else orig_get_name(i)

cfg = OmegaConf.load(
    "/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3/"
    "egomimic/hydra_configs/model/dfot/base.yaml"
)
dfot = instantiate(cfg.robomimic_model, norm_stats=MockNormStats(), _recursive_=True)
dfot.nets = dfot.nets.cpu()
dfot.device = torch.device("cpu")

# Synthetic single-env-tick obs (normalized).
obs = {
    "state_agent_obj": torch.randn(1, 5),
    "front_img_1": torch.rand(1, 3, 96, 96),
}

# ---- AR mode (default) ----
dfot.inference_mode = "ar"
dfot.ar_inference_chunk_size = 1
dfot.ar_inference_step_size = 1
print(f"[ar] inference_mode={dfot.inference_mode}, chunk={dfot.ar_inference_chunk_size}, step={dfot.ar_inference_step_size}")

action_t0 = dfot.inference_step(obs, t=0, emb_id=0)
print(f"[ar] action @ t=0: {action_t0}  shape={action_t0.shape}")
assert action_t0.shape == (2,), f"expected (2,), got {action_t0.shape}"
assert np.isfinite(action_t0).all()

action_t1 = dfot.inference_step(obs, t=1, emb_id=0)
print(f"[ar] action @ t=1: {action_t1}")
assert action_t1.shape == (2,)

# ---- Chunk mode ----
dfot.inference_mode = "chunk"
if hasattr(dfot, "_sim_state"):
    del dfot._sim_state
print(f"\n[chunk] inference_mode={dfot.inference_mode}")
action_chunk_t0 = dfot.inference_step(obs, t=0, emb_id=0)
print(f"[chunk] action @ t=0: {action_chunk_t0}  shape={action_chunk_t0.shape}")
assert action_chunk_t0.shape == (2,)

print("\nPASS — refactored DFoT inference_step works in both ar + chunk modes")
