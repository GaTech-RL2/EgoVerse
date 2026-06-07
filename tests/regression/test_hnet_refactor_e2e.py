"""End-to-end smoke: instantiate refactored HNet via hydra config,
run forward_training, check the resulting predictions dict is well-formed."""
import pytest  # noqa: E402

pytest.skip(
    "manual regression smoke: hardcoded EgoVerse-clone-3 paths + configs "
    "removed from this repo + needs GPU/checkpoints; run directly with "
    "python, skipped under pytest collection",
    allow_module_level=True,
)

import sys, torch
sys.path.insert(0, "/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3")
from omegaconf import OmegaConf
from hydra.utils import instantiate


class MockNormStats:
    def __init__(self):
        self._t = {0: {"action_keys": ["actions"], "proprio_keys": ["state_agent_obj"],
                       "lang_keys": [], "camera_keys": ["front_img_1"]}}
    def keys_of_type(self, kind, eid): return self._t.get(eid, {}).get(kind, [])
    def is_key_with_embodiment(self, k, e): return True
    def zarr_key_to_keyname(self, k, e): return k
    def normalize(self, b, e): return b
    def unnormalize(self, b, e): return b


import egomimic.rldb.embodiment.embodiment as _emb
orig = _emb.get_embodiment_id
_emb.get_embodiment_id = lambda n: 0 if n == "pushshapes_sim" else orig(n)


cfg = OmegaConf.load(
    "/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3/"
    "egomimic/hydra_configs/model/hnet_pushshapes.yaml"
)
hnet = instantiate(cfg.robomimic_model, norm_stats=MockNormStats(), _recursive_=True)
hnet.nets = hnet.nets.cpu()
hnet.device = torch.device("cpu")
# NOTE: leaving in train mode. Use packed mode below to match production.

print(f"[init] HNet built. nets keys: {list(hnet.nets.keys())}")
print(f"[init] outer_stage: {type(hnet.outer_stage).__name__}")
print(f"[init] loss: {type(hnet.loss).__name__}")
print(f"[init] policy (alias): {type(hnet.policy).__name__}")
print(f"[init] cond_encoder (property): {type(hnet.cond_encoder).__name__}")
print(f"[init] hnet (property): {type(hnet.hnet).__name__}")
print(f"[init] action_horizon: {hnet.action_horizon}")

# Packed batch — matches production training. Two episodes packed together.
T1, T2 = 12, 20
T_total = T1 + T2
cu = torch.tensor([0, T1, T_total], dtype=torch.long)
batch = {
    0: {
        "actions": torch.randn(T_total, 2),
        "state_agent_obj": torch.randn(T_total, 5),
        "front_img_1": torch.rand(T_total, 3, 96, 96),
        "cu_seqlens": cu,
        "max_seq_len": T2,
        "seq_lens": torch.tensor([T1, T2], dtype=torch.long),
        "_packed": True,
    },
}
torch.manual_seed(42)
predictions = hnet.forward_training(batch)
loss_dict = hnet.compute_losses(predictions, batch)

print(f"\n[forward_training] predictions keys: {sorted(predictions.keys())}")
print(f"[forward_training] 0_action_loss: {predictions['0_action_loss'].item():.6f}")
print(f"[forward_training] 0_ratio_loss: {predictions['0_ratio_loss'].item():.6f}")
print(f"[compute_losses] losses: {list(loss_dict.keys())}")
assert torch.isfinite(predictions["0_action_loss"]).all()
print("\nPASS — refactored HNet instantiates from yaml + runs forward_training")
