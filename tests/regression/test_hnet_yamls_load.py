"""Batch smoke: instantiate every H-Net yaml AND run a tiny packed-mode
forward_training to verify each algo's end-to-end path works after the
outer_stage refactor. Stage-based configs need cond_encoder + hnet
stages + chunker to pump aux through to HNetLoss; flat-fused configs
should also work but use ctx.aux=[] (no chunker).

Padded mode hits a pre-existing torch SDPA bug for some causal+train
configurations; packed mode matches production training and avoids
that path."""
import pytest  # noqa: E402

pytest.skip(
    "manual regression smoke: hardcoded EgoVerse-clone-3 paths + configs "
    "removed from this repo + needs GPU/checkpoints; run directly with "
    "python, skipped under pytest collection",
    allow_module_level=True,
)

import sys, torch
sys.path.insert(0, "/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3")
from pathlib import Path
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


YAML_DIR = Path("/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3/"
                "egomimic/hydra_configs/model")
CONFIGS = [
    "hnet_pushshapes.yaml",
    "hnet_pushshapes_big.yaml",
    "hnet_pushshapes_crossattn.yaml",
    "hnet_pushshapes_mamba_encdec.yaml",
    "hnet_pushshapes_obs_ar.yaml",
    "hnet_pushshapes_obs_ar_large.yaml",
    "hnet_pushshapes_recipe.yaml",
    "hnet_pushshapes_fused.yaml",
    "hnet_pushshapes_fused_lowlr.yaml",
    "hnet_pushshapes_fused_pusher.yaml",
]


def build_packed_batch(action_dim, device):
    # Two short episodes packed together — packed mode mirrors production.
    T1, T2 = 12, 20
    T_total = T1 + T2
    cu = torch.tensor([0, T1, T_total], dtype=torch.long, device=device)
    return {
        0: {
            "actions": torch.randn(T_total, action_dim, device=device),
            "state_agent_obj": torch.randn(T_total, 5, device=device),
            "front_img_1": torch.rand(T_total, 3, 96, 96, device=device),
            "cu_seqlens": cu,
            "max_seq_len": T2,
            "seq_lens": torch.tensor([T1, T2], dtype=torch.long, device=device),
            "_packed": True,
        },
    }


results = []
for name in CONFIGS:
    path = YAML_DIR / name
    try:
        cfg = OmegaConf.load(path)
        model = instantiate(cfg.robomimic_model, norm_stats=MockNormStats(), _recursive_=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.nets = model.nets.to(device)
        model.device = device
        n_params = sum(p.numel() for p in model.nets.parameters())
        action_dim = model.outer_stage.action_dim
        batch = build_packed_batch(action_dim, device)
        # Run forward_training (refactored to go through outer_stage + loss).
        torch.manual_seed(42)
        predictions = model.forward_training(batch)
        loss_dict = model.compute_losses(predictions, batch)
        action_loss = float(loss_dict["action_loss"])
        results.append((name, "OK", n_params, action_loss))
        print(f"[OK]   {name:45s} params={n_params/1e6:6.2f}M  action_loss={action_loss:.4f}")
    except Exception as e:
        results.append((name, "FAIL", 0, str(e)))
        print(f"[FAIL] {name:45s} {type(e).__name__}: {e}")

n_ok = sum(1 for _, s, _, _ in results if s == "OK")
print(f"\n{n_ok}/{len(CONFIGS)} configs instantiate + forward_training succeed")
if n_ok != len(CONFIGS):
    print("\nFailures:")
    for name, status, _, err in results:
        if status != "OK":
            print(f"  {name}: {err}")
assert n_ok == len(CONFIGS), "some configs failed"
