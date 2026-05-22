"""Batch smoke: instantiate every stage-based H-Net yaml via hydra and
confirm it builds without error. Doesn't run forward — just confirms the
new outer_stage block schema parses and HNet.__init__ accepts it."""
import sys
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
]

results = []
for name in CONFIGS:
    path = YAML_DIR / name
    try:
        cfg = OmegaConf.load(path)
        model = instantiate(cfg.robomimic_model, norm_stats=MockNormStats(), _recursive_=True)
        n_params = sum(p.numel() for p in model.nets.parameters())
        results.append((name, "OK", n_params))
        print(f"[OK]   {name:45s} params={n_params/1e6:.2f}M  outer={type(model.outer_stage).__name__}")
    except Exception as e:
        results.append((name, "FAIL", str(e)))
        print(f"[FAIL] {name:45s} {type(e).__name__}: {e}")

n_ok = sum(1 for _, s, _ in results if s == "OK")
print(f"\n{n_ok}/{len(CONFIGS)} configs instantiated successfully")
assert n_ok == len(CONFIGS), "some configs failed to instantiate"
