import sys, torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from hydra.utils import instantiate
import egomimic
from egomimic.algo.hpt import HPT

CFG_DIR = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/egomimic/hydra_configs/model"

class FakeNormStats:
    norm_mode = "minmax"
    def keys_of_type(self, key_type, embodiment_id):
        return []
    def is_key_with_embodiment(self, key, embodiment_id):
        return False
    def zarr_key_to_keyname(self, key, embodiment_id):
        return key

def build(model_name):
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        cfg = compose(config_name=model_name)
    rm = cfg.robomimic_model
    # force CPU
    container = OmegaConf.to_container(rm, resolve=True)
    container["device"] = "cpu"
    # instantiate HPT directly (not via ModelWrapper) so no trainer/datamodule needed
    m = instantiate({**container, "norm_stats": FakeNormStats(), "device": "cpu"})
    return m, cfg

def npar(m): return sum(p.numel() for p in m.parameters())

name = sys.argv[1]
m, cfg = build(name)
policy = m.nets["policy"]
total = npar(m.nets)
print(f"=== {name} ===")
print(f"nets total params = {total:,} ({total/1e6:.4f}M)")
print(f"domains = {m.domains}")
# trunk
print(f"trunk embed_dim={policy.embed_dim} action_horizon={policy.action_horizon}")
# head act_seq introspection
for d, head in policy.heads.items():
    aq = getattr(head, 'act_seq', None)
    ah = getattr(head, 'action_horizon', None)
    # FMPolicy stores model CrossTransformer with act_seq
    inner = getattr(head, 'net', None) or getattr(head, 'model', None)
    inner_aq = getattr(inner, 'act_seq', None) if inner is not None else None
    print(f"  head[{d}] type={type(head).__name__} act_seq={aq} action_horizon={ah} inner_act_seq={inner_aq}")
