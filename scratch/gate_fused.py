"""Identity gate for the fused-cluster inheritance refactor.

Re-dump each of the 11 leaf model configs exactly the way dump_phase2_configs.py
did (compose train_zarr_cartesian model=<name>, OmegaConf.to_container(cfg.model,
resolve=True), sort_keys json) and diff vs the baseline at
scratch/config_phase2_baseline/resolved/model/<name>.json. ALL must be EMPTY.
"""
import json, os, sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")
BASE = os.path.join(REPO, "scratch/config_phase2_baseline/resolved/model")
ENTRY = "train_zarr_cartesian"

LEAVES = [
    "hnet_pushshapes_fused",
    "hnet_pushshapes_fused_goal",
    "hnet_pushshapes_fused_lowlr",
    "hnet_pushshapes_fused_pusher",
    "hnet_pushshapes_fused_pusher_resnet",
    "hnet_pushshapes_fused_ar_crossattn_resnet",
    "hnet_pushshapes_fused_windowed",
    "hnet_pushshapes_fused_windowed_resnet",
    "hnet_pushshapes_fused_windowed_resnet_cos15",
    "hnet_pushshapes_fused_windowed_crossattn_resnet",
    "hnet_pushshapes_fused_windowed_crossattn_resnet_lowlr",
]

def norm(c):
    return json.dumps(c, sort_keys=True, indent=2)

fails = []
with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
    for name in LEAVES:
        cfg = compose(config_name=ENTRY, overrides=[f"model={name}"])
        got = OmegaConf.to_container(cfg.model, resolve=True)
        with open(os.path.join(BASE, name + ".json")) as f:
            want = json.load(f)
        if norm(got) == norm(want):
            print(f"PASS  {name}")
        else:
            fails.append(name)
            print(f"FAIL  {name}")
            gl = norm(got).splitlines()
            wl = norm(want).splitlines()
            import difflib
            for line in difflib.unified_diff(wl, gl, "baseline", "new", lineterm=""):
                print("   " + line)

print(f"\nGATE: {len(LEAVES)-len(fails)}/{len(LEAVES)} PASS", flush=True)
if fails:
    print("FAILED: " + ", ".join(fails))
    sys.exit(1)
print("ALL EMPTY DIFFS ✓")
