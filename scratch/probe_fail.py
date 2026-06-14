import json, os, traceback
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")

m = json.load(open(os.path.join(REPO, "scratch/config_phase2_baseline/resolved/methods.json")))
print("=== rawload-fallback / FAIL entries ===")
for k, v in sorted(m.items()):
    if "compose failed" in v or v.startswith("FAIL") or "non-composable" in v:
        print(f"{k} :: {v[:160]}")

print("\n=== live error: compose entry with data=eva ===")
with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
    try:
        cfg = compose(config_name="train_zarr_cartesian", overrides=["data=eva"])
        print("data=eva OK, cfg.data type:", type(cfg.data))
    except Exception as e:
        traceback.print_exc()

print("\n=== live error: compose entry train_zarr_cartesian (no overrides), full resolve ===")
with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
    try:
        cfg = compose(config_name="train_zarr_cartesian", overrides=[])
        OmegaConf.to_container(cfg, resolve=True)
        print("full resolve OK")
    except Exception as e:
        traceback.print_exc()

print("\n=== live error: model=hpt_bc_flow_pushshapes with data=tsimulation_hpt ===")
with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
    try:
        cfg = compose(config_name="train_zarr_cartesian", overrides=["model=hpt_bc_flow_pushshapes", "data=tsimulation_hpt"])
        OmegaConf.to_container(cfg.model, resolve=True)
        print("hpt_bc_flow_pushshapes+tsimulation_hpt OK")
    except Exception as e:
        traceback.print_exc()
