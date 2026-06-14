import os, traceback
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")

def res(name, ovr, sub):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        try:
            cfg = compose(config_name=name, overrides=ovr)
            node = cfg
            for p in (sub.split('.') if sub else []):
                node = getattr(node, p)
            OmegaConf.to_container(node, resolve=True)
            print(f"OK   {name} {ovr} sub={sub}")
        except Exception as e:
            print(f"FAIL {name} {ovr} sub={sub}: {type(e).__name__}: {str(e)[:140]}")

res("train_zarr_cartesian", ["data=eva"], "data")
res("train_zarr_cartesian", ["data=scale_pi"], "data")
res("train_zarr_cartesian", ["+data_schematic=hpt"], "data_schematic")

from omegaconf import OmegaConf as OC
try:
    n = OC.load(os.path.join(CONFIG_DIR, "model/hpt_bc_flow_pushshapes.yaml"))
    OC.to_container(n, resolve=False)
    print("OK   rawload hpt_bc_flow_pushshapes")
except Exception as e:
    print("FAIL rawload hpt_bc_flow_pushshapes:", e)

with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
    cfg = compose(config_name="train_zarr_cartesian", overrides=["data=eva"])
    print("paths.dataset_dir =", OmegaConf.select(cfg, "paths.dataset_dir"))
    try:
        OmegaConf.to_container(cfg.data, resolve=True)
    except Exception:
        traceback.print_exc()
