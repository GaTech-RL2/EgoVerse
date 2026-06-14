import os, traceback
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")

# Resolve the FULL cfg root, then extract the sub-tree. This makes absolute
# interpolations like ${paths.dataset_dir} resolve correctly.
def res_from_root(ovr, group):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="train_zarr_cartesian", overrides=ovr)
        full = OmegaConf.to_container(cfg, resolve=True)  # may fail on hydra: runtime
        return full[group]

for grp, name in [("data","eva"), ("data","scale_pi"), ("data","tsimulation_hpt"),
                  ("data","cotrain_pi_lang"), ("trainer","ddp"), ("logger","wandb"),
                  ("paths","default")]:
    ovr = [f"{grp}={name}"] if grp != "data_schematic" else [f"+{grp}={name}"]
    try:
        sub = res_from_root(ovr, grp)
        print(f"OK   full-root {grp}={name}")
    except Exception as e:
        print(f"FAIL full-root {grp}={name}: {type(e).__name__}: {str(e)[:130]}")
