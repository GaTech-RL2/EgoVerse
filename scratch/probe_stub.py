import os
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")

# Register a stub 'hydra' resolver so ${hydra:runtime.output_dir} etc. don't crash.
if not OmegaConf.has_resolver("hydra"):
    OmegaConf.register_new_resolver("hydra", lambda key: f"__HYDRA__{key}__")

def dump_group_from_root(group, name, prefix=False):
    ov = [f"+{group}={name}"] if prefix else [f"{group}={name}"]
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="train_zarr_cartesian", overrides=ov)
        # inject the commented-out root_dir so paths.* can resolve deterministically
        OmegaConf.update(cfg, "paths.root_dir", "__PROJECT_ROOT__", force_add=True)
        sub = getattr(cfg, group)
        c = OmegaConf.to_container(sub, resolve=True)
        return c

# data should NOT reference hydra/root_dir in its own subtree -> values are real
for name in ["eva", "scale_pi", "tsimulation_hpt", "cotrain_pi_lang"]:
    try:
        c = dump_group_from_root("data", name)
        # show one resolved folder_path to confirm it's the REAL dataset_dir, not a stub
        import json
        s = json.dumps(c)
        has_stub = "__HYDRA__" in s or "__PROJECT_ROOT__" in s
        print(f"OK data={name}  contains_stub={has_stub}")
    except Exception as e:
        print(f"FAIL data={name}: {type(e).__name__}: {str(e)[:120]}")

for grp, name in [("trainer","ddp"), ("logger","wandb"), ("paths","default"),
                  ("evaluator","eval_hpt"), ("callbacks","checkpoints")]:
    try:
        c = dump_group_from_root(grp, name)
        import json
        s = json.dumps(c)
        print(f"OK {grp}={name}  contains_stub={('__HYDRA__' in s or '__PROJECT_ROOT__' in s)}")
    except Exception as e:
        print(f"FAIL {grp}={name}: {type(e).__name__}: {str(e)[:120]}")
