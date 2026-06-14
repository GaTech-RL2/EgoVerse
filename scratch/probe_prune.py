import os, json
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")

# Keys whose interpolation needs the live hydra runtime / commented-out root_dir.
RUNTIME_KEYS = [
    "paths.output_dir", "paths.log_dir", "paths.work_dir",
    "trainer.default_root_dir",
    "logger.save_dir",
    "hydra",
]

def data_resolved(name):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="train_zarr_cartesian", overrides=[f"data={name}"])
        # Resolve cfg.data with cfg as the root context: convert whole cfg but
        # only after removing runtime-only keys that would crash resolution.
        OmegaConf.set_struct(cfg, False)
        real_dataset_dir = OmegaConf.select(cfg, "paths.dataset_dir")
        for k in ["paths", "trainer", "logger", "callbacks"]:
            if k in cfg:
                del cfg[k]
        # re-add a minimal paths with the REAL composed dataset_dir so ${paths.dataset_dir} resolves
        cfg.paths = {"dataset_dir": real_dataset_dir}
        full = OmegaConf.to_container(cfg, resolve=True)
        return full["data"]

for name in ["eva", "scale_pi", "cotrain_pi_lang", "tsimulation_hpt", "aria_pi"]:
    try:
        c = data_resolved(name)
        s = json.dumps(c)
        # confirm the real dataset_dir made it in (interpolation resolved), no stub
        print(f"OK data={name}  has_dataset_dir={'egoverseS3ZarrDataset' in s}  len={len(s)}")
    except Exception as e:
        print(f"FAIL data={name}: {type(e).__name__}: {str(e)[:120]}")
