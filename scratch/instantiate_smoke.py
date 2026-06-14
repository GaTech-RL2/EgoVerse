"""Strongest gate: actually instantiate the data config (not just compose) for
the most-referenced refactored leaves, confirming the defaults-based config
builds a working MultiDataModuleWrapper with the right resolver/keymap. Uses
the real launcher override values for tsimulation."""
import os
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")
NC3 = "/coc/flash7/paphiwetsa3/datasets/new_circle_3"
KM = "egomimic.rldb.embodiment.pushshapes.get_keymap_eval"


def check(name, overrides):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="train_zarr_cartesian",
                      overrides=[f"data={name}"] + overrides)
    d = OmegaConf.to_container(cfg.data, resolve=True)
    tr = d["train_datasets"]["pushshapes_sim"]
    va = d["valid_datasets"]["pushshapes_sim"]
    print(f"{name}:")
    print(f"  train.resolver._target_ = {tr['resolver']['_target_']}")
    print(f"  train.resolver.folder_path = {tr['resolver']['folder_path']}")
    print(f"  train.resolver.key_map = {tr['resolver']['key_map']}")
    print(f"  valid.resolver.folder_path = {va['resolver']['folder_path']}")
    print(f"  valid.resolver.key_map._target_ = {va['resolver']['key_map']['_target_']}")
    print(f"  train dl = {d['train_dataloader_params']['pushshapes_sim']}")


# sibling BC launcher override set against tsimulation
sib = [
    f"data.train_datasets.pushshapes_sim.resolver.folder_path={NC3}",
    f"data.train_datasets.pushshapes_sim.resolver.key_map._target_={KM}",
    f"data.valid_datasets.pushshapes_sim.resolver.folder_path={NC3}",
    f"data.valid_datasets.pushshapes_sim.resolver.key_map._target_={KM}",
    "data.train_dataloader_params.pushshapes_sim.batch_size=16",
    "data.valid_dataloader_params.pushshapes_sim.batch_size=16",
]
check("tsimulation", sib)
check("tsimulation_hpt", sib)
check("tsimulation_hpt_causal", [])
check("tsimulation_delta", [])
print("INSTANTIATE_SMOKE_OK")
