"""Test the EXACT ratio-launcher override sequence (with ~delete + re-add)
against the current config dir. This is the real gate for tsimulation.yaml's
fullhist_ratio launcher."""
import os
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")
NC3 = "/coc/flash7/paphiwetsa3/datasets/new_circle_3"
KM = "egomimic.rldb.embodiment.pushshapes.get_keymap_eval"

RATIO_OVERRIDES = [
    "data=tsimulation",
    "model=bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist_ratio",
    "data.train_dataloader_params.pushshapes_sim.batch_size=16",
    "data.valid_dataloader_params.pushshapes_sim.batch_size=16",
    "data.train_datasets.pushshapes_sim.resolver.folder_path=" + NC3,
    "data.train_datasets.pushshapes_sim.resolver.key_map._target_=" + KM,
    "~data.valid_datasets.pushshapes_sim",
    "+data.valid_datasets.pushshapes_sim._target_=egomimic.rldb.zarr.zarr_dataset_packed.ZarrEpisodePackedDataset.from_resolver",
    "+data.valid_datasets.pushshapes_sim.resolver._target_=egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver",
    "+data.valid_datasets.pushshapes_sim.resolver.folder_path=" + NC3,
    "+data.valid_datasets.pushshapes_sim.resolver.key_map._target_=" + KM,
    "+data.valid_datasets.pushshapes_sim.resolver.key_map.action_horizon=1024",
    "+data.valid_datasets.pushshapes_sim.resolver.transform_list=null",
    "+data.valid_datasets.pushshapes_sim.chunking=none",
    "+data.valid_datasets.pushshapes_sim.min_seq_len=64",
    "+data.valid_datasets.pushshapes_sim.max_seq_len=null",
]

# Standard sibling BC launcher pattern: overrides valid resolver sub-keys directly
SIBLING_OVERRIDES = [
    "data=tsimulation",
    "model=bc_rnn_pushshapes",
    "data.train_datasets.pushshapes_sim.resolver.folder_path=" + NC3,
    "data.train_datasets.pushshapes_sim.resolver.key_map._target_=" + KM,
    "data.valid_datasets.pushshapes_sim.resolver.folder_path=" + NC3,
    "data.valid_datasets.pushshapes_sim.resolver.key_map._target_=" + KM,
    "data.train_dataloader_params.pushshapes_sim.batch_size=16",
    "data.valid_dataloader_params.pushshapes_sim.batch_size=16",
]


def run(label, overrides):
    try:
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            cfg = compose(config_name="train_zarr_cartesian", overrides=overrides)
        # resolve valid block to confirm it points at NC3 + get_keymap_eval
        vd = OmegaConf.to_container(cfg.data.valid_datasets.pushshapes_sim, resolve=True)
        fp = vd.get("resolver", {}).get("folder_path")
        km = vd.get("resolver", {}).get("key_map", {}).get("_target_")
        print(f"{label}: COMPOSE_OK  valid.folder_path={fp}  valid.key_map={km}")
        return True
    except Exception as e:
        print(f"{label}: FAIL {repr(e).splitlines()[0][:200]}")
        return False


print("=== ratio launcher (~delete + re-add explicit) ===")
run("RATIO", RATIO_OVERRIDES)
print("=== sibling BC launcher (.resolver.* sub-overrides on valid) ===")
run("SIBLING", SIBLING_OVERRIDES)
