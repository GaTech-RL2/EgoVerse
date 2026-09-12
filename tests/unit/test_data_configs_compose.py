"""Every hydra data config must compose under the train config, fully resolve,
and have key_map / transform_list nodes that instantiate.

KNOWN_BROKEN* list configs that fail for reasons outside this branch; they are
strict xfails, so the day one is fixed the test fails until its entry is removed.
"""

from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

import egomimic.hydra_configs as _cfg_pkg
from egomimic.rldb.embodiment.human import Human

DATA_DIR = Path(_cfg_pkg.__file__).parent / "data"
DATA_CONFIGS = sorted(p.stem for p in DATA_DIR.glob("*.yaml"))

# name -> reason. Strict xfail: remove the entry once the config is fixed.
KNOWN_BROKEN_COMPOSE: dict[str, str] = {
    "cotrain_pi_latent": (
        "omegaconf UnsupportedValueType: a list is assigned where a primitive is "
        "expected (config never composes; not touched by this branch)"
    ),
}
KNOWN_BROKEN_INSTANTIATE: dict[str, str] = {
    **KNOWN_BROKEN_COMPOSE,
    "aria_pi": "Human.get_keymap called without keymap_mode (TypeError)",
    "mecka_pi": "Human.get_keymap called without keymap_mode (TypeError)",
    "scale_pi": "Human.get_keymap called without keymap_mode (TypeError)",
    "industry_eva_pi": "eva domain: Eva.get_keymap called without keymap_mode (TypeError)",
    "mecka_scale_cotrain_pi": "Human.get_keymap called without keymap_mode (TypeError)",
}


def _params(known_broken: dict[str, str]):
    return [
        pytest.param(
            name,
            marks=pytest.mark.xfail(strict=True, reason=known_broken[name])
            if name in known_broken
            else (),
        )
        for name in DATA_CONFIGS
    ]


def compose_data(name: str, overrides: list[str] | None = None):
    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"data={name}", *(overrides or [])],
        )
    OmegaConf.resolve(cfg.data)
    return cfg


@pytest.mark.parametrize("name", _params(KNOWN_BROKEN_COMPOSE))
def test_data_config_composes_and_resolves(name):
    cfg = compose_data(name)
    assert cfg.data.train_datasets, f"{name}: no train_datasets"
    for ds_name, ds in cfg.data.train_datasets.items():
        if ds is None:
            continue
        assert (
            ds_name in cfg.data.train_dataloader_params
        ), f"{name}: {ds_name} has no dataloader params"


@pytest.mark.parametrize("name", _params(KNOWN_BROKEN_INSTANTIATE))
def test_data_config_keymap_and_transforms_instantiate(name):
    """The resolver's key_map / transform_list nodes are plain function calls
    (Embodiment.get_keymap / get_transform_list); instantiating them needs no
    data and catches missing or misspelled arguments."""
    cfg = compose_data(name)
    for split in ("train_datasets", "valid_datasets"):
        for ds_name, ds in cfg.data[split].items():
            if ds is None:
                continue
            key_map = hydra.utils.instantiate(ds.resolver.key_map)
            assert (
                isinstance(key_map, dict) and key_map
            ), f"{name}/{split}/{ds_name}: empty key_map"
            if ds.resolver.get("transform_list") is not None:
                transforms = hydra.utils.instantiate(ds.resolver.transform_list)
                assert (
                    transforms is not None
                ), f"{name}/{split}/{ds_name}: get_transform_list returned None (unknown mode?)"


@pytest.mark.parametrize("name", ["eva", "aria", "mecka", "scale"])
def test_vendor_valid_split_follows_train(name):
    cfg = compose_data(name)
    emb = next(iter(cfg.data.train_datasets))
    assert (
        cfg.data.valid_datasets[emb].resolver.folder_path
        == cfg.data.train_datasets[emb].resolver.folder_path
    )


def test_scale_loads_head_pose():
    cfg = compose_data("scale")
    km = cfg.data.train_datasets.human_bimanual.resolver.key_map
    assert (
        km.get("has_head_pose", True) is True
    ), "scale exports carry obs_head_pose; the flag was stale"
    assert "obs_head_pose" in Human.get_keymap(keymap_mode="cartesian")


def test_cotrain_pi_base_human_uses_pi_keymap():
    cfg = compose_data("cotrain_pi_base")
    km = cfg.data.train_datasets.human_bimanual.resolver.key_map
    assert km.keymap_mode == "cartesian_pi"
