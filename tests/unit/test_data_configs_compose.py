"""Every hydra data config must compose under the train config and fully resolve.

KNOWN_BROKEN lists configs that fail for reasons outside this branch; they are
strict xfails, so the day one is fixed the test fails until its entry is removed.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

import egomimic.hydra_configs as _cfg_pkg
from egomimic.rldb.embodiment.human import Human

DATA_DIR = Path(_cfg_pkg.__file__).parent / "data"
DATA_CONFIGS = sorted(p.stem for p in DATA_DIR.glob("*.yaml"))

# name -> reason. Strict xfail: remove the entry once the config is fixed.
KNOWN_BROKEN: dict[str, str] = {
    "cotrain_pi_latent": (
        "omegaconf UnsupportedValueType: a list is assigned where a primitive is "
        "expected (config never composes; not touched by this branch)"
    ),
}


def compose_data(name: str, overrides: list[str] | None = None):
    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"data={name}", "paths.output_dir=/x", *(overrides or [])],
        )
    OmegaConf.resolve(cfg.data)
    return cfg


@pytest.mark.parametrize("name", DATA_CONFIGS)
def test_data_config_composes_and_resolves(request, name):
    if name in KNOWN_BROKEN:
        request.applymarker(pytest.mark.xfail(strict=True, reason=KNOWN_BROKEN[name]))
    cfg = compose_data(name)
    assert cfg.data.train_datasets, f"{name}: no train_datasets"
    for ds_name, ds in cfg.data.train_datasets.items():
        if ds is None:
            continue
        assert (
            ds_name in cfg.data.train_dataloader_params
        ), f"{name}: {ds_name} has no dataloader params"


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
