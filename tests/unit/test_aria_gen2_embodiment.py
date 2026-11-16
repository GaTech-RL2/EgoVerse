"""Aria Gen 2 is its own embodiment; Gen 1 / human embodiments are unchanged."""

from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize_config_module

import egomimic
from egomimic.rldb.embodiment.embodiment import (
    EMBODIMENT,
    canonical_embodiment_name,
    get_embodiment_id,
)
from egomimic.rldb.zarr.episode_rules import EMBODIMENTS


def test_existing_embodiment_ids_are_unchanged():
    assert {e.name: e.value for e in EMBODIMENT if e.value <= 6} == {
        "HUMAN_RIGHT_ARM": 1,
        "HUMAN_LEFT_ARM": 2,
        "HUMAN_BIMANUAL": 3,
        "EVA_RIGHT_ARM": 4,
        "EVA_LEFT_ARM": 5,
        "EVA_BIMANUAL": 6,
    }


@pytest.mark.parametrize(
    "name, canonical",
    [
        ("human_bimanual", "HUMAN_BIMANUAL"),
        ("aria_bimanual", "HUMAN_BIMANUAL"),  # legacy Gen 1 vendor alias
        ("mecka_left_arm", "HUMAN_LEFT_ARM"),
        ("eva_bimanual", "EVA_BIMANUAL"),
        ("aria_gen2", "ARIA_GEN2_BIMANUAL"),  # SQL rows name the device
        ("aria_gen2_bimanual", "ARIA_GEN2_BIMANUAL"),
        ("aria_gen2_right_arm", "ARIA_GEN2_RIGHT_ARM"),
    ],
)
def test_embodiment_names_resolve(name, canonical):
    assert canonical_embodiment_name(name) == canonical


def test_aria_gen2_has_its_own_id():
    gen2 = {get_embodiment_id(n) for n in ("aria_gen2", "aria_gen2_bimanual")}
    assert gen2 == {EMBODIMENT.ARIA_GEN2_BIMANUAL.value}
    assert gen2.isdisjoint({get_embodiment_id("human_bimanual")})
    assert "aria_gen2_bimanual" in EMBODIMENTS


def _dataset_keys(data_config):
    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        cfg = compose(
            config_name="train_zarr_cartesian", overrides=[f"data={data_config}"]
        )
    return list(cfg.data.train_datasets), cfg


def test_aria_gen2_config_resolves_to_the_gen2_embodiment():
    keys, cfg = _dataset_keys("aria_gen2")
    assert keys == ["aria_gen2_bimanual"]
    assert get_embodiment_id(keys[0]) == EMBODIMENT.ARIA_GEN2_BIMANUAL.value
    lambdas = cfg.data.train_datasets.aria_gen2_bimanual.filters.filter_lambdas
    assert any("'aria_gen2'" in f for f in lambdas)


@pytest.mark.parametrize("data_config", ["human", "aria", "aria_keypoints"])
def test_gen1_configs_still_resolve_to_human(data_config):
    keys, _ = _dataset_keys(data_config)
    assert "human_bimanual" in keys
    assert get_embodiment_id("human_bimanual") == EMBODIMENT.HUMAN_BIMANUAL.value


MODEL_DIR = Path(egomimic.__file__).parent / "hydra_configs" / "model"
PI_CONFIGS = sorted(
    p.stem for p in MODEL_DIR.glob("*.yaml") if "action_converters" in p.read_text()
)


@pytest.mark.parametrize("model", PI_CONFIGS)
def test_action_converter_rules_match_ac_keys(model):
    """Pi registers each rule under ac_keys[get_embodiment_id(rule)]; a rule
    for an embodiment the model has no ac_key for fails at init (no openpi
    needed to catch it)."""
    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        cfg = compose(config_name=f"model/{model}").model.robomimic_model
    ac_ids = {get_embodiment_id(k) for k in cfg.ac_keys}
    for rule, conv in cfg.action_converters.rules.items():
        assert get_embodiment_id(rule) in ac_ids, f"{model}: rule {rule} has no ac_key"
        assert hydra.utils.instantiate(conv) is not None
