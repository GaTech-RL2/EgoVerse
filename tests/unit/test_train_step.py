"""CPU tier: every (vendor, algo) recipe builds and takes 2 optimizer steps on
synthetic data through the real trainHydra.train() path. Pi's openpi network
is a stub; the EgoVerse Pi wrapper (keymaps, converters, prompts, batch
assembly) is real."""

from __future__ import annotations

import math
import os

import pytest
from fixtures.recipes import (
    RECIPES,
    abc_dit_small_overrides,
    common_overrides,
    cpu_trainer_overrides,
    flowvla_small_overrides,
    hpt_small_overrides,
    pi_cpu_overrides,
)
from fixtures.train_harness import (
    STATE_DIM,
    BatchKeySpy,
    StubPI0,
    compose_recipe,
    hermetic_env,
    pi_unavailable,
    write_fixtures,
)

import egomimic.trainHydra as train_hydra
from egomimic.algo.hpt import HPT
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Human

VENDOR_NAMES = ["eva", "aria", "mecka", "scale"]
STEPS = 2


def _run(cfg) -> float:
    metrics, objects = train_hydra.train(cfg)
    loss = float(metrics["Train/action_loss"])
    assert math.isfinite(loss), f"non-finite loss {loss}"
    assert objects["trainer"].global_step == STEPS
    return loss


def _expected_keys(embodiment: str) -> set[str]:
    """Batch keys every recipe must hand the algo: eva trains on the cartesian
    action, human on the wrist-frame hand-keypoint action (plus the cartesian
    ee_pose proprio the keypoint data configs emit for Pi's prompt)."""
    if embodiment == "eva_bimanual":
        return {
            "actions_cartesian",
            "observations.state.ee_pose",
            Eva.VIZ_IMAGE_KEY,
            "observations.images.right_wrist_img",
            "observations.images.left_wrist_img",
        }
    return {
        "actions_keypoints",
        "observations.state.keypoints",
        "observations.state.ee_pose",
        Human.VIZ_IMAGE_KEY,
    }


@pytest.mark.parametrize("vendor", VENDOR_NAMES)
def test_hpt_train_step(tmp_path, monkeypatch, vendor):
    hermetic_env(monkeypatch)
    recipe = RECIPES[(vendor, "hpt")]
    data, out, hashes = write_fixtures(tmp_path, vendor)
    cfg = compose_recipe(
        recipe,
        common_overrides(
            recipe.embodiment,
            data,
            out,
            batch_size=2,
            num_workers=0,
            episode_hashes=hashes,
        )
        + cpu_trainer_overrides(STEPS)
        + hpt_small_overrides(recipe.embodiment),
        out,
    )
    spy = BatchKeySpy(monkeypatch, HPT)
    _run(cfg)
    seen = spy.keys[recipe.embodiment]
    expected = _expected_keys(recipe.embodiment)
    assert (
        expected <= seen
    ), f"{vendor}/hpt missing {expected - seen}; saw {sorted(seen)}"


@pytest.mark.parametrize("vendor", VENDOR_NAMES)
def test_pi_train_step(tmp_path, monkeypatch, vendor):
    reason = pi_unavailable()
    if reason:
        if os.environ.get("EGOVERSE_REQUIRE_PI"):  # set in CI
            pytest.fail(reason)
        pytest.skip(reason)
    import openpi.models_pytorch.pi0_pytorch as pi0_pytorch

    from egomimic.algo.pi import PI

    hermetic_env(monkeypatch)
    recipe = RECIPES[(vendor, "pi")]
    is_eva = recipe.embodiment == "eva_bimanual"
    monkeypatch.setattr(pi0_pytorch, "PI0Pytorch", StubPI0)
    monkeypatch.setattr(
        StubPI0,
        "expect",
        {"state_dim": STATE_DIM[recipe.embodiment], "wrist_present": is_eva},
    )
    data, out, hashes = write_fixtures(tmp_path, vendor)
    cfg = compose_recipe(
        recipe,
        common_overrides(
            recipe.embodiment,
            data,
            out,
            batch_size=2,
            num_workers=0,
            episode_hashes=hashes,
        )
        + cpu_trainer_overrides(STEPS)
        + pi_cpu_overrides(),
        out,
    )
    spy = BatchKeySpy(monkeypatch, PI)
    _run(cfg)
    # The wrapper sees dataset camera names; StubPI0 asserts it was handed
    # openpi's slot names, which is the remap working end to end.
    seen = spy.keys[recipe.embodiment]
    expected = _expected_keys(recipe.embodiment) | {"annotations"}
    assert (
        expected <= seen
    ), f"{vendor}/pi missing {expected - seen}; saw {sorted(seen)}"


def test_flowvla_train_step(tmp_path, monkeypatch):
    """FlowVLA + ResNetTextBackbone takes two optimizer steps on synthetic mecka
    keypoint episodes through the real trainHydra.train() path."""
    hermetic_env(monkeypatch)
    recipe = RECIPES[("mecka", "flowvla")]
    data, out, hashes = write_fixtures(tmp_path, "mecka")
    cfg = compose_recipe(
        recipe,
        common_overrides(
            recipe.embodiment,
            data,
            out,
            batch_size=2,
            num_workers=0,
            episode_hashes=hashes,
        )
        + cpu_trainer_overrides(STEPS)
        + flowvla_small_overrides(),
        out,
    )
    _run(cfg)


def test_abc_dit_train_step(tmp_path, monkeypatch):
    """ABC-DiT takes two optimizer steps on synthetic mecka keypoint episodes
    through the real trainHydra.train() path, with both pretrained encoders
    stubbed out."""
    hermetic_env(monkeypatch)
    recipe = RECIPES[("mecka", "abc_dit")]
    data, out, hashes = write_fixtures(tmp_path, "mecka")
    cfg = compose_recipe(
        recipe,
        common_overrides(
            recipe.embodiment,
            data,
            out,
            batch_size=2,
            num_workers=0,
            episode_hashes=hashes,
        )
        + cpu_trainer_overrides(STEPS)
        + abc_dit_small_overrides(),
        out,
    )
    _run(cfg)
