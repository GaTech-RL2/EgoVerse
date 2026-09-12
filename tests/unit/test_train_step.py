"""CPU tier: every (vendor, algo) recipe builds and takes 2 optimizer steps on
synthetic data through the real trainHydra.train() path. Pi's openpi network
is a stub; the EgoVerse Pi wrapper (keymaps, converters, prompts, batch
assembly) is real."""

from __future__ import annotations

import math

import pytest
from fixtures.recipes import (
    RECIPES,
    common_overrides,
    cpu_trainer_overrides,
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


@pytest.mark.parametrize("vendor", VENDOR_NAMES)
def test_hpt_train_step(tmp_path, monkeypatch, vendor):
    hermetic_env(monkeypatch)
    recipe = RECIPES[(vendor, "hpt")]
    data, out = write_fixtures(tmp_path, vendor)
    cfg = compose_recipe(
        recipe,
        common_overrides(recipe.embodiment, data, out, batch_size=2, num_workers=0)
        + cpu_trainer_overrides(STEPS)
        + hpt_small_overrides(recipe.embodiment),
        out,
    )
    spy = BatchKeySpy(monkeypatch, HPT)
    _run(cfg)
    seen = spy.keys[recipe.embodiment]
    is_eva = recipe.embodiment == "eva_bimanual"
    front = Eva.VIZ_IMAGE_KEY if is_eva else Human.VIZ_IMAGE_KEY
    expected = {"actions_cartesian", "observations.state.ee_pose", front}
    if is_eva:
        expected |= {
            "observations.images.right_wrist_img",
            "observations.images.left_wrist_img",
        }
    assert (
        expected <= seen
    ), f"{vendor}/hpt missing {expected - seen}; saw {sorted(seen)}"


@pytest.mark.parametrize("vendor", VENDOR_NAMES)
def test_pi_train_step(tmp_path, monkeypatch, vendor):
    reason = pi_unavailable()
    if reason:
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
    data, out = write_fixtures(tmp_path, vendor)
    cfg = compose_recipe(
        recipe,
        common_overrides(recipe.embodiment, data, out, batch_size=2, num_workers=0)
        + cpu_trainer_overrides(STEPS)
        + pi_cpu_overrides(),
        out,
    )
    spy = BatchKeySpy(monkeypatch, PI)
    _run(cfg)
    seen = spy.keys[recipe.embodiment]
    expected = {
        "actions_cartesian",
        "observations.state.ee_pose",
        "base_0_rgb",
        "annotations",
    }
    if is_eva:
        expected |= {"left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert (
        expected <= seen
    ), f"{vendor}/pi missing {expected - seen}; saw {sorted(seen)}"
