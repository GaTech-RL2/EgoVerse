"""CPU tier: every (vendor, algo) recipe builds and takes 2 optimizer steps on
synthetic data through the real trainHydra.train() path. Pi's openpi network
is a stub; the EgoVerse Pi wrapper (keymaps, converters, prompts, batch
assembly) is real. Dexmate has an HPT keypoints recipe only, plus a checkpoint
save/reload (the shared overrides disable checkpointing)."""

from __future__ import annotations

import math
import os

import pytest
import torch
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
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.embodiment.dexmate import Dexmate
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
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
    front = Eva.VIZ_IMAGE_KEY if is_eva else Human.VIZ_IMAGE_KEY
    expected = {"actions_cartesian", "observations.state.ee_pose", front, "annotations"}
    if is_eva:
        expected |= {
            "observations.images.left_wrist_img",
            "observations.images.right_wrist_img",
        }
    assert (
        expected <= seen
    ), f"{vendor}/pi missing {expected - seen}; saw {sorted(seen)}"


def _dexmate_overrides(data, out, hashes, max_epochs: int = 1) -> list[str]:
    trainer = [
        f"trainer.max_epochs={max_epochs}" if o.startswith("trainer.max_epochs=") else o
        for o in cpu_trainer_overrides(STEPS)
    ]
    emb = RECIPES[("dexmate", "hpt")].embodiment
    return (
        common_overrides(
            emb, data, out, batch_size=2, num_workers=0, episode_hashes=hashes
        )
        + trainer
        + hpt_small_overrides(emb)
    )


def test_hpt_train_step_dexmate(tmp_path, monkeypatch):
    hermetic_env(monkeypatch)
    recipe = RECIPES[("dexmate", "hpt")]
    data, out, hashes = write_fixtures(tmp_path, "dexmate")
    spy = BatchKeySpy(monkeypatch, HPT)
    _run(compose_recipe(recipe, _dexmate_overrides(data, out, hashes), out))
    seen = spy.keys[recipe.embodiment]
    expected = {
        "actions_keypoints",
        "observations.state.keypoints",
        Dexmate.VIZ_IMAGE_KEY,
        *Dexmate.WRIST_IMAGE_KEYS.values(),
    }
    assert (
        expected <= seen
    ), f"dexmate/hpt missing {expected - seen}; saw {sorted(seen)}"


def test_dexmate_hpt_checkpoint_saves_and_reloads(tmp_path, monkeypatch):
    hermetic_env(monkeypatch)
    recipe = RECIPES[("dexmate", "hpt")]
    emb_id = get_embodiment_id(recipe.embodiment)
    data, out, hashes = write_fixtures(tmp_path, "dexmate")
    cfg = compose_recipe(recipe, _dexmate_overrides(data, out, hashes), out)
    _, objects = train_hydra.train(cfg)
    assert objects["trainer"].global_step == STEPS
    ckpt = out / "dexmate.ckpt"
    objects["trainer"].save_checkpoint(ckpt)
    trained = objects["model"].state_dict()

    # The checkpoint alone rebuilds the model: its hyperparameters hold the
    # config tree and the norm stats, keyed by embodiment id 8.
    reloaded = ModelWrapper.load_from_checkpoint(
        ckpt, map_location="cpu", weights_only=False
    )
    assert reloaded.hparams.norm_stats_state["norm_stats"].keys() == {emb_id}
    assert reloaded.model.ac_keys[emb_id] == "actions_keypoints"
    assert sorted(reloaded.model.camera_keys[emb_id]) == sorted(
        [Dexmate.VIZ_IMAGE_KEY, *Dexmate.WRIST_IMAGE_KEYS.values()]
    )
    state = reloaded.state_dict()
    assert state.keys() == trained.keys()
    for key, value in trained.items():
        torch.testing.assert_close(state[key], value.cpu(), msg=key)

    # Lightning resumes from it, as a requeued run does.
    overrides = _dexmate_overrides(data, out, hashes, max_epochs=2)
    resumed = compose_recipe(recipe, overrides + [f"ckpt_path={ckpt}"], out)
    _, objects = train_hydra.train(resumed)
    assert objects["trainer"].global_step == 2 * STEPS
