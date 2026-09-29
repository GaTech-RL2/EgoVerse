"""CPU tier: every (vendor, algo) recipe builds and takes 2 optimizer steps on
synthetic data through the real trainHydra.train() path. Pi's openpi network
is a stub; the EgoVerse Pi wrapper (keymaps, converters, prompts, batch
assembly) is real."""

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
    rdt_small_overrides,
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
from egomimic.algo.hpt import HPT, HPTModel
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Human

VENDOR_NAMES = ["eva", "aria", "mecka", "scale"]
STEPS = 2


def _run(cfg) -> dict:
    metrics, objects = train_hydra.train(cfg)
    loss = float(metrics["Train/action_loss"])
    assert math.isfinite(loss), f"non-finite loss {loss}"
    assert objects["trainer"].global_step == STEPS
    return objects


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


def test_hpt_builds_the_policy_on_cpu_while_gpus_are_visible(tmp_path, monkeypatch):
    """Under submitit every rank sees all 8 GPUs, so an algo that picks its own
    construction device puts all 8 policies on cuda:0 and OOMs at 3B. Lightning
    owns the move, so nothing may leave CPU in ``__init__``. Also: HPT.__init__
    finalizes the policy exactly once (a second pass re-inits and re-draws the
    action tokens), and the model's enable_grad_norm reaches ModelWrapper."""
    hermetic_env(monkeypatch)
    recipe = RECIPES[("aria", "hpt")]
    data, out, hashes = write_fixtures(tmp_path, "aria")
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
        + hpt_small_overrides(recipe.embodiment)
        + ["+model.enable_grad_norm=false"],
        out,
    )

    finalized = []
    finalize = HPTModel.finalize_modules
    monkeypatch.setattr(
        HPTModel,
        "finalize_modules",
        lambda self: finalized.append(self) or finalize(self),
    )
    built = {}
    init = HPT.__init__

    def spy(self, *args, **kwargs):
        # Only __init__ sees the GPUs: faking them for the whole run would send
        # the rest of the stack down CUDA paths this CPU tier cannot take.
        with monkeypatch.context() as gpus_visible:
            gpus_visible.setattr(torch.cuda, "is_available", lambda: True)
            gpus_visible.setattr(torch.cuda, "device_count", lambda: 8)
            init(self, *args, **kwargs)
        built["param_devices"] = {p.device.type for p in self.nets.parameters()}
        built["algo_device"] = self.device
        built["policy_device"] = self.nets["policy"].device

    monkeypatch.setattr(HPT, "__init__", spy)
    objects = _run(cfg)

    assert len(finalized) == 1
    assert objects["model"].enable_grad_norm is False
    assert built["param_devices"] == {"cpu"}
    assert built["algo_device"] is None
    assert built["policy_device"] is None


def test_rdt_train_step(tmp_path, monkeypatch):
    hermetic_env(monkeypatch)
    recipe = RECIPES[("mecka", "rdt")]
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
        + rdt_small_overrides(recipe.embodiment),
        out,
    )
    spy = BatchKeySpy(monkeypatch, HPT)
    _run(cfg)
    expected = {
        "actions_keypoints",
        "observations.state.ee_pose",
        Human.VIZ_IMAGE_KEY,
        f"{Human.VIZ_IMAGE_KEY}_hist",
        "fps",
    }
    assert expected <= spy.keys[recipe.embodiment]


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


def test_val_heads_are_wired_from_the_evaluator(tmp_path, monkeypatch):
    """train() builds the canonical head (renamed by data.valid_prefix) and a
    train_viz head, both from cfg.evaluator, and hands each the pipeline's
    action stride. Every other train() test drops the evaluator."""
    hermetic_env(monkeypatch)
    recipe = RECIPES[("mecka", "hpt")]
    data, out, hashes = write_fixtures(tmp_path, "mecka")
    overrides = common_overrides(
        recipe.embodiment,
        data,
        out,
        batch_size=2,
        num_workers=0,
        episode_hashes=hashes,
    )
    overrides += cpu_trainer_overrides(STEPS) + hpt_small_overrides(recipe.embodiment)
    overrides = [
        o for o in overrides if o not in ("~evaluator", "trainer.limit_val_batches=0")
    ] + [
        "trainer.limit_val_batches=1",
        "evaluator.viz_every_n_epochs=0",
        "+data.valid_prefix=seen_op_valid",
    ]
    metrics, objects = train_hydra.train(compose_recipe(recipe, overrides, out))
    model = objects["model"]
    heads = (model.evaluator, model.train_viz_evaluator)
    assert [ev.prefix for ev in heads] == ["seen_op_valid", "train_viz"]
    for ev in heads:
        assert ev.viz_every_n_epochs == 0  # follows cfg.evaluator
        assert ev.action_stride == {"human_bimanual": 29 / 99}
    val_keys = [k for k in metrics if "Valid/" in k]
    assert val_keys and all(
        k.startswith(("seen_op_valid/Valid/", "train_viz/Valid/")) for k in val_keys
    ), val_keys
