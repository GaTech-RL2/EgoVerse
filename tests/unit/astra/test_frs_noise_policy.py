"""Synthetic CPU-only checks; no learned robot-control performance is measured."""

import copy
import json

import numpy as np
import pytest
import torch

from astra_reversal.frs_noise_policy import (
    AuxiliaryNoisePolicy,
    VisualNoiseMean,
    make_training_sample,
    noise_loss,
    preprocess_observation,
)
from astra_reversal.frs_operators import repeated_gaussian_noise
from astra_reversal.records import digest


@pytest.fixture(autouse=True)
def few_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observation(value=0):
    return {
        "observation/image": np.full((16, 20, 3), value, np.uint8),
        "observation/wrist_image": np.full((20, 16, 3), 255 - value, np.uint8),
        "observation/state": np.arange(8, dtype=np.float32) / 10,
    }


def sample(index=0, *, kind="frs_edit", target=None):
    noise = repeated_gaussian_noise(np.random.default_rng(index))["noise"]
    if target is not None:
        noise[..., :7] = np.asarray(target, np.float32)
    return make_training_sample(
        observation(index),
        noise,
        kind=kind,
        observation_step=index * 10,
        source_id=f"executed:{index}",
    )


def fit(actor, index=0, *, samples=None):
    return actor.fit_accepted_rollout(
        f"rollout:{index}",
        [sample(index)] if samples is None else samples,
        judge_verdict="better",
        judge_sha256="a" * 64,
    )


def test_untrained_prediction_is_exact_base_and_consumes_no_rng_or_observation():
    actor = AuxiliaryNoisePolicy("synthetic-task", 7, device="cpu")
    rng = np.random.default_rng(2)
    base = repeated_gaussian_noise(np.random.default_rng(3))["noise"]
    before = digest(rng.bit_generator.state)
    result = actor.predict(object(), rng, base)  # No encoder before an update.
    np.testing.assert_array_equal(result["noise"], base)
    assert result["noise"] is not base
    assert digest(rng.bit_generator.state) == before
    assert not result["receipt"]["auxiliary_forward_performed"]
    assert result["receipt"]["source"] == "exact_untrained_base_fallback"


def test_initialization_is_task_seeded_and_does_not_change_global_torch_rng():
    state = torch.random.get_rng_state().clone()
    a = AuxiliaryNoisePolicy("task-a", 8, device="cpu")
    b = AuxiliaryNoisePolicy("task-a", 8, device="cpu")
    c = AuxiliaryNoisePolicy("task-b", 8, device="cpu")
    assert torch.equal(state, torch.random.get_rng_state())
    assert a.metadata()["parameter_sha256"] == b.metadata()["parameter_sha256"]
    assert a.metadata()["parameter_sha256"] != c.metadata()["parameter_sha256"]


def test_preprocessing_preserves_camera_order_and_proprio_without_mutating_source():
    raw = observation(20)
    before = digest(raw)
    images, state = preprocess_observation(raw)
    assert images.shape == (2, 84, 84, 3) and images.dtype == np.uint8
    assert np.all(images[0] == 20) and np.all(images[1] == 235)
    np.testing.assert_array_equal(state, raw["observation/state"])
    assert digest(raw) == before


def test_encoder_has_gradients_for_both_cameras_and_proprio_and_bounded_output():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(9)
        model = VisualNoiseMean()
    images = torch.ones((2, 2, 3, 84, 84), requires_grad=True)
    state = torch.ones((2, 8), requires_grad=True)
    output = model(images, state)
    assert output.shape == (2, 7) and torch.all(output.abs() <= 5)
    output.square().sum().backward()
    assert images.grad[:, 0].abs().sum() > 0
    assert images.grad[:, 1].abs().sum() > 0
    assert state.grad.abs().sum() > 0


def test_loss_uses_raw_targets_and_declared_mean_penalty():
    predicted = torch.full((2, 7), 4.0, requires_grad=True)
    target = torch.full((2, 7), 7.0)
    total, mse, regularization = noise_loss(predicted, target)
    assert mse.item() == 9
    assert regularization.item() == 16
    assert total.item() == pytest.approx(9.016)
    total.backward()
    assert torch.all(predicted.grad < 0)  # Target was not silently clipped to 5.


def test_sample_binds_exact_executed_noise_for_both_edit_and_defer():
    for kind in ("frs_edit", "native_defer"):
        item = sample(kind=kind)
        assert item["metadata"]["kind"] == kind
        np.testing.assert_array_equal(item["target"], item["executed_noise"][0, 0, :7])
        assert item["metadata"]["executed_noise_sha256"] == digest(
            item["executed_noise"]
        )
    noise = repeated_gaussian_noise(np.random.default_rng(3))["noise"]
    noise[0, 1, 0] += 0.01
    with pytest.raises(ValueError, match="identical"):
        make_training_sample(
            observation(), noise, kind="frs_edit", observation_step=0, source_id="bad"
        )


@pytest.mark.parametrize("verdict", ["same", "worse", "unknown", "success"])
def test_non_better_judgment_leaves_all_state_unchanged(verdict):
    actor = AuxiliaryNoisePolicy("test", 7, device="cpu")
    before = actor.metadata()
    result = actor.fit_accepted_rollout(
        "rejected", object(), judge_verdict=verdict, judge_sha256="b" * 64
    )
    assert result["optimizer_updates"] == 0
    assert actor.metadata() == before


def test_empty_approved_rollout_does_not_initialize_or_update():
    actor = AuxiliaryNoisePolicy("test", 7, device="cpu")
    before = actor.metadata()
    assert fit(actor, samples=[])["reason"] == "no_executed_samples"
    assert actor.metadata() == before


def test_production_fit_refuses_cpu_before_any_state_mutation():
    actor = AuxiliaryNoisePolicy("test", 7, device="cpu")
    before = actor.metadata()
    with pytest.raises(RuntimeError, match="requires CUDA"):
        fit(actor)
    assert actor.metadata() == before


def test_synthetic_update_changes_parameters_and_replays_both_sample_kinds():
    actor = AuxiliaryNoisePolicy("test", 7, device="cpu", _test_only_cpu_steps=2)
    before = actor.metadata()["parameter_sha256"]
    first = fit(
        actor,
        samples=[
            sample(0, target=[6, 0, 0, 0, 0, 0, 0]),
            sample(1, kind="native_defer"),
        ],
    )
    assert first["optimizer_updates"] == 2 and first["synthetic_cpu_test_only"]
    assert actor.metadata()["parameter_sha256"] != before
    assert first["target_out_of_range_fraction"] == pytest.approx(1 / 14)
    assert first["targets_raw"][0, 0] == 6
    assert first["targets_clipped_for_diagnostics_only"][0, 0] == 5
    assert not first["loss_targets_clipped"]
    second = fit(actor, 2)
    assert actor.rounds == 2 and second["replay_samples"] == 3
    assert second["sample_kinds"] == {"frs_edit": 2, "native_defer": 1}
    np.testing.assert_array_equal(second["targets_raw"][:2], first["targets_raw"])
    assert second["loss_trace"].shape == (2, 3)


def test_learned_prediction_has_deterministic_mean_and_independent_padding():
    actor = AuxiliaryNoisePolicy("test", 7, device="cpu", _test_only_cpu_steps=1)
    fit(actor)
    base = repeated_gaussian_noise(np.random.default_rng(1))["noise"]
    a = actor.predict(observation(), np.random.default_rng(2), base)
    b = actor.predict(observation(), np.random.default_rng(3), base)
    np.testing.assert_array_equal(a["mean_first7"], b["mean_first7"])
    np.testing.assert_array_equal(
        a["noise"][0, :, :7], np.broadcast_to(a["mean_first7"], (10, 7))
    )
    assert not np.array_equal(a["noise"][..., 7:], b["noise"][..., 7:])
    np.testing.assert_array_equal(
        a["noise"][..., 7:],
        np.random.default_rng(2).standard_normal((1, 10, 25)).astype(np.float32),
    )


def test_checkpoint_restores_optimizer_replay_and_untrained_fallback(tmp_path):
    actor = AuxiliaryNoisePolicy("test", 7, device="cpu", _test_only_cpu_steps=1)
    initial = actor.metadata()
    actor.save(tmp_path / "initial")
    fit(actor)
    first = actor.metadata()
    actor.save(tmp_path / "round1")
    fit(actor, 1)
    actor.load(tmp_path / "round1")
    assert actor.metadata() == first
    actor.load(tmp_path / "initial")
    assert actor.metadata() == initial and not actor.trained
    assert actor.replay == actor.history == actor.accepted_rollouts == []


def test_restored_optimizer_produces_same_next_parameters(tmp_path):
    actor = AuxiliaryNoisePolicy("test", 7, device="cpu", _test_only_cpu_steps=1)
    fit(actor)
    actor.save(tmp_path / "round1")
    fit(actor, 1)
    expected = actor.metadata()["parameter_sha256"]
    actor.load(tmp_path / "round1")
    fit(actor, 1)
    assert actor.metadata()["parameter_sha256"] == expected


def test_failed_update_rolls_back_and_restores_backend_flags(monkeypatch):
    actor = AuxiliaryNoisePolicy("test", 7, device="cpu", _test_only_cpu_steps=2)
    before = actor.metadata()
    flags = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cuda.matmul.allow_tf32,
    )
    original = actor.optimizer.step
    calls = []

    def fail_after_mutation():
        original()
        calls.append(True)
        raise RuntimeError("synthetic optimizer failure")

    monkeypatch.setattr(actor.optimizer, "step", fail_after_mutation)
    with pytest.raises(RuntimeError, match="synthetic optimizer failure"):
        fit(actor)
    assert calls and actor.metadata() == before
    assert (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cuda.matmul.allow_tf32,
    ) == flags


@pytest.mark.parametrize("tamper", ["file", "manifest", "task", "test_scope"])
def test_checkpoint_corruption_or_wrong_identity_does_not_mutate_live_state(
    tmp_path, tamper
):
    actor = AuxiliaryNoisePolicy("test", 7, device="cpu", _test_only_cpu_steps=1)
    actor.save(tmp_path / "checkpoint")
    if tamper == "file":
        with (tmp_path / "checkpoint/state.pt").open("ab") as stream:
            stream.write(b"tamper")
    elif tamper == "manifest":
        path = tmp_path / "checkpoint/manifest.json"
        data = json.loads(path.read_text())
        data["policy"]["parameter_sha256"] = "f" * 64
        path.write_text(json.dumps(data))
    elif tamper == "task":
        actor = AuxiliaryNoisePolicy(
            "different-task", 7, device="cpu", _test_only_cpu_steps=1
        )
    else:
        actor = AuxiliaryNoisePolicy("test", 7, device="cpu")
    before = actor.metadata()
    with pytest.raises(ValueError, match="mismatch"):
        actor.load(tmp_path / "checkpoint")
    assert actor.metadata() == before


def test_replay_rejects_modified_arrays_duplicate_sources_and_fourth_round():
    actor = AuxiliaryNoisePolicy("test", 7, device="cpu", _test_only_cpu_steps=1)
    bad = copy.deepcopy(sample())
    bad["target"][0] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        fit(actor, samples=[bad])
    fit(actor)
    with pytest.raises(ValueError, match="already included"):
        fit(actor)
    with pytest.raises(ValueError, match="Duplicate"):
        fit(actor, 1, samples=[sample()])
    fit(actor, 1)
    fit(actor, 2)
    before = actor.metadata()
    with pytest.raises(ValueError, match="three"):
        fit(actor, 3)
    assert actor.metadata() == before
