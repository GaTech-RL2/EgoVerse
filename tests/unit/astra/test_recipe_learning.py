"""Synthetic CPU evidence for masked learning; no robot-performance claim."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from astra_reversal import flow
from astra_reversal import recipe_learning as learning
from astra_reversal.action_adapter import ActionAdapter, ActionSpec
from astra_reversal.policy_adapter import Condition
from astra_reversal.records import digest, file_sha256, to_numpy


class NativeFlow(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            self.action_out_proj = nn.Linear(1024, 32)
            self.action_out_proj.weight.data.mul_(0.1)
            self.action_out_proj.bias.data.mul_(0.1)
        self.fail = False
        self.last_features = None

    def velocity(self, x, timestep, signal):
        if self.fail:
            raise RuntimeError("synthetic denoise failure")
        features = torch.zeros((*x.shape[:2], 1024), dtype=torch.float32)
        features[..., :32] = x
        features[..., 32:64] = x.mean(dim=1, keepdim=True)
        features[..., 64] = timestep
        features[..., 65] = signal
        self.last_features = features.clone()
        return self.action_out_proj(features)


class Adapter:
    def __init__(self):
        self.policy = nn.Module()
        self.policy.model = NativeFlow()
        self.policy.eval().requires_grad_(False)
        self.model = self.policy.model
        self.device = torch.device("cpu")
        self.horizon, self.action_dim = 10, 32
        self.config = SimpleNamespace(compile_model=False, gradient_checkpointing=False)
        self.metadata = {
            "input_profile": "openpi_libero",
            "artifact_sha256": {"synthetic": "1" * 64},
            "model_source_sha256": "2" * 64,
            "transformers_source_sha256": {"synthetic": "8" * 64},
            "torch_version": str(torch.__version__),
            "transformers_version": "synthetic",
            "adapter_source_sha256": "3" * 64,
            "processor_source_sha256": {"synthetic": "4" * 64},
            "input_profile_source_sha256": "5" * 64,
            "input_profile_assets": {"synthetic": "6" * 64},
            "normalization": {"type": "nonidentity_synthetic_quantiles"},
            "tokenizer_sha256": "7" * 64,
        }
        self.prepared = []
        self.q01 = np.arange(7, dtype=np.float64) / 10 - 2
        self.q99 = self.q01 + 3

    def tensor(self, value):
        return torch.tensor(np.array(value), dtype=torch.float32)

    def prepare(self, observation, observation_id, prompt):
        raw = {**copy.deepcopy(observation), "prompt": prompt}
        self.prepared.append(digest(raw))
        signal = observation["observation/image"].mean() / 255 + len(prompt) / 100
        return Condition(
            digest(raw),
            observation_id,
            prompt,
            raw,
            self.tensor(np.zeros((1, 32))),
            lambda x, t: self.model.velocity(x, t, signal),
            0.0,
        )

    def sample(self, condition, noise, **kwargs):
        return flow.generate(condition.velocity, noise, **kwargs)

    def input_transform(self, raw):
        physical = (
            (np.asarray(raw["actions"]) - self.q01) / (self.q99 - self.q01 + 1e-6) * 2
            - 1
        ).astype(np.float32)
        return {
            "actions": np.pad(physical, ((0, 0), (0, 25))),
            "state": np.zeros(32, np.float32),
        }

    def output_transform(self, raw):
        return {
            "actions": (raw["actions"][..., :7] + 1) / 2 * (self.q99 - self.q01 + 1e-6)
            + self.q01
        }


@pytest.fixture(autouse=True)
def one_thread():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


@pytest.fixture
def policy():
    return Adapter()


@pytest.fixture
def observation():
    return {
        "observation/image": np.full((224, 224, 3), 30, np.uint8),
        "observation/wrist_image": np.full((224, 224, 3), 80, np.uint8),
        "observation/state": np.arange(8, dtype=np.float32) / 10,
    }


def adapter(policy):
    return ActionAdapter(
        ActionSpec("synthetic", 10, 32, 0.05, (-1,) * 7, (1,) * 7, {}),
        policy.input_transform,
        policy.output_transform,
    )


def imputation(policy, observation, prompt="original task"):
    return learning.native_imputation(
        policy, observation, prompt, "synthetic-window", test_only_allow_cpu=True
    )


def window(policy, observation, *, actions=None, native=None):
    return learning.make_executed_window(
        policy,
        adapter(policy),
        observation,
        original_prompt="original task",
        executed_actions=np.full((3, 7), 0.25, np.float32)
        if actions is None
        else actions,
        native_chunk=imputation(policy, observation) if native is None else native,
        source_id="synthetic-window",
        source_receipt_sha256="a" * 64,
    )


def bank(policy, observation, *, anchor=False):
    return learning.capture_samples(
        policy,
        adapter(policy),
        observation,
        "original task",
        np.full((3, 7), 0.25, np.float32),
        "synthetic-window" + str(anchor),
        sample_id="synthetic-window" + str(anchor),
        source_receipt_sha256="a" * 64,
        anchor=anchor,
        test_only_allow_cpu=True,
    )


def weights(policy):
    return digest(
        {name: to_numpy(value) for name, value in policy.policy.state_dict().items()}
    )


def synthetic_fit_bank(identity, *, anchor=False, cue=1.0):
    count = 5
    n = 8 * count
    h = np.zeros((n, 1024), np.float32)
    h[:, 0] = cue
    noise = np.full((8, 1, 10, 32), 0.2, np.float32)
    native = np.zeros((n, 7), np.float32)
    target = native.copy() if anchor else np.full((n, 7), 0.2, np.float32)
    return learning.FeatureBank(
        {
            "h": h,
            "noise": noise,
            "times": np.full(8, 0.5, np.float32),
            "x0": np.zeros((1, 10, 32), np.float32),
            "native_velocity": native,
            "full_native_velocity": np.zeros((8, 1, 10, 32), np.float32),
            "target_velocity": target,
            "residual_target": target - native,
            "action_row": np.tile(np.arange(count, dtype=np.int64), 8),
            "draw_index": np.repeat(np.arange(8, dtype=np.int64), count),
        },
        {
            "schema_version": learning.SCHEMA,
            "config": learning.training_config(),
            "source_sha256": file_sha256(learning.__file__),
            "kind": "anchor" if anchor else "executed",
            "base_identity": identity,
            "test_only": True,
            "window": {"executed_count": 0 if anchor else count},
            "counts": {"velocity_evaluations": 0, "feature_rows": n},
            "fixture": "synthetic detached feature fixture, not native model evidence",
        },
    )


def test_frozen_config_and_exact_zero_parameters_do_not_consume_global_rng(policy):
    state = torch.random.get_rng_state().clone()
    head = learning.ResidualActionHead(learning.base_identity(policy))
    assert torch.equal(state, torch.random.get_rng_state())
    assert head.is_zero() and sum(p.numel() for p in head.parameters()) == 7175
    config = learning.training_config()
    assert (
        config["optimizer_steps"],
        config["learning_rate"],
        config["batch_feature_rows"],
        config["seed"],
    ) == (1000, 1e-4, 128, 61)
    assert config["anchor_weight"] == 1 and config["l2_weight"] == 1e-4
    assert config["apply_rows"] == config["anchor_rows"] == 5


def test_fixed_native_draws_are_keyed_and_do_not_consume_global_numpy_rng():
    before = digest(np.random.get_state())
    a, b, c = (learning.fixed_draws(key) for key in ("one", "one", "two"))
    assert digest(a) == digest(b) != digest(c)
    assert digest(np.random.get_state()) == before
    assert a["noise"].shape == (8, 1, 10, 32)
    assert a["times"].dtype == np.float32
    assert np.all((a["times"] >= 0.001) & (a["times"] <= 1))


def test_recorded_json_list_action_spec_encodes_exactly(policy, observation):
    original = adapter(policy)
    recorded_spec = ActionSpec(**json.loads(json.dumps(original.spec.as_dict())))
    restored = ActionAdapter(
        recorded_spec, policy.input_transform, policy.output_transform
    )
    actions = np.full((3, 7), 0.25, np.float32)
    native = imputation(policy, observation)
    expected = window(policy, observation, actions=actions, native=native)
    actual = learning.make_executed_window(
        policy,
        restored,
        observation,
        original_prompt="original task",
        executed_actions=actions,
        native_chunk=native,
        source_id="synthetic-window",
        source_receipt_sha256="a" * 64,
    )
    assert digest(actual) == digest(expected)


def test_resealed_bank_wrong_source_or_flow_time_is_rejected(policy, observation):
    original = bank(policy, observation)
    for key, value in (("source_sha256", "f" * 64), ("schema_version", "wrong")):
        provenance = copy.deepcopy(original.provenance)
        provenance[key] = value
        with pytest.raises(ValueError, match="source/config"):
            learning.FeatureBank(original.arrays, provenance)
    arrays = {name: value.copy() for name, value in original.arrays.items()}
    arrays["times"][0] = 0
    with pytest.raises(ValueError, match="native support"):
        learning.FeatureBank(arrays, original.provenance)


def test_actual_normalized_prefix_and_imputed_suffix_have_distinct_roles(
    policy, observation
):
    native = imputation(policy, observation)
    before = digest({"observation": observation, "native": native})
    actions = np.full((2, 7), 0.35, np.float32)
    result = window(policy, observation, actions=actions, native=native)
    expected = (
        (actions - policy.q01) / (policy.q99 - policy.q01 + 1e-6) * 2 - 1
    ).astype(np.float32)
    np.testing.assert_array_equal(result["x0"][0, :2, :7], expected)
    np.testing.assert_array_equal(result["x0"][0, 2:, :7], native["actions"][0, 2:, :7])
    assert np.any(native["actions"][..., 7:] != 0)
    assert not np.any(result["x0"][..., 7:])
    assert result["label_mask"].sum() == 2 * 7
    assert not np.any(result["label_mask"][:, 2:])
    assert digest({"observation": observation, "native": native}) == before


@pytest.mark.parametrize("count", [0, 6, 10])
def test_only_executed_first_one_to_five_rows_are_accepted(policy, observation, count):
    with pytest.raises(ValueError, match="K<=5"):
        window(policy, observation, actions=np.zeros((count, 7), np.float32))


@pytest.mark.parametrize("change", ["task", "camera", "state", "native_array", "base"])
def test_wrong_imputation_condition_or_modified_native_chunk_fails_closed(
    policy, observation, change
):
    native = imputation(policy, observation)
    raw = copy.deepcopy(observation)
    if change == "task":
        native["provenance"]["original_prompt"] = "different task"
    elif change == "camera":
        raw["observation/image"][0, 0, 0] += 1
    elif change == "state":
        raw["observation/state"][0] += 0.01
    elif change == "native_array":
        native["actions"][0, 9, 1] += 0.1
    else:
        native["provenance"]["base_identity"]["hidden_dim"] = 2
    with pytest.raises(ValueError, match="imputation"):
        window(policy, raw, native=native)


@pytest.mark.parametrize("value", [np.nan, np.inf, 1.00001])
def test_invalid_executed_labels_are_not_silently_clipped(policy, observation, value):
    labels = np.zeros((2, 7), np.float64)
    labels[0, 0] = value
    with pytest.raises(ValueError):
        window(policy, observation, actions=labels)


def test_original_task_and_adaptation_split_are_required(policy, observation):
    with pytest.raises(ValueError, match="original task"):
        imputation(policy, {**observation, "prompt": "modified instruction"})
    with pytest.raises(ValueError, match="adaptation"):
        learning.make_executed_window(
            policy,
            adapter(policy),
            observation,
            original_prompt="original task",
            executed_actions=np.zeros((2, 7), np.float32),
            native_chunk={},
            source_id="id",
            source_receipt_sha256="a" * 64,
            split="heldout",
        )


def test_captured_native_features_targets_and_counts_exclude_unexecuted_labels(
    policy, observation
):
    before = weights(policy)
    value = bank(policy, observation)
    assert value.h.shape == (8 * 3, 1024)
    assert value.residual_target.shape == (8 * 3, 7)
    assert (
        value.counts["velocity_evaluations"] == 18
        and value.counts["prefix_preparations"] == 2
    )
    assert np.max(value.arrays["action_row"]) == 2
    for index in range(8):
        expected = (
            value.arrays["noise"][index, 0, :3, :7] - value.arrays["x0"][0, :3, :7]
        )
        np.testing.assert_array_equal(
            value.arrays["target_velocity"][index * 3 : (index + 1) * 3], expected
        )
    assert weights(policy) == before
    assert all(
        p.grad is None and not p.requires_grad for p in policy.policy.parameters()
    )
    assert (
        not policy.model.action_out_proj._forward_hooks
        and not policy.model.action_out_proj._forward_pre_hooks
    )
    assert len(policy.prepared) == 2 and len(set(policy.prepared)) == 1


def test_anchor_targets_native_velocity_not_recorded_controller_labels(
    policy, observation
):
    value = learning.capture_samples(
        policy,
        adapter(policy),
        observation,
        "original task",
        object(),
        "anchor-window",
        sample_id="anchor",
        source_receipt_sha256="a" * 64,
        anchor=True,
        test_only_allow_cpu=True,
    )
    assert value.h.shape == (40, 1024) and not np.any(value.residual_target)
    np.testing.assert_array_equal(
        value.arrays["target_velocity"], value.arrays["native_velocity"]
    )
    assert not value.provenance["anchor_recorded_actions_used_as_labels"]


def test_resealed_unexecuted_mask_still_cannot_become_labels(policy, observation):
    value = window(policy, observation)
    value["label_mask"][0, 7, :7] = True
    value["provenance"]["label_mask_sha256"] = digest(value["label_mask"])
    with pytest.raises(ValueError, match="mask binding"):
        learning.capture_flow_features(
            policy, value, **learning.fixed_draws("x"), test_only_allow_cpu=True
        )


def test_native_failure_and_foreign_hooks_are_cleaned_or_preserved(policy, observation):
    value = window(policy, observation)
    policy.model.fail = True
    with pytest.raises(RuntimeError, match="synthetic denoise"):
        learning.capture_flow_features(
            policy, value, **learning.fixed_draws("x"), test_only_allow_cpu=True
        )
    assert not policy.model.action_out_proj._forward_pre_hooks
    policy.model.fail = False
    external = policy.model.action_out_proj.register_forward_hook(lambda *_: None)
    with pytest.raises(ValueError, match="active hook"):
        learning.capture_flow_features(
            policy, value, **learning.fixed_draws("x"), test_only_allow_cpu=True
        )
    assert len(policy.model.action_out_proj._forward_hooks) == 1
    external.remove()


def test_zero_head_has_exact_native_condition_and_full_chunk_parity(
    policy, observation
):
    head = learning.ResidualActionHead(learning.base_identity(policy), test_only=True)
    native = policy.prepare(observation, "obs", "original task")
    adapted, proof = learning.prepare_adapted(
        policy, observation, "obs", "original task", head
    )
    noise = policy.tensor(
        np.random.default_rng(8).standard_normal((1, 10, 32)).astype(np.float32)
    )
    a = policy.sample(native, noise, steps=10).value
    b = policy.sample(adapted, noise, steps=10).value
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    assert adapted.condition_id == native.condition_id and not proof["enabled"]
    assert proof["projection_calls"] == 0


def test_nonzero_head_writes_only_first_five_physical_projection_rows(
    policy, observation
):
    head = learning.ResidualActionHead(learning.base_identity(policy), test_only=True)
    with torch.no_grad():
        head.linear.bias.fill_(1.0)
    before = weights(policy)
    native = policy.prepare(observation, "obs", "original task")
    adapted, proof = learning.prepare_adapted(
        policy, observation, "obs", "original task", head
    )
    x = torch.randn(1, 10, 32)
    a, b = native.velocity(x, 0.4), adapted.velocity(x, 0.4)
    assert not torch.equal(a[:, :5, :7], b[:, :5, :7])
    assert torch.equal(a[:, 5:].view(torch.int32), b[:, 5:].view(torch.int32))
    assert torch.equal(
        a[..., 7:].contiguous().view(torch.int32),
        b[..., 7:].contiguous().view(torch.int32),
    )
    assert 0 < proof["residual_max_abs"] < 0.5
    assert proof["raw_outside_bound_values"] == proof["residual_values"] == 35
    assert weights(policy) == before and not policy.model.action_out_proj._forward_hooks
    assert digest(adapted.raw) == digest(native.raw)
    with torch.no_grad():
        head.linear.bias.add_(0.01)
    with pytest.raises(ValueError, match="updated head"):
        adapted.velocity(x, 0.4)


def test_feature_bank_roundtrip_and_tamper_rejection(policy, observation, tmp_path):
    value = bank(policy, observation)
    folder = tmp_path / "features"
    manifest = value.save(folder)
    loaded = learning.FeatureBank.load(folder)
    assert loaded.metadata() == {
        k: v for k, v in manifest.items() if k != "arrays_file_sha256"
    }
    np.testing.assert_array_equal(loaded.h, value.h)
    value.arrays["h"].flags.writeable = True
    value.arrays["h"][0, 0] += 1
    with pytest.raises(ValueError, match="binding changed"):
        value.verify()
    data = (folder / "arrays.npz").read_bytes()
    (folder / "arrays.npz").write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match="file hash"):
        learning.FeatureBank.load(folder)


def test_real_adam_path_reduces_masked_loss_and_teacher_anchor_limits_drift(policy):
    identity = learning.base_identity(policy)
    data = synthetic_fit_bank(identity)
    anchors = synthetic_fit_bank(identity, anchor=True)
    before = weights(policy)
    head = learning.ResidualActionHead(identity, test_only=True)
    receipt = head.fit([data], [anchors], admission_receipt_sha256="b" * 64)
    assert (
        receipt["optimizer_steps"] == 1000 and receipt["trainable_parameters"] == 7175
    )
    assert (
        receipt["after"]["masked_residual_mse"]
        < receipt["before"]["masked_residual_mse"]
    )
    # Same-feature anchor wants zero, correction wants0.2. The compromise
    # detects an omitted anchor objective or training on the wrong targets.
    actual = head(torch.tensor(data.h.copy())).detach().numpy()
    assert 0.07 < actual.mean() < 0.13
    assert receipt["base_backward_passes"] == receipt["velocity_evaluations"] == 0
    assert weights(policy) == before
    assert all(p.grad is None for p in policy.policy.parameters())


def test_fit_determinism_save_load_and_rollback_preserve_exact_head(policy, tmp_path):
    identity = learning.base_identity(policy)
    data, anchors = (
        synthetic_fit_bank(identity),
        synthetic_fit_bank(identity, anchor=True, cue=0),
    )
    a, b = (learning.ResidualActionHead(identity, test_only=True) for _ in range(2))
    zero = a.snapshot()
    before_rng = torch.random.get_rng_state().clone()
    for head in (a, b):
        head.fit([data], [anchors], admission_receipt_sha256="b" * 64, _test_steps=30)
    assert a.parameter_sha256() == b.parameter_sha256()
    assert torch.equal(before_rng, torch.random.get_rng_state())
    folder = tmp_path / "head"
    a.save(folder)
    restored = learning.ResidualActionHead.load(
        folder, expected_base_identity=identity, allow_test_only=True
    )
    assert restored.metadata() == a.metadata()
    assert torch.equal(
        restored(torch.tensor(data.h.copy())), a(torch.tensor(data.h.copy()))
    )
    with pytest.raises(ValueError, match="test scope"):
        learning.ResidualActionHead.load(folder, expected_base_identity=identity)
    a.restore(zero)
    assert a.is_zero() and a.optimizer_steps == 0 and not a.history
    path = folder / "state.pt"
    value = path.read_bytes()
    path.write_bytes(value + b"corruption")
    with pytest.raises(ValueError, match="file hash"):
        learning.ResidualActionHead.load(
            folder, expected_base_identity=identity, allow_test_only=True
        )


def test_cpu_production_training_or_extraction_is_refused(policy, observation):
    identity = learning.base_identity(policy)
    head = learning.ResidualActionHead(identity)
    with pytest.raises(ValueError, match="requires CUDA"):
        head.fit([], [], admission_receipt_sha256="b" * 64)
    with pytest.raises(ValueError, match="requires CUDA"):
        learning.native_imputation(policy, observation, "original task", "x")


def test_training_rejects_role_duplicates_base_drift_and_empty_anchors(policy):
    identity = learning.base_identity(policy)
    head = learning.ResidualActionHead(identity, test_only=True)
    data, anchors = (
        synthetic_fit_bank(identity),
        synthetic_fit_bank(identity, anchor=True),
    )
    for corrections, retained in (
        ([data], []),
        (
            [data, data],
            [anchors],
        ),
        ([anchors], [data]),
    ):
        with pytest.raises(ValueError):
            head.fit(
                corrections, retained, admission_receipt_sha256="b" * 64, _test_steps=2
            )
    other = copy.deepcopy(identity)
    other["native"]["artifact_sha256"] = {"wrong": "f" * 64}
    with pytest.raises(ValueError, match="identity differs"):
        head.fit(
            [synthetic_fit_bank(other)],
            [anchors],
            admission_receipt_sha256="b" * 64,
            _test_steps=2,
        )
    assert head.is_zero()


def test_nonfinite_gradient_failure_rolls_back_trainable_state(policy):
    identity = learning.base_identity(policy)
    head = learning.ResidualActionHead(identity, test_only=True)
    data, anchors = (
        synthetic_fit_bank(identity, cue=1e38),
        synthetic_fit_bank(identity, anchor=True),
    )
    before = head.metadata()
    deterministic = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(RuntimeError):
        head.fit([data], [anchors], admission_receipt_sha256="b" * 64, _test_steps=2)
    assert head.metadata() == before
    assert torch.are_deterministic_algorithms_enabled() == deterministic
