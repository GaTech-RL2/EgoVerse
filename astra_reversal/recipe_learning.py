"""Constrained native-flow regression on executed controller-action windows.

Only a separate zero-initialized 1024->7 affine residual is learned. It reads
the frozen final action-expert features immediately before action_out_proj.
Native image/text conditioning and every base parameter remain unchanged.

This is masked flow matching with an explicit native-generated suffix input,
not full-horizon demonstration supervision. Action attention couples rows, so
the imputed suffix can affect the features even though it is never a label.
The first five physical output rows receive 0.5*tanh(raw/0.5); other rows and
padding are unchanged at that projection. Later denoising states may change
indirectly through attention. Teacher anchors constrain the same first-five
physical slots at separately recorded native flow inputs.
"""

import copy
import inspect
import json
import os
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .records import digest, file_sha256, to_numpy

HORIZON, MODEL_DIM, PHYSICAL_DIM, HIDDEN_DIM = 10, 32, 7, 1024
EXECUTE_STEPS, DRAWS, SEED = 5, 8, 61
UPDATES, BATCH_SIZE = 1000, 128
LEARNING_RATE, ANCHOR_WEIGHT, L2_WEIGHT, GRAD_CLIP = 1e-4, 1.0, 1e-4, 1.0
RESIDUAL_BOUND = 0.5
CAMERAS = ("observation/image", "observation/wrist_image")
SCHEMA = "recipe-flow-head-1.0"


def training_config():
    return {
        "schema_version": SCHEMA,
        "objective": "constrained_masked_native_flow_matching_with_teacher_anchor",
        "flow": "x_t=t*noise+(1-t)*x0; target=noise-x0",
        "imputation": "new_original_condition_native_physical_suffix; x0_padding_zero",
        "labels": "actual_executed_contiguous_first_K_controller_actions_only",
        "horizon": HORIZON,
        "model_dim": MODEL_DIM,
        "physical_dim": PHYSICAL_DIM,
        "hidden_dim": HIDDEN_DIM,
        "trainable_parameters": (HIDDEN_DIM + 1) * PHYSICAL_DIM,
        "apply_rows": EXECUTE_STEPS,
        "anchor_rows": EXECUTE_STEPS,
        "draws_per_window": DRAWS,
        "time_distribution": {
            "beta_alpha": 1.5,
            "beta_beta": 1.0,
            "scale": 0.999,
            "offset": 0.001,
        },
        "residual": "0.5*tanh(affine_final_expert_features/0.5)",
        "residual_bound": RESIDUAL_BOUND,
        "optimizer": "Adam",
        "optimizer_steps": UPDATES,
        "batch_feature_rows": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "anchor_weight": ANCHOR_WEIGHT,
        "l2_weight": L2_WEIGHT,
        "l2_definition": "mean_squared_all_residual_weights_and_biases",
        "gradient_norm_clip": GRAD_CLIP,
        "seed": SEED,
        "base_weights_trainable": False,
        "padding_supervised": False,
        "unexecuted_teacher_suffix_supervised": False,
    }


def _hex(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _array(value, shape, dtype, name):
    value = np.asarray(to_numpy(value))
    if (
        value.shape != shape
        or value.dtype != np.dtype(dtype)
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be finite {np.dtype(dtype)} {shape}")
    return value.copy()


def _observation(observation, prompt):
    _text(prompt, "original_prompt")
    if set(observation) - {*CAMERAS, "observation/state", "prompt"}:
        raise ValueError("Only original paired RGB and state may condition learning")
    if "prompt" in observation and observation["prompt"] != prompt:
        raise ValueError("Observation prompt differs from original task")
    return {
        **{
            key: _array(observation[key], (224, 224, 3), np.uint8, key)
            for key in CAMERAS
        },
        "observation/state": _array(
            observation["observation/state"], (8,), np.float32, "state"
        ),
    }


def _condition_id(observation, prompt):
    return digest({**observation, "prompt": prompt})


def base_identity(policy):
    """Bind the unmodified checkpoint, preprocessing and native implementation."""
    projection = policy.model.action_out_proj
    if (
        policy.horizon != HORIZON
        or policy.action_dim != MODEL_DIM
        or policy.metadata.get("input_profile") != "openpi_libero"
        or not isinstance(projection, nn.Linear)
        or projection.in_features != HIDDEN_DIM
        or projection.out_features != MODEL_DIM
        or projection.weight.dtype != torch.float32
    ):
        raise ValueError(
            "Only the pinned float32 OpenPI-input PI05 10x32/1024 head is supported"
        )
    if policy.policy.training or any(
        p.requires_grad for p in policy.policy.parameters()
    ):
        raise ValueError("Native policy must remain eval/frozen")
    if getattr(policy.config, "compile_model", False) or getattr(
        policy.config, "gradient_checkpointing", False
    ):
        raise ValueError("Compiled/checkpointed native hooks are unsupported")
    keys = (
        "artifact_sha256",
        "model_source_sha256",
        "transformers_source_sha256",
        "torch_version",
        "transformers_version",
        "adapter_source_sha256",
        "processor_source_sha256",
        "input_profile",
        "input_profile_source_sha256",
        "input_profile_assets",
        "normalization",
        "tokenizer_sha256",
    )
    identity = {key: copy.deepcopy(policy.metadata[key]) for key in keys}
    return {
        "native": identity,
        "horizon": HORIZON,
        "model_dim": MODEL_DIM,
        "hidden_dim": HIDDEN_DIM,
    }


def _versions(policy):
    return tuple(
        (name, id(p), p._version, p.requires_grad)
        for name, p in policy.policy.named_parameters()
    )


def _cuda(device, test_only_allow_cpu):
    device = torch.device(device)
    if device.type != "cuda" and not test_only_allow_cpu:
        raise ValueError(
            "Actual feature extraction/training requires CUDA; CPU override is synthetic-test-only"
        )
    if device.type == "cuda" and (
        torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
    ):
        raise ValueError("Recipe extraction/training requires TF32 disabled")
    return device


def _rng(key, stream):
    _text(key, "key")
    checksum = digest({"key": key, "stream": stream, "seed": SEED})
    return np.random.default_rng(
        np.random.SeedSequence(
            [SEED, *[int(checksum[i : i + 8], 16) for i in range(0, 64, 8)]]
        )
    )


def fixed_draws(key):
    """Pinned distribution, materialized fixed draws; no global RNG is consumed."""
    rng = _rng(key, "native_flow_training_draws")
    noise = rng.standard_normal((DRAWS, 1, HORIZON, MODEL_DIM)).astype(np.float32)
    times = (rng.beta(1.5, 1.0, size=DRAWS) * 0.999 + 0.001).astype(np.float32)
    return {"noise": noise, "times": times}


def native_imputation(
    policy, observation, original_prompt, key, *, test_only_allow_cpu=False
):
    """Run NEW keyed native Euler10 under raw images/original task, never a teacher suffix."""
    _cuda(policy.device, test_only_allow_cpu)
    identity, versions = base_identity(policy), _versions(policy)
    raw = _observation(observation, original_prompt)
    condition = policy.prepare(raw, "recipe-imputation:" + digest(key), original_prompt)
    if condition.condition_id != _condition_id(raw, original_prompt):
        raise ValueError("Imputation did not use the original native condition")
    noise = (
        _rng(key, "new_native_suffix_imputation")
        .standard_normal((1, HORIZON, MODEL_DIM))
        .astype(np.float32)
    )
    result = policy.sample(
        condition, policy.tensor(noise), steps=10, solver="euler", time_power=1.0
    )
    values = _array(
        result.value, (1, HORIZON, MODEL_DIM), np.float32, "native imputation"
    )
    if result.velocity_evaluations != 10 or _versions(policy) != versions:
        raise ValueError("Native imputation count or frozen parameter versions changed")
    provenance = {
        "schema_version": SCHEMA,
        "kind": "new_original_condition_native_imputation",
        "base_identity": identity,
        "condition_id": condition.condition_id,
        "observation_sha256": digest(raw),
        "original_prompt": original_prompt,
        "original_prompt_sha256": digest(original_prompt),
        "key_sha256": digest(key),
        "actions_sha256": digest(values),
        "noise_sha256": digest(noise),
        "velocity_evaluations": 10,
        "prefix_preparations": 1,
        "base_parameter_versions_unchanged": True,
        "test_only": bool(test_only_allow_cpu),
    }
    return {"actions": values, "noise": noise, "provenance": provenance}


def _imputation(policy, raw, prompt, native_chunk):
    identity = base_identity(policy)
    values = _array(
        native_chunk["actions"], (1, HORIZON, MODEL_DIM), np.float32, "native chunk"
    )
    noise = _array(native_chunk["noise"], values.shape, np.float32, "imputation noise")
    proof = native_chunk["provenance"]
    if (
        proof["kind"] != "new_original_condition_native_imputation"
        or proof["base_identity"] != identity
        or proof["condition_id"] != _condition_id(raw, prompt)
        or proof["observation_sha256"] != digest(raw)
        or proof["original_prompt"] != prompt
        or proof["original_prompt_sha256"] != digest(prompt)
        or proof["actions_sha256"] != digest(values)
        or proof["noise_sha256"] != digest(noise)
        or proof["velocity_evaluations"] != 10
    ):
        raise ValueError(
            "Native imputation task, observation, base or array binding differs"
        )
    return values, copy.deepcopy(proof)


def make_executed_window(
    policy,
    action_adapter,
    observation,
    *,
    original_prompt,
    executed_actions,
    native_chunk,
    source_id,
    source_receipt_sha256,
    split="adaptation",
):
    """Normalize actual executed values; retain no unexecuted teacher labels."""
    if split != "adaptation":
        raise ValueError("Only declared adaptation data may supply training labels")
    _text(source_id, "source_id")
    _hex(source_receipt_sha256, "source_receipt_sha256")
    raw = _observation(observation, original_prompt)
    native, imputation = _imputation(policy, raw, original_prompt, native_chunk)
    actual = np.asarray(executed_actions)
    if (
        actual.ndim != 2
        or actual.shape[1] != PHYSICAL_DIM
        or not 1 <= len(actual) <= EXECUTE_STEPS
    ):
        raise ValueError("Executed actions must be a contiguous first K<=5 by7 window")
    if actual.dtype.kind not in "fiu" or not np.isfinite(actual).all():
        raise ValueError("Executed actions must be finite numeric values")
    if (
        action_adapter.spec.horizon != HORIZON
        or action_adapter.spec.model_action_dim != MODEL_DIM
    ):
        raise ValueError("ActionAdapter horizon/model dimension differs")
    padded = np.zeros((HORIZON, PHYSICAL_DIM), np.float32)
    padded[: len(actual)] = actual
    # Validate original precision before the standard adapter's float32 conversion.
    if np.any(actual < action_adapter.spec.lower) or np.any(
        actual > action_adapter.spec.upper
    ):
        raise ValueError(
            "Executed labels violate controller bounds; no silent clipping"
        )
    encoded = action_adapter.encode(padded, {**raw, "prompt": original_prompt})
    encoded = _array(encoded, (1, HORIZON, MODEL_DIM), np.float32, "encoded labels")
    if np.any(encoded[..., PHYSICAL_DIM:] != 0):
        raise ValueError("Native action input padding must be zero")
    x0 = np.zeros_like(native)
    x0[..., :PHYSICAL_DIM] = native[..., :PHYSICAL_DIM]
    x0[:, : len(actual), :PHYSICAL_DIM] = encoded[:, : len(actual), :PHYSICAL_DIM]
    mask = np.zeros_like(x0, dtype=bool)
    mask[:, : len(actual), :PHYSICAL_DIM] = True
    provenance = {
        "schema_version": SCHEMA,
        "kind": "executed",
        "split": split,
        "source_id": source_id,
        "source_receipt_sha256": source_receipt_sha256,
        "original_prompt": original_prompt,
        "original_prompt_sha256": digest(original_prompt),
        "observation_sha256": digest(raw),
        "condition_id": _condition_id(raw, original_prompt),
        "base_identity": base_identity(policy),
        "imputation": imputation,
        "executed_count": len(actual),
        "executed_actions_sha256": digest(padded[: len(actual)]),
        "action_spec_sha256": digest(action_adapter.spec.as_dict()),
        "x0_sha256": digest(x0),
        "label_mask_sha256": digest(mask),
        "unexecuted_teacher_suffix_used": False,
        "x0_padding_zero": True,
    }
    return {"observation": raw, "x0": x0, "label_mask": mask, "provenance": provenance}


def _anchor_window(
    policy, observation, original_prompt, native_chunk, source_id, source_receipt_sha256
):
    _text(source_id, "source_id")
    _hex(source_receipt_sha256, "source_receipt_sha256")
    raw = _observation(observation, original_prompt)
    native, imputation = _imputation(policy, raw, original_prompt, native_chunk)
    x0 = np.zeros_like(native)
    x0[..., :PHYSICAL_DIM] = native[..., :PHYSICAL_DIM]
    mask = np.zeros_like(x0, dtype=bool)
    mask[:, :EXECUTE_STEPS, :PHYSICAL_DIM] = True
    return {
        "observation": raw,
        "x0": x0,
        "label_mask": mask,
        "provenance": {
            "schema_version": SCHEMA,
            "kind": "anchor",
            "split": "retention_train",
            "source_id": source_id,
            "source_receipt_sha256": source_receipt_sha256,
            "original_prompt": original_prompt,
            "original_prompt_sha256": digest(original_prompt),
            "observation_sha256": digest(raw),
            "condition_id": _condition_id(raw, original_prompt),
            "base_identity": base_identity(policy),
            "imputation": imputation,
            "executed_count": 0,
            "anchor_count": EXECUTE_STEPS,
            "x0_sha256": digest(x0),
            "label_mask_sha256": digest(mask),
            "unexecuted_teacher_suffix_used": False,
            "x0_padding_zero": True,
        },
    }


@contextmanager
def _projection_hook(projection, callback, *, pre=False):
    # Native execution is synchronous per worker. Reject concurrent/foreign
    # hooks instead of silently capturing an already adapted projection.
    if projection._forward_hooks or projection._forward_pre_hooks:
        raise ValueError("Action projection already has an active hook")
    handle = (
        projection.register_forward_pre_hook(callback)
        if pre
        else projection.register_forward_hook(callback)
    )
    try:
        yield
    finally:
        handle.remove()


class FeatureBank:
    """Detached recorded projection features and exact masked regression targets."""

    def __init__(self, arrays, provenance):
        self.arrays = {
            name: np.array(value, copy=True) for name, value in arrays.items()
        }
        for value in self.arrays.values():
            value.flags.writeable = False
        self.provenance = copy.deepcopy(provenance)
        self.bank_id = digest(self.metadata(include_id=False))
        self.verify()

    @property
    def h(self):
        return self.arrays["h"]

    @property
    def residual_target(self):
        return self.arrays["residual_target"]

    @property
    def counts(self):
        return copy.deepcopy(self.provenance["counts"])

    def metadata(self, *, include_id=True):
        result = {
            "schema_version": SCHEMA,
            "provenance": copy.deepcopy(self.provenance),
            "arrays": {
                name: {
                    "shape": list(a.shape),
                    "dtype": str(a.dtype),
                    "sha256": digest(a),
                }
                for name, a in self.arrays.items()
            },
        }
        if include_id:
            result["bank_id"] = self.bank_id
        return result

    def verify(self):
        if digest(self.metadata(include_id=False)) != self.bank_id:
            raise ValueError("Feature bank array/provenance binding changed")
        p, a = self.provenance, self.arrays
        if (
            p.get("schema_version") != SCHEMA
            or p.get("config") != training_config()
            or p.get("source_sha256") != file_sha256(__file__)
        ):
            raise ValueError("Feature bank capture source/config differs")
        if set(a) != {
            "h",
            "native_velocity",
            "target_velocity",
            "residual_target",
            "noise",
            "times",
            "x0",
            "full_native_velocity",
            "action_row",
            "draw_index",
        }:
            raise ValueError("Feature bank array inventory differs")
        n = len(a["h"])
        _array(a["h"], (n, HIDDEN_DIM), np.float32, "features")
        for name in ("native_velocity", "target_velocity", "residual_target"):
            _array(a[name], (n, PHYSICAL_DIM), np.float32, name)
        _array(a["noise"], (DRAWS, 1, HORIZON, MODEL_DIM), np.float32, "bank noise")
        _array(a["times"], (DRAWS,), np.float32, "bank times")
        if np.any(a["times"] < 0.001) or np.any(a["times"] > 1):
            raise ValueError("Feature bank times exceed native support [.001,1]")
        _array(a["x0"], (1, HORIZON, MODEL_DIM), np.float32, "bank x0")
        _array(
            a["full_native_velocity"],
            (DRAWS, 1, HORIZON, MODEL_DIM),
            np.float32,
            "native velocity",
        )
        if np.any(a["x0"][..., PHYSICAL_DIM:] != 0):
            raise ValueError("Feature bank action padding must remain zero")
        count = (
            p["window"]["executed_count"] if p["kind"] == "executed" else EXECUTE_STEPS
        )
        if (
            p["kind"] not in ("executed", "anchor")
            or not 1 <= count <= EXECUTE_STEPS
            or n != DRAWS * count
        ):
            raise ValueError("Feature bank known action horizon differs")
        for name, expected in (
            ("action_row", np.tile(np.arange(count), DRAWS)),
            ("draw_index", np.repeat(np.arange(DRAWS), count)),
        ):
            if not np.array_equal(a[name], expected) or a[name].dtype != np.int64:
                raise ValueError("Feature bank row mask includes unexecuted positions")
        native = a["full_native_velocity"][
            a["draw_index"], 0, a["action_row"], :PHYSICAL_DIM
        ]
        target = (
            native
            if p["kind"] == "anchor"
            else a["noise"][a["draw_index"], 0, a["action_row"], :PHYSICAL_DIM]
            - a["x0"][0, a["action_row"], :PHYSICAL_DIM]
        )
        if not (
            np.array_equal(native, a["native_velocity"])
            and np.array_equal(target, a["target_velocity"])
            and np.array_equal(target - native, a["residual_target"])
        ):
            raise ValueError(
                "Native target/residual labels differ from recorded flow inputs"
            )

    def save(self, directory):
        self.verify()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory / "arrays.npz", **self.arrays)
        manifest = {
            **self.metadata(),
            "arrays_file_sha256": file_sha256(directory / "arrays.npz"),
        }
        _write_json(directory / "manifest.json", manifest)
        return manifest

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if file_sha256(directory / "arrays.npz") != manifest["arrays_file_sha256"]:
            raise ValueError("Feature bank file hash differs")
        with np.load(directory / "arrays.npz", allow_pickle=False) as values:
            result = cls(
                {name: values[name] for name in values.files}, manifest["provenance"]
            )
        if result.metadata() != {
            k: v for k, v in manifest.items() if k != "arrays_file_sha256"
        }:
            raise ValueError("Feature bank manifest differs")
        return result


def capture_flow_features(policy, window, *, noise, times, test_only_allow_cpu=False):
    """One native prefix preparation and eight native denoise forwards, no fitting."""
    _cuda(policy.device, test_only_allow_cpu)
    identity, versions = base_identity(policy), _versions(policy)
    p = copy.deepcopy(window["provenance"])
    raw = _observation(window["observation"], p["original_prompt"])
    x0 = _array(window["x0"], (1, HORIZON, MODEL_DIM), np.float32, "x0")
    mask = _array(window["label_mask"], x0.shape, bool, "label mask")
    count = p["executed_count"] if p["kind"] == "executed" else EXECUTE_STEPS
    expected_mask = np.zeros_like(mask)
    expected_mask[:, :count, :PHYSICAL_DIM] = True
    if (
        p["base_identity"] != identity
        or p["observation_sha256"] != digest(raw)
        or p["condition_id"] != _condition_id(raw, p["original_prompt"])
        or p["original_prompt_sha256"] != digest(p["original_prompt"])
        or p["x0_sha256"] != digest(x0)
        or p["label_mask_sha256"] != digest(mask)
        or not 1 <= count <= EXECUTE_STEPS
        or not np.array_equal(mask, expected_mask)
        or np.any(x0[..., PHYSICAL_DIM:] != 0)
    ):
        raise ValueError(
            "Window task, observation, input or executed mask binding differs"
        )
    noise = _array(noise, (DRAWS, 1, HORIZON, MODEL_DIM), np.float32, "noise")
    times = _array(times, (DRAWS,), np.float32, "times")
    if np.any(times < 0.001) or np.any(times > 1):
        raise ValueError("Times must follow the declared native support [.001,1]")
    condition = policy.prepare(
        raw, "recipe-features:" + digest(p["source_id"]), p["original_prompt"]
    )
    if condition.condition_id != p["condition_id"]:
        raise ValueError("Feature condition is not original native conditioning")
    features, velocities, inputs = [], [], []
    began = time.perf_counter()
    for index, timestep in enumerate(times):
        captured = []

        def hook(_module, args):
            captured.append(
                _array(
                    args[0].detach(),
                    (1, HORIZON, HIDDEN_DIM),
                    np.float32,
                    "native final expert features",
                )
            )

        with torch.no_grad():
            t = torch.tensor(timestep, dtype=torch.float32, device=policy.device)
            x = t * policy.tensor(noise[index]) + (1 - t) * policy.tensor(x0)
            with _projection_hook(policy.model.action_out_proj, hook, pre=True):
                velocity = condition.velocity(x, float(timestep))
        if len(captured) != 1:
            raise ValueError(
                "Expected one native final projection per denoise evaluation"
            )
        features.append(captured[0][0, :count])
        velocities.append(_array(velocity, x0.shape, np.float32, "native velocity"))
        inputs.append(digest(to_numpy(x)))
    if _versions(policy) != versions:
        raise ValueError("Base parameters changed during feature extraction")
    full_native = np.stack(velocities)
    native = full_native[:, 0, :count, :PHYSICAL_DIM].reshape(-1, PHYSICAL_DIM)
    target = (
        native.copy()
        if p["kind"] == "anchor"
        else (noise - x0[None])[:, 0, :count, :PHYSICAL_DIM].reshape(-1, PHYSICAL_DIM)
    )
    arrays = {
        "h": np.concatenate(features),
        "native_velocity": native,
        "target_velocity": target,
        "residual_target": target - native,
        "noise": noise,
        "times": times,
        "x0": x0,
        "full_native_velocity": full_native,
        "action_row": np.tile(np.arange(count, dtype=np.int64), DRAWS),
        "draw_index": np.repeat(np.arange(DRAWS, dtype=np.int64), count),
    }
    provenance = {
        "schema_version": SCHEMA,
        "config": training_config(),
        "kind": p["kind"],
        "window": p,
        "base_identity": identity,
        "source_sha256": file_sha256(__file__),
        "native_projection_source_sha256": digest(
            inspect.getsource(type(policy.model.action_out_proj))
        ),
        "native_flow_input_sha256": inputs,
        "feature_latency_seconds": time.perf_counter() - began,
        "counts": {
            "velocity_evaluations": DRAWS,
            "prefix_preparations": 1,
            "projection_calls": DRAWS,
            "feature_rows": len(native),
        },
        "base_parameter_versions_unchanged": True,
        "test_only": bool(test_only_allow_cpu),
    }
    return FeatureBank(arrays, provenance)


def capture_anchor_features(
    policy,
    observation,
    original_prompt,
    *,
    native_chunk,
    source_id,
    source_receipt_sha256,
    noise,
    times,
    test_only_allow_cpu=False,
):
    window = _anchor_window(
        policy,
        observation,
        original_prompt,
        native_chunk,
        source_id,
        source_receipt_sha256,
    )
    return capture_flow_features(
        policy,
        window,
        noise=noise,
        times=times,
        test_only_allow_cpu=test_only_allow_cpu,
    )


def capture_samples(
    policy,
    action_adapter,
    observation,
    original_prompt,
    executed_actions,
    key,
    *,
    sample_id,
    source_receipt_sha256,
    anchor=False,
    test_only_allow_cpu=False,
):
    """One corpus window -> detached bank; exactly18VF/two prefix preparations.

    Native-anchor recorded actions are provenance only. Their regression target
    is the frozen native velocity on a newly imputed native flow input.
    """
    if type(anchor) is not bool:
        raise ValueError("anchor must be an explicit boolean")
    imputed = native_imputation(
        policy,
        observation,
        original_prompt,
        key,
        test_only_allow_cpu=test_only_allow_cpu,
    )
    draws = fixed_draws(key)
    if anchor:
        bank = capture_anchor_features(
            policy,
            observation,
            original_prompt,
            native_chunk=imputed,
            source_id=sample_id,
            source_receipt_sha256=source_receipt_sha256,
            **draws,
            test_only_allow_cpu=test_only_allow_cpu,
        )
    else:
        window = make_executed_window(
            policy,
            action_adapter,
            observation,
            original_prompt=original_prompt,
            executed_actions=executed_actions,
            native_chunk=imputed,
            source_id=sample_id,
            source_receipt_sha256=source_receipt_sha256,
        )
        bank = capture_flow_features(
            policy, window, **draws, test_only_allow_cpu=test_only_allow_cpu
        )
    provenance = copy.deepcopy(bank.provenance)
    provenance["counts"] = {
        **bank.counts,
        "velocity_evaluations": 10 + DRAWS,
        "imputation_velocity_evaluations": 10,
        "feature_velocity_evaluations": DRAWS,
        "prefix_preparations": 2,
    }
    provenance["anchor_recorded_actions_used_as_labels"] = False
    return FeatureBank(bank.arrays, provenance)


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


class ResidualActionHead(nn.Module):
    def __init__(self, identity, *, device="cpu", test_only=False):
        super().__init__()
        self.base_identity = copy.deepcopy(identity)
        with torch.random.fork_rng(devices=[]):
            self.linear = nn.Linear(HIDDEN_DIM, PHYSICAL_DIM, bias=True)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
        self.to(device=device, dtype=torch.float32)
        self.test_only, self.optimizer_steps = bool(test_only), 0
        self.history, self._optimizer_state = [], None

    def forward(self, features):
        return RESIDUAL_BOUND * torch.tanh(self.linear(features) / RESIDUAL_BOUND)

    @property
    def device(self):
        return self.linear.weight.device

    def is_zero(self):
        return not any(bool(torch.count_nonzero(p).item()) for p in self.parameters())

    def parameter_sha256(self):
        return digest(
            {name: to_numpy(value) for name, value in self.state_dict().items()}
        )

    def metadata(self):
        return {
            "schema_version": SCHEMA,
            "base_identity": copy.deepcopy(self.base_identity),
            "config": training_config(),
            "source_sha256": file_sha256(__file__),
            "parameter_sha256": self.parameter_sha256(),
            "optimizer_steps": self.optimizer_steps,
            "test_only": self.test_only,
            "zero_effect": self.is_zero(),
            "history": copy.deepcopy(self.history),
        }

    def snapshot(self):
        return copy.deepcopy(
            {
                "identity": self.base_identity,
                "state": self.state_dict(),
                "optimizer": self._optimizer_state,
                "steps": self.optimizer_steps,
                "history": self.history,
                "test_only": self.test_only,
            }
        )

    def restore(self, snapshot):
        if (
            snapshot["identity"] != self.base_identity
            or snapshot["test_only"] != self.test_only
        ):
            raise ValueError("Rollback base identity or test scope differs")
        self.load_state_dict(snapshot["state"], strict=True)
        self.zero_grad(set_to_none=True)
        self._optimizer_state = copy.deepcopy(snapshot["optimizer"])
        self.optimizer_steps, self.history = (
            snapshot["steps"],
            copy.deepcopy(snapshot["history"]),
        )

    def fit(
        self,
        executed_banks,
        anchor_banks,
        *,
        admission_receipt_sha256,
        _test_steps=None,
    ):
        """Fit detached row features only; no base-policy object enters optimization."""
        _hex(admission_receipt_sha256, "admission_receipt_sha256")
        _cuda(self.device, self.test_only)
        if _test_steps is not None and not self.test_only:
            raise ValueError("Production update budget cannot be overridden")
        steps = UPDATES if _test_steps is None else _test_steps
        if type(steps) is not int or not 1 <= steps <= UPDATES:
            raise ValueError("Invalid optimizer step count")
        if self.device.type == "cuda" and os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ) not in (":4096:8", ":16:8"):
            raise ValueError(
                "CUDA deterministic training requires CUBLAS_WORKSPACE_CONFIG"
            )
        executed_banks, anchor_banks = list(executed_banks), list(anchor_banks)
        if not executed_banks or not anchor_banks:
            raise ValueError(
                "Executed labels and independent native anchors are both required"
            )
        ids = []
        for kind, banks in (("executed", executed_banks), ("anchor", anchor_banks)):
            for bank in banks:
                bank.verify()
                if (
                    bank.provenance["kind"] != kind
                    or bank.provenance["base_identity"] != self.base_identity
                    or bank.provenance["test_only"] != self.test_only
                ):
                    raise ValueError(
                        "Feature bank role, base or test-only identity differs"
                    )
                ids.append(bank.bank_id)
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate feature bank would silently reweight examples")
        x = torch.tensor(
            np.concatenate([b.h for b in executed_banks]), device=self.device
        )
        target = torch.tensor(
            np.concatenate([b.residual_target for b in executed_banks]),
            device=self.device,
        )
        anchors = torch.tensor(
            np.concatenate([b.h for b in anchor_banks]), device=self.device
        )
        before = self.snapshot()
        before_hash = self.parameter_sha256()
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        if self._optimizer_state is not None:
            optimizer.load_state_dict(self._optimizer_state)
        rng = _rng(str(len(self.history)), "head_minibatch_rows")
        deterministic, warn_only = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )

        def diagnostics():
            with torch.no_grad():
                prediction, raw = self(x), self.linear(x)
                return {
                    "masked_residual_mse": float((prediction - target).square().mean()),
                    "mean_absolute_target_error": float(
                        (prediction - target).abs().mean()
                    ),
                    "anchor_mse": float(self(anchors).square().mean()),
                    "target_outside_bound_fraction": float(
                        (target.abs() > RESIDUAL_BOUND).float().mean()
                    ),
                    "raw_residual_outside_bound_fraction": float(
                        (raw.abs() > RESIDUAL_BOUND).float().mean()
                    ),
                    "soft_saturation_fraction": float(
                        (prediction.abs() >= 0.99 * RESIDUAL_BOUND).float().mean()
                    ),
                    "hard_clipping_used": False,
                }

        started = time.perf_counter()
        try:
            torch.use_deterministic_algorithms(True)
            initial = diagnostics()
            for _ in range(steps):
                ix = torch.tensor(
                    rng.integers(len(x), size=BATCH_SIZE), device=self.device
                )
                ia = torch.tensor(
                    rng.integers(len(anchors), size=BATCH_SIZE), device=self.device
                )
                optimizer.zero_grad(set_to_none=True)
                supervised = (self(x[ix]) - target[ix]).square().mean()
                anchor = self(anchors[ia]).square().mean()
                l2 = (
                    torch.cat([p.reshape(-1) for p in self.parameters()])
                    .square()
                    .mean()
                )
                loss = supervised + ANCHOR_WEIGHT * anchor + L2_WEIGHT * l2
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite head training objective")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    self.parameters(), GRAD_CLIP, error_if_nonfinite=True
                )
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("Nonfinite residual-head gradient")
                optimizer.step()
            final = diagnostics()
            if not all(
                np.isfinite(v) for k, v in final.items() if k != "hard_clipping_used"
            ):
                raise FloatingPointError("Nonfinite fitted-head diagnostics")
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.optimizer_steps += steps
            self._optimizer_state = copy.deepcopy(optimizer.state_dict())
            receipt = {
                "status": "fitted",
                "objective": training_config()["objective"],
                "admission_receipt_sha256": admission_receipt_sha256,
                "executed_bank_ids": [b.bank_id for b in executed_banks],
                "anchor_bank_ids": [b.bank_id for b in anchor_banks],
                "feature_rows": len(x),
                "anchor_rows": len(anchors),
                "optimizer_steps": steps,
                "total_optimizer_steps": self.optimizer_steps,
                "trainable_parameters": sum(p.numel() for p in self.parameters()),
                "before_parameter_sha256": before_hash,
                "after_parameter_sha256": self.parameter_sha256(),
                "before": initial,
                "after": final,
                "wall_seconds": time.perf_counter() - started,
                "device": str(self.device),
                "test_only": self.test_only,
                "velocity_evaluations": 0,
                "base_backward_passes": 0,
            }
            self.history.append(copy.deepcopy(receipt))
            return receipt
        except Exception:
            self.restore(before)
            raise
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        torch.save(self.snapshot(), directory / "state.pt")
        manifest = {
            **self.metadata(),
            "state_file_sha256": file_sha256(directory / "state.pt"),
        }
        _write_json(directory / "manifest.json", manifest)
        return manifest

    @classmethod
    def load(
        cls, directory, *, expected_base_identity, device="cpu", allow_test_only=False
    ):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest["base_identity"] != expected_base_identity
            or manifest["config"] != training_config()
            or manifest["source_sha256"] != file_sha256(__file__)
            or manifest["state_file_sha256"] != file_sha256(directory / "state.pt")
            or manifest["test_only"]
            and not allow_test_only
        ):
            raise ValueError(
                "Head checkpoint identity, file hash or test scope differs"
            )
        result = cls(
            expected_base_identity, device=device, test_only=manifest["test_only"]
        )
        result.restore(
            torch.load(directory / "state.pt", map_location=device, weights_only=True)
        )
        if result.metadata() != {
            k: v for k, v in manifest.items() if k != "state_file_sha256"
        }:
            raise ValueError("Head checkpoint metadata differs from restored state")
        return result


def prepare_adapted(policy, observation, observation_id, original_prompt, head):
    """Return (native-prefix Condition, mutable application provenance)."""
    raw = _observation(observation, original_prompt)
    if (
        base_identity(policy) != head.base_identity
        or torch.device(policy.device) != head.device
    ):
        raise ValueError("Residual head base/device differs from native policy")
    condition = policy.prepare(raw, observation_id, original_prompt)
    if condition.condition_id != _condition_id(raw, original_prompt):
        raise ValueError("Adaptation must use raw images and original task")
    zero = head.is_zero()
    provenance = {
        "schema_version": SCHEMA,
        "operator": "bounded_final_action_expert_residual",
        "head": head.metadata(),
        "original_condition_id": condition.condition_id,
        "observation_sha256": digest(raw),
        "original_prompt_sha256": digest(original_prompt),
        "native_prefix_unchanged": True,
        "enabled": not zero,
        "direct_rows_5_to_9_unchanged": True,
        "direct_padding_unchanged": True,
        "projection_calls": 0,
        "residual_values": 0,
        "raw_outside_bound_values": 0,
        "soft_saturated_values": 0,
        "residual_max_abs": 0.0,
        "hard_clipping_used": False,
    }
    if zero:
        return (
            condition,
            provenance,
        )  # Exact native callable/output path, no +0 rounding.
    native_velocity = condition.velocity
    versions = tuple(p._version for p in head.parameters())

    def velocity(x, t):
        if tuple(p._version for p in head.parameters()) != versions:
            raise ValueError(
                "Prepared condition refers to an updated head; prepare it again"
            )
        calls = []

        def hook(_module, args, output):
            features = args[0][:, :EXECUTE_STEPS]
            if features.shape != (1, EXECUTE_STEPS, HIDDEN_DIM) or output.shape != (
                1,
                HORIZON,
                MODEL_DIM,
            ):
                raise ValueError("Native projection layout changed")
            with torch.no_grad():
                raw_delta = head.linear(features)
                delta = RESIDUAL_BOUND * torch.tanh(raw_delta / RESIDUAL_BOUND)
                if not bool(torch.isfinite(delta).all()):
                    raise FloatingPointError("Nonfinite learned residual")
                changed = output.clone()
                changed[:, :EXECUTE_STEPS, :PHYSICAL_DIM] = (
                    output[:, :EXECUTE_STEPS, :PHYSICAL_DIM] + delta
                )
                if not torch.equal(
                    changed[:, EXECUTE_STEPS:], output[:, EXECUTE_STEPS:]
                ) or not torch.equal(
                    changed[..., PHYSICAL_DIM:], output[..., PHYSICAL_DIM:]
                ):
                    raise ValueError("Residual wrote outside executed physical slots")
                calls.append(True)
                provenance["projection_calls"] += 1
                provenance["residual_values"] += delta.numel()
                provenance["raw_outside_bound_values"] += int(
                    (raw_delta.abs() > RESIDUAL_BOUND).sum()
                )
                provenance["soft_saturated_values"] += int(
                    (delta.abs() >= 0.99 * RESIDUAL_BOUND).sum()
                )
                provenance["residual_max_abs"] = max(
                    provenance["residual_max_abs"], float(delta.abs().max())
                )
            return changed

        with _projection_hook(policy.model.action_out_proj, hook):
            result = native_velocity(x, t)
        if len(calls) != 1:
            raise ValueError("Expected exactly one adapted native projection")
        return result

    condition_id = digest(
        {
            "native_condition_id": condition.condition_id,
            "head": head.parameter_sha256(),
            "config": training_config(),
        }
    )
    return replace(condition, condition_id=condition_id, velocity=velocity), provenance
