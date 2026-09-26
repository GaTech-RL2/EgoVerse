"""Auditable action references and padding handling for flow reversal steering.

The paper uses noise time 0 and action time 1. This repository uses action time
0 and noise time 1: callers must use ``invert(..., solver='euler', steps=10)``
then ``sample(..., solver='euler', steps=10)``, with ``time_power=1.0``. Finite
Euler reversal is not an exact round trip. These helpers do not run the model.

``directional_reference`` implements the normalized-coordinate reference in
arXiv:2606.13675v2 D.2/D.3. ``edit_native`` is a separate controller-input edit.
Both expose source and target arrays for recording. The optional repeated mean
in ``resample_padding_noise`` is for the auxiliary-policy learning experiment;
the paper-like online operator leaves recovered physical-channel noise intact.
"""

import copy
from pathlib import Path

import numpy as np

from .records import digest, file_sha256, to_numpy

HORIZON = 10
ACTION_DIM = 32
PHYSICAL_DIM = 7


def _spec(adapter):
    if (adapter.spec.horizon, adapter.spec.model_action_dim) != (
        HORIZON,
        ACTION_DIM,
    ):
        raise ValueError("FRS requires the pinned [10, 32] action representation")
    return adapter.spec


def _vector(value, *, name, lower, upper):
    if isinstance(value, (list, tuple)) and any(
        isinstance(x, (bool, np.bool_)) for x in value
    ):
        raise ValueError(f"{name} must contain three finite numeric values")
    raw = np.asarray(value)
    if (
        raw.shape != (3,)
        or raw.dtype.kind not in "fiu"
        or not np.isfinite(raw).all()
        or np.any(raw < lower)
        or np.any(raw > upper)
    ):
        raise ValueError(f"{name} must be a finite 3-vector in [{lower}, {upper}]")
    return raw.astype(np.float64)


def _encode(adapter, observation, actions):
    encoded = adapter.encode(actions, observation)
    if encoded.dtype != np.float32 or encoded.shape != (1, HORIZON, ACTION_DIM):
        raise ValueError("ActionAdapter must encode float32 [1, 10, 32]")
    # pi05 only consumes seven physical action channels. Padding must be zero
    # BEFORE reversal even if a transform accidentally produces nonzero padding.
    corrected = encoded.copy()
    corrected[..., PHYSICAL_DIM:] = 0
    return encoded, corrected


def _receipt(adapter, observation, source, target, **details):
    return {
        "schema_version": "frs-reference-1.0",
        "operator_source_sha256": file_sha256(Path(__file__)),
        "action_spec_id": adapter.spec.action_spec_id,
        "action_spec_sha256": digest(adapter.spec.as_dict()),
        "observation_sha256": digest(observation),
        "source_model_sha256": digest(source),
        "target_model_sha256": digest(target),
        "horizon": HORIZON,
        "execute_steps": HORIZON,
        "internal_shape": [1, HORIZON, ACTION_DIM],
        "inverse_padding_zero": bool(np.all(target[..., PHYSICAL_DIM:] == 0)),
        "solver": "euler",
        "inverse_steps": 10,
        "generation_steps": 10,
        "time_power": 1.0,
        "time_convention": "action0_noise1",
        "known_generating_noise": None,
        **details,
    }


def directional_reference(adapter, observation, *, coords, motion_amount):
    """Build D.2's constant reference without treating normalized XYZ as raw XYZ.

    ``coords`` is in {-1,0,1}^3. Normalize a nonzero vector to unit length, then
    multiply by 0.5 for ``less``. Start from *encoded raw zeros*, replace only
    normalized XYZ, and zero the 25 padding channels. Thus normalized rotation
    and gripper values correctly retain the encoding of unnormalized zero.
    """
    _spec(adapter)
    vector = _vector(coords, name="coords", lower=-1, upper=1)
    if not np.isin(vector, [-1, 0, 1]).all():
        raise ValueError("coords entries must be exactly -1, 0, or 1")
    if motion_amount not in ("less", "more"):
        raise ValueError("motion_amount must be 'less' or 'more'")
    length = float(np.linalg.norm(vector))
    direction = vector / length if length else vector.copy()
    if motion_amount == "less":
        direction *= 0.5
    source_actions = np.zeros((HORIZON, PHYSICAL_DIM), dtype=np.float32)
    source_model, target_model = _encode(adapter, observation, source_actions)
    target_model[..., :3] = direction.astype(np.float32)
    receipt = _receipt(
        adapter,
        observation,
        source_model,
        target_model,
        operator="paper_directional_reference",
        reference_space="normalized_policy_translation",
        coords=vector.astype(int).tolist(),
        motion_amount=motion_amount,
        normalized_translation=direction.astype(np.float32).tolist(),
        raw_rotation_and_gripper=[0.0, 0.0, 0.0, 0.0],
        encoded_rotation_gripper_preserved=bool(
            np.array_equal(source_model[..., 3:7], target_model[..., 3:7])
        ),
        source_actions_sha256=digest(source_actions),
        source_padding_nonzero_count=int(np.count_nonzero(source_model[..., 7:])),
    )
    return {
        "source_actions": source_actions,
        "source_model_actions": source_model,
        "target_model_actions": target_model,
        "receipt": receipt,
    }


def edit_native(
    adapter,
    observation,
    native_actions,
    *,
    delta_xyz,
    apply_steps,
    gripper="keep",
):
    """Edit a decoded native chunk in controller units, with explicit clipping.

    Delta proposals outside [-0.5,0.5] are rejected. Valid proposals are added
    to the first ``apply_steps`` rows, after which controller bounds are applied
    and the exact clipping mask/count/magnitude are returned. Rotation channels
    and the untouched suffix are preserved bitwise. Open/close use the verified
    controller's negative/positive gripper endpoints for those same rows.
    """
    spec = _spec(adapter)
    source_actions = spec.validate_actions(native_actions)
    delta = _vector(delta_xyz, name="delta_xyz", lower=-0.5, upper=0.5)
    if type(apply_steps) is not int or not 1 <= apply_steps <= HORIZON:
        raise ValueError("apply_steps must be an integer in [1, 10]")
    if gripper not in ("keep", "open", "close"):
        raise ValueError("gripper must be 'keep', 'open', or 'close'")
    if not spec.lower[6] < 0 < spec.upper[6]:
        raise ValueError(
            "Expected verified negative-open/positive-close gripper bounds"
        )
    requested = source_actions.copy()
    requested[:apply_steps, :3] += delta.astype(np.float32)
    if gripper != "keep":
        requested[:apply_steps, 6] = (
            spec.lower[6] if gripper == "open" else spec.upper[6]
        )
    target_actions = np.clip(requested, spec.lower, spec.upper).astype(np.float32)
    clipping_mask = requested != target_actions
    clipping_delta = target_actions.astype(np.float64) - requested
    source_model, _ = _encode(adapter, observation, source_actions)
    encoded_target, target_model = _encode(adapter, observation, target_actions)
    receipt = _receipt(
        adapter,
        observation,
        source_model,
        target_model,
        operator="edit_native",
        reference_space="environment_controller_input_before_checkpoint_normalization",
        delta_xyz=delta.tolist(),
        apply_steps=apply_steps,
        gripper=gripper,
        source_actions_sha256=digest(source_actions),
        requested_actions_sha256=digest(requested),
        target_actions_sha256=digest(target_actions),
        rotation_preserved=bool(
            np.array_equal(source_actions[:, 3:6], target_actions[:, 3:6])
        ),
        suffix_preserved=bool(
            np.array_equal(source_actions[apply_steps:], target_actions[apply_steps:])
        ),
        clipping={
            "count": int(np.count_nonzero(clipping_mask)),
            "fraction": float(np.mean(clipping_mask)),
            "max_abs": float(np.max(np.abs(clipping_delta))),
            "mask_sha256": digest(clipping_mask),
            "rule": "explicit_controller_bounds_after_validated_delta",
        },
        target_transform_padding_nonzero_count=int(
            np.count_nonzero(encoded_target[..., 7:])
        ),
    )
    return {
        "source_actions": source_actions,
        "requested_actions": requested,
        "target_actions": target_actions,
        "clipping_mask": clipping_mask,
        "source_model_actions": source_model,
        "target_model_actions": target_model,
        "receipt": receipt,
    }


def resample_padding_noise(recovered, rng, *, repeat_mean=False):
    """Refresh padding after inversion, optionally repeating the physical mean.

    ``repeat_mean=False`` implements the paper's online padding treatment.
    ``True`` projects physical noise onto one seven-vector *before execution*,
    making the executed label exactly representable by the auxiliary policy.
    Source inversion noise is retained; it is never described as known noise.
    """
    source = np.array(to_numpy(recovered), copy=True)
    if (
        source.shape != (1, HORIZON, ACTION_DIM)
        or source.dtype != np.float32
        or not np.isfinite(source).all()
    ):
        raise ValueError("Recovered noise must be finite float32 [1, 10, 32]")
    if type(repeat_mean) is not bool or not isinstance(rng, np.random.Generator):
        raise ValueError("Expected a NumPy Generator and boolean repeat_mean")
    mean = source[0, :, :7].astype(np.float64).mean(axis=0).astype(np.float32)
    noise = source.copy()
    if repeat_mean:
        noise[..., :7] = mean
    before = copy.deepcopy(rng.bit_generator.state)
    noise[..., 7:] = rng.standard_normal(
        (1, HORIZON, ACTION_DIM - PHYSICAL_DIM)
    ).astype(np.float32)
    difference = noise[..., :7].astype(np.float64) - source[..., :7]
    receipt = {
        "schema_version": "frs-noise-1.0",
        "operator_source_sha256": file_sha256(Path(__file__)),
        "source_noise_sha256": digest(source),
        "generation_noise_sha256": digest(noise),
        "mean_first7_sha256": digest(mean),
        "repeat_mean_before_execution": repeat_mean,
        "mean_reduction": "float64_accumulator_then_float32",
        "first7_preserved": bool(np.array_equal(source[..., :7], noise[..., :7])),
        "first7_change_max_abs": float(np.max(np.abs(difference))),
        "first7_change_rmse": float(np.sqrt(np.mean(difference**2))),
        "padding_distribution": "independent_standard_normal",
        "resampled_padding_dimensions": [7, 32],
        "resampled_values": HORIZON * (ACTION_DIM - PHYSICAL_DIM),
        "rng_bit_generator": type(rng.bit_generator).__name__,
        "rng_state_before_sha256": digest(before),
        "rng_state_after_sha256": digest(rng.bit_generator.state),
        "known_generating_noise": None,
    }
    return {
        "source_noise": source,
        "noise": noise,
        "mean_first7": mean,
        "receipt": receipt,
    }


def repeated_gaussian_noise(rng):
    """Matched loop baseline: one unit-variance 7-vector, not ten-vector averaging."""
    if not isinstance(rng, np.random.Generator):
        raise ValueError("Expected a NumPy Generator")
    before = copy.deepcopy(rng.bit_generator.state)
    mean = rng.standard_normal(PHYSICAL_DIM).astype(np.float32)
    noise = np.empty((1, HORIZON, ACTION_DIM), np.float32)
    noise[..., :PHYSICAL_DIM] = mean
    noise[..., PHYSICAL_DIM:] = rng.standard_normal(
        (1, HORIZON, ACTION_DIM - PHYSICAL_DIM)
    ).astype(np.float32)
    return {
        "noise": noise,
        "mean_first7": mean,
        "receipt": {
            "schema_version": "frs-loop-baseline-noise-1.0",
            "operator_source_sha256": file_sha256(Path(__file__)),
            "generation_noise_sha256": digest(noise),
            "physical_noise_distribution": "one_independent_standard_normal_7vector_repeated_10",
            "physical_gaussian_variance": 1.0,
            "averaged_independent_gaussian_rows": False,
            "padding_distribution": "independent_standard_normal",
            "rng_bit_generator": type(rng.bit_generator).__name__,
            "rng_state_before_sha256": digest(before),
            "rng_state_after_sha256": digest(rng.bit_generator.state),
        },
    }
