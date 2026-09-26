"""FRS action-space boundaries and the actual noise used for generation."""

import copy

import numpy as np
import pytest

from astra_reversal.action_adapter import ActionAdapter
from astra_reversal.frs_operators import (
    directional_reference,
    edit_native,
    repeated_gaussian_noise,
    resample_padding_noise,
)
from astra_reversal.records import digest


@pytest.fixture
def adapter(spec):
    offset = np.arange(7, dtype=np.float32) * 0.1 + 0.05
    scale = np.arange(7, dtype=np.float32) + 2

    def encode(data):
        # Deliberately nonzero padding tests the mandatory pre-inverse repair.
        actions = np.pad(
            (data["actions"] - offset) / scale, ((0, 0), (0, 25)), constant_values=9
        )
        return {"actions": actions, "state": data["observation/state"]}

    return ActionAdapter(
        spec, encode, lambda data: {"actions": data["actions"][:, :7] * scale + offset}
    )


def observation():
    return {"observation/state": np.zeros(8, np.float32)}


def test_direction_is_normalized_space_and_retains_encoded_raw_zero_rotations(adapter):
    raw = observation()
    before = copy.deepcopy(raw)
    reference = directional_reference(
        adapter, raw, coords=[1, -1, 1], motion_amount="less"
    )
    target = reference["target_model_actions"]
    np.testing.assert_allclose(
        target[..., :3],
        np.broadcast_to(np.array([1, -1, 1]) / np.sqrt(3) / 2, (1, 10, 3)),
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        target[..., 3:7], reference["source_model_actions"][..., 3:7]
    )
    assert np.all(target[..., 3:7] != 0)  # Raw zeros are not normalized zeros.
    assert np.all(target[..., 7:] == 0)
    assert reference["receipt"]["source_padding_nonzero_count"] == 250
    assert digest(raw) == digest(before)
    assert reference["receipt"]["target_model_sha256"] == digest(target)
    assert reference["receipt"]["time_convention"] == "action0_noise1"


def test_direction_more_and_zero_vector_are_defined(adapter):
    for coords, expected in (([0, 0, 1], [0, 0, 1]), ([0, 0, 0], [0, 0, 0])):
        value = directional_reference(
            adapter, observation(), coords=coords, motion_amount="more"
        )
        np.testing.assert_array_equal(value["target_model_actions"][0, 0, :3], expected)


@pytest.mark.parametrize(
    "coords",
    [[True, 0, 0], [0.1, 0, 0], [2, 0, 0], [0, np.nan, 0], [1, 0], ["1", "0", "0"]],
)
def test_invalid_direction_proposal_is_rejected(adapter, coords):
    with pytest.raises(ValueError):
        directional_reference(
            adapter, observation(), coords=coords, motion_amount="more"
        )


def test_native_edit_exposes_actual_clipping_and_preserves_rotations_and_suffix(
    adapter,
):
    native = np.full((10, 7), 0.1, np.float32)
    native[:3, 0] = 0.9
    original = native.copy()
    result = edit_native(
        adapter,
        observation(),
        native,
        delta_xyz=[0.5, -0.5, 0.25],
        apply_steps=3,
        gripper="open",
    )
    assert np.all(result["requested_actions"][:3, 0] > 1)
    assert np.all(result["target_actions"][:3, 0] == 1)
    assert result["receipt"]["clipping"]["count"] == 3
    assert result["receipt"]["clipping"]["max_abs"] == pytest.approx(0.4)
    assert np.all(result["target_actions"][:3, 6] == -1)
    np.testing.assert_array_equal(result["target_actions"][:, 3:6], original[:, 3:6])
    np.testing.assert_array_equal(result["target_actions"][3:], original[3:])
    np.testing.assert_array_equal(native, original)
    assert np.all(result["target_model_actions"][..., 7:] == 0)
    np.testing.assert_array_equal(
        result["target_model_actions"][..., :7],
        adapter.encode(result["target_actions"], observation())[..., :7],
    )


@pytest.mark.parametrize("gripper,value", [("keep", 0.2), ("open", -1), ("close", 1)])
def test_gripper_is_confined_to_selected_prefix(adapter, gripper, value):
    native = np.full((10, 7), 0.2, np.float32)
    result = edit_native(
        adapter,
        observation(),
        native,
        delta_xyz=[0, 0, 0],
        apply_steps=1,
        gripper=gripper,
    )
    assert result["target_actions"][0, 6] == pytest.approx(value)
    np.testing.assert_array_equal(result["target_actions"][1:], native[1:])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"delta_xyz": [0.51, 0, 0]},
        {"delta_xyz": [False, 0, 0]},
        {"apply_steps": 0},
        {"apply_steps": 11},
        {"apply_steps": True},
        {"gripper": "grab"},
    ],
)
def test_native_invalid_proposals_are_not_silently_bounded(adapter, kwargs):
    arguments = {"delta_xyz": [0, 0, 0], "apply_steps": 10, "gripper": "keep", **kwargs}
    with pytest.raises(ValueError):
        edit_native(adapter, observation(), np.zeros((10, 7), np.float32), **arguments)


def test_online_noise_preserves_recovered_physical_channels_and_replaces_only_padding():
    recovered = np.arange(320, dtype=np.float32).reshape(1, 10, 32)
    source = recovered.copy()
    result = resample_padding_noise(recovered, np.random.default_rng(19))
    expected_padding = (
        np.random.default_rng(19).standard_normal((1, 10, 25)).astype(np.float32)
    )
    np.testing.assert_array_equal(result["noise"][..., :7], recovered[..., :7])
    np.testing.assert_array_equal(result["noise"][..., 7:], expected_padding)
    np.testing.assert_array_equal(result["source_noise"], recovered)
    np.testing.assert_array_equal(recovered, source)
    assert result["receipt"]["first7_preserved"]
    assert result["receipt"]["known_generating_noise"] is None


def test_loop_projects_noise_before_execution_and_retains_unprojected_inverse():
    recovered = np.random.default_rng(2).standard_normal((1, 10, 32)).astype(np.float32)
    result = resample_padding_noise(
        recovered, np.random.default_rng(3), repeat_mean=True
    )
    expected = recovered[0, :, :7].astype(np.float64).mean(axis=0).astype(np.float32)
    np.testing.assert_array_equal(
        result["noise"][0, :, :7], np.broadcast_to(expected, (10, 7))
    )
    assert result["receipt"]["repeat_mean_before_execution"]
    assert not result["receipt"]["first7_preserved"]
    np.testing.assert_array_equal(result["source_noise"], recovered)


def test_matched_baseline_draws_one_unit_variance_vector_and_independent_padding():
    result = repeated_gaussian_noise(np.random.default_rng(41))
    reference = np.random.default_rng(41)
    vector = reference.standard_normal(7).astype(np.float32)
    padding = reference.standard_normal((1, 10, 25)).astype(np.float32)
    np.testing.assert_array_equal(
        result["noise"][0, :, :7], np.broadcast_to(vector, (10, 7))
    )
    np.testing.assert_array_equal(result["noise"][..., 7:], padding)
    assert result["receipt"]["physical_gaussian_variance"] == 1.0
    assert not result["receipt"]["averaged_independent_gaussian_rows"]


def test_wrong_noise_dtype_or_shape_fails_closed():
    for noise in (
        np.zeros((10, 32), np.float32),
        np.zeros((1, 10, 32), np.float64),
        np.full((1, 10, 32), np.nan, np.float32),
    ):
        with pytest.raises(ValueError):
            resample_padding_noise(noise, np.random.default_rng(0))
