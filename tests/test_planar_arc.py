import math

import numpy as np
import pytest
import torch

from egomimic.pipeline.pushshapes import (
    PlanarArcWaypointZeroNativeDecoder,
    PlanarCommon5NativeDecoder,
)
from egomimic.rldb.zarr.planar_arc import (
    PadPlanarAction,
    TokenizePlanarArcLength,
    TokenizeUSocketArcVelocity,
)


@pytest.mark.parametrize("width", [2, 3, 4])
def test_pad_planar_action_has_common_layout(width):
    native = np.arange(3 * width, dtype=np.float32).reshape(3, width) / 10
    output = PadPlanarAction().transform({"actions": native.copy()})["actions"]
    assert output.shape == (3, 5)
    np.testing.assert_allclose(output[:, :2], native[:, :2])
    theta = native[:, 2] if width >= 3 else np.zeros(3)
    grip = native[:, 3] if width == 4 else np.zeros(3)
    np.testing.assert_allclose(output[:, 2], np.cos(theta))
    np.testing.assert_allclose(output[:, 3], np.sin(theta))
    np.testing.assert_allclose(output[:, 4], grip)


def test_planar_arc_has_two_streams_and_no_mean_speed_row():
    action = np.column_stack(
        (np.arange(6, dtype=np.float64), np.zeros(6), np.linspace(0, 0.5, 6))
    )
    token = TokenizeUSocketArcVelocity(
        min_distance_unit=3, resampled_vector_length=4, dt=0.5
    ).tokenize(action)
    assert token.shape == (8, 5)
    translation, rotation = token[:4], token[4:]
    np.testing.assert_allclose(translation[:, 2:4], 0)
    np.testing.assert_allclose(rotation[:, :2], 0)
    np.testing.assert_allclose(translation[:, 4], 2.0)
    np.testing.assert_allclose(rotation[:, 4], 0.2)
    np.testing.assert_allclose(translation[0, :2], action[0, :2])
    np.testing.assert_allclose(rotation[0, 2:4], [1, 0], atol=1e-8)


def test_existing_robot_arc_token_schema_is_unchanged():
    action = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.1]])
    token = TokenizePlanarArcLength(resampled_vector_length=4).tokenize(action)
    assert token.shape == (5, 5)


def test_local_velocities_are_not_replaced_by_a_chunk_mean():
    action = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [4.0, 0.0, 0.4]])
    token = TokenizeUSocketArcVelocity(
        min_distance_unit=4,
        rotation_distance_unit=0.4,
        resampled_vector_length=3,
        dt=1.0,
    ).tokenize(action)
    linear = token[:3, 4]
    angular = token[3:, 4]
    assert linear[0] != pytest.approx(linear[1])
    assert angular[0] != pytest.approx(angular[1])


def test_signed_angular_velocity_is_preserved():
    action = np.array([[0.0, 0.0, 0.4], [0.0, 0.0, 0.2], [0.0, 0.0, -0.2]])
    token = TokenizeUSocketArcVelocity(
        min_distance_unit=1,
        rotation_distance_unit=0.6,
        resampled_vector_length=4,
        dt=0.5,
    ).tokenize(action)
    assert np.all(token[4:, 4] < 0)


def test_translation_and_rotation_have_independent_budgets():
    action = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.5], [20.0, 0.0, 1.0]])
    token = TokenizeUSocketArcVelocity(
        min_distance_unit=20,
        rotation_distance_unit=0.25,
        resampled_vector_length=3,
    ).tokenize(action)
    np.testing.assert_allclose(token[2, :2], [20.0, 0.0])
    assert math.atan2(token[5, 3], token[5, 2]) == pytest.approx(0.25)


@pytest.mark.parametrize("budget", [0.0, -1.0, float("nan")])
def test_rotation_budget_must_be_positive_and_finite(budget):
    with pytest.raises(ValueError, match="rotation_distance_unit"):
        TokenizeUSocketArcVelocity(rotation_distance_unit=budget)


def test_zero_motion_holds_both_streams_and_zeroes_velocities():
    action = np.repeat(np.array([[4.0, 7.0, 0.5]]), 5, axis=0)
    token = TokenizeUSocketArcVelocity(resampled_vector_length=3).tokenize(action)
    np.testing.assert_allclose(token[:3, :2], [[4, 7]] * 3)
    np.testing.assert_allclose(token[:, 4], 0)
    np.testing.assert_allclose(token[3:, 2], math.cos(0.5))
    np.testing.assert_allclose(token[3:, 3], math.sin(0.5))


@pytest.mark.parametrize("native_dim", [2, 3])
def test_common_and_arc_adapters_decode_same_anchor(native_dim):
    dense_token = torch.tensor(
        [[[2.0, 3.0, 0.0, 1.0, 0.0], [9.0, 8.0, 1.0, 0.0, 0.0]]]
    )
    dense = PlanarCommon5NativeDecoder(2, native_dim).decode(dense_token)
    arc_token = torch.zeros(1, 4, 5)
    arc_token[:, 0, :2] = dense_token[:, 0, :2]
    arc_token[:, 2, 2:4] = dense_token[:, 0, 2:4]
    arc = PlanarArcWaypointZeroNativeDecoder(2, native_dim).decode(arc_token)
    torch.testing.assert_close(arc, dense[:, :1])


def test_arc_accepts_common_five_and_rejects_nonfinite_or_short_input():
    transform = TokenizeUSocketArcVelocity()
    native = np.array([[0.0, 0.0, 0.2], [1.0, 0.0, 0.3]])
    common = PadPlanarAction().transform({"actions": native.copy()})["actions"]
    np.testing.assert_allclose(transform.tokenize(native), transform.tokenize(common))
    with pytest.raises(ValueError):
        transform.transform({"actions": np.zeros((1, 3))})
    with pytest.raises(ValueError):
        transform.transform({"actions": np.array([[0, 0], [np.nan, 1]])})
    with pytest.raises(ValueError, match="U-Socket"):
        transform.transform({"actions": np.zeros((2, 4))})
