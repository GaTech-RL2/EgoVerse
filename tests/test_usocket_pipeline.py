import numpy as np
import pytest
import torch

from egomimic.pipeline.pushshapes import (
    USocketArcLengthRolloutAdapter,
    USocketRotVecRolloutAdapter,
)
from egomimic.rldb.zarr.action_chunk_transforms import (
    RotVecToTheta,
    ThetaToRotVec,
)
from egomimic.rldb.zarr.arc_length_tokenizer import TokenizeUSocketArcLength


def test_usocket_theta_rotvec_round_trip_across_wrap() -> None:
    actions = np.array(
        [
            [10.0, 20.0, -np.pi + 1e-4],
            [11.0, 21.0, np.pi - 1e-4],
            [12.0, 22.0, 0.5],
        ],
        dtype=np.float32,
    )
    encoded = ThetaToRotVec(keys=["actions"]).transform({"actions": actions.copy()})[
        "actions"
    ]
    decoded = RotVecToTheta(keys=["actions"]).transform({"actions": encoded.copy()})[
        "actions"
    ]

    assert encoded.shape == (3, 4)
    assert encoded.dtype == actions.dtype
    np.testing.assert_allclose(
        np.square(encoded[:, 2]) + np.square(encoded[:, 3]), 1.0, atol=1e-6
    )
    np.testing.assert_allclose(decoded[:, :2], actions[:, :2], atol=1e-6)
    np.testing.assert_allclose(
        np.angle(np.exp(1j * decoded[:, 2])),
        np.angle(np.exp(1j * actions[:, 2])),
        atol=1e-6,
    )


def test_usocket_theta_rotvec_rejects_malformed_present_key() -> None:
    with pytest.raises(ValueError, match="angle_col=2"):
        ThetaToRotVec(keys=["actions"]).transform(
            {"actions": np.zeros((4, 2), dtype=np.float32)}
        )


def test_usocket_rollout_adapter_returns_xy_theta() -> None:
    actions = torch.tensor([[[10.0, 20.0, 0.0, 1.0], [11.0, 21.0, -1.0, 0.0]]])

    decoded = USocketRotVecRolloutAdapter().decode(actions)

    assert decoded.shape == (1, 2, 3)
    torch.testing.assert_close(decoded[..., :2], actions[..., :2])
    torch.testing.assert_close(decoded[..., 2], torch.tensor([[np.pi / 2.0, np.pi]]))


def test_usocket_arc_tokenizer_produces_uniform_waypoints_and_timing() -> None:
    actions = np.column_stack(
        [
            np.linspace(0.0, 200.0, 101),
            np.zeros(101),
            np.zeros(101),
        ]
    ).astype(np.float32)
    tokenizer = TokenizeUSocketArcLength(
        min_distance_unit=200.0,
        resampled_vector_length=25,
        dt=1.0 / 30.0,
        rotation_radius=40.0,
    )

    token = tokenizer.transform({"actions": actions.copy()})["actions"]
    decoded = tokenizer.detokenize(token, action_horizon=101)

    assert token.shape == (26, 4)
    assert token.dtype == np.float32
    np.testing.assert_allclose(token[:25, 0], np.linspace(0.0, 200.0, 25))
    np.testing.assert_allclose(
        token[:25, 2:], np.tile(np.array([[1.0, 0.0]]), (25, 1)), atol=1e-6
    )
    np.testing.assert_allclose(token[25], [60.0, 0.0, 0.0, 60.0], atol=1e-5)
    np.testing.assert_allclose(decoded, actions, atol=1e-4)


def test_usocket_arc_tokenizer_preserves_rotation_in_place_across_wrap() -> None:
    theta_unwrapped = np.linspace(np.pi - 0.2, np.pi + 0.2, 61)
    actions = np.column_stack(
        [np.full(61, 100.0), np.full(61, 200.0), theta_unwrapped]
    ).astype(np.float32)
    actions[:, 2] = np.angle(np.exp(1j * actions[:, 2]))
    tokenizer = TokenizeUSocketArcLength(
        min_distance_unit=200.0,
        resampled_vector_length=25,
        dt=1.0 / 30.0,
        rotation_radius=40.0,
    )

    token = tokenizer.transform({"actions": actions.copy()})["actions"]
    decoded = tokenizer.detokenize(token, action_horizon=61)

    assert token[-1, 3] > 0.0
    np.testing.assert_allclose(decoded[:, :2], actions[:, :2], atol=1e-5)
    np.testing.assert_allclose(
        np.angle(np.exp(1j * decoded[:, 2])), actions[:, 2], atol=1e-5
    )


def test_usocket_arc_rollout_adapter_returns_fixed_rate_xy_theta() -> None:
    actions = np.column_stack(
        [
            np.linspace(10.0, 110.0, 101),
            np.linspace(20.0, 70.0, 101),
            np.linspace(-0.2, 0.4, 101),
        ]
    ).astype(np.float32)
    tokenizer = TokenizeUSocketArcLength(
        min_distance_unit=200.0,
        resampled_vector_length=25,
    )
    token = tokenizer.transform({"actions": actions.copy()})["actions"]
    adapter = USocketArcLengthRolloutAdapter(
        min_distance_unit=200.0,
        resampled_vector_length=25,
        action_horizon=100,
    )

    decoded = adapter.decode(torch.from_numpy(token).unsqueeze(0))

    assert decoded.shape == (1, 100, 3)
    assert torch.isfinite(decoded).all()
    torch.testing.assert_close(decoded[0, 0], torch.from_numpy(actions[0]))
