import numpy as np
import pytest
import torch

from egomimic.pipeline.pushshapes import USocketRotVecRolloutAdapter
from egomimic.rldb.zarr.action_chunk_transforms import (
    RotVecToTheta,
    ThetaToRotVec,
)


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
