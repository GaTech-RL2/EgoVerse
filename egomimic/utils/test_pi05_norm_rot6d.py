"""Round-trip tests for the normalized continuous-6D rotation encoding.

Covers the data transform (ypr <-> 6D) and the converter 32D packers
(``to32_norm_6d`` / ``from32_norm_6d``) for both the robot bimanual (14D ypr /
20D 6D, with gripper) and human bimanual (12D ypr / 18D 6D, no gripper) layouts.
"""

import numpy as np
import pytest
import torch

from egomimic.rldb.zarr.action_chunk_transforms import (
    CartesianRot6DToYPR,
    CartesianYPRToRot6D,
)
from egomimic.utils.action_utils import (
    BaseActionConverter,
    HumanBimanualCartesianEuler,
    RobotBimanualCartesianEuler,
)
from egomimic.utils.pose_utils import _rot6d_to_ypr, _ypr_to_rot6d


def _eva_ypr_chunk(T: int = 5) -> np.ndarray:
    # [L xyz ypr g, R xyz ypr g]; moderate angles to avoid gimbal/wrap ambiguity.
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-1.0, 1.0, size=(T, 3))
    ypr = rng.uniform(-1.0, 1.0, size=(T, 3))  # radians, well inside (-pi, pi)
    g = rng.uniform(0.0, 1.0, size=(T, 1))
    arm = np.concatenate([xyz, ypr, g], axis=-1)
    return np.concatenate([arm, arm], axis=-1)  # 14D


def _aria_ypr_chunk(T: int = 5) -> np.ndarray:
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-1.0, 1.0, size=(T, 3))
    ypr = rng.uniform(-1.0, 1.0, size=(T, 3))
    arm = np.concatenate([xyz, ypr], axis=-1)
    return np.concatenate([arm, arm], axis=-1)  # 12D


def test_ypr_rot6d_helpers_round_trip():
    ypr = np.random.default_rng(2).uniform(-1.0, 1.0, size=(7, 3))
    six = _ypr_to_rot6d(ypr)
    assert six.shape == (7, 6)
    np.testing.assert_allclose(_rot6d_to_ypr(six), ypr, atol=1e-6)


@pytest.mark.parametrize(
    "chunk_fn,ypr_dim,six_dim",
    [(_eva_ypr_chunk, 14, 20), (_aria_ypr_chunk, 12, 18)],
)
def test_cartesian_ypr_rot6d_transform_round_trips(chunk_fn, ypr_dim, six_dim):
    ypr = chunk_fn()
    assert ypr.shape[-1] == ypr_dim

    fwd = CartesianYPRToRot6D(action_key="actions_cartesian")
    rev = CartesianRot6DToYPR(action_key="actions_cartesian")

    batch = {"actions_cartesian": ypr.copy()}
    batch = fwd.transform(batch)
    assert batch["actions_cartesian"].shape[-1] == six_dim

    batch = rev.transform(batch)
    np.testing.assert_allclose(batch["actions_cartesian"], ypr, atol=1e-6)


def test_transform_preserves_tensor_type():
    ypr = torch.from_numpy(_eva_ypr_chunk())
    out = CartesianYPRToRot6D().transform({"actions_cartesian": ypr})[
        "actions_cartesian"
    ]
    assert isinstance(out, torch.Tensor)
    assert out.shape[-1] == 20


def test_robot_bimanual_norm_6d_pack_round_trips():
    converter = RobotBimanualCartesianEuler()
    six = torch.from_numpy(_eva_ypr_chunk()).float()
    six6d = torch.from_numpy(
        CartesianYPRToRot6D().transform({"actions_cartesian": six.numpy()})[
            "actions_cartesian"
        ]
    ).float()[None]  # (1, T, 20)

    packed = converter.to32_norm_6d(six6d)
    assert packed.shape[-1] == 32
    decoded = converter.from32_norm_6d(packed)
    torch.testing.assert_close(decoded, six6d, atol=1e-6, rtol=1e-6)


def test_human_bimanual_norm_6d_pack_round_trips_and_zeros_gripper():
    converter = HumanBimanualCartesianEuler()
    six6d = torch.from_numpy(
        CartesianYPRToRot6D().transform({"actions_cartesian": _aria_ypr_chunk()})[
            "actions_cartesian"
        ]
    ).float()[None]  # (1, T, 18)

    packed = converter.to32_norm_6d(six6d)
    assert packed.shape[-1] == 32
    # gripper slots (9, 19) must be zero for human (no gripper signal).
    torch.testing.assert_close(packed[..., 9], torch.zeros_like(packed[..., 9]))
    torch.testing.assert_close(packed[..., 19], torch.zeros_like(packed[..., 19]))

    decoded = converter.from32_norm_6d(packed)
    torch.testing.assert_close(decoded, six6d, atol=1e-6, rtol=1e-6)


def test_base_converter_rejects_norm_6d_encoding():
    converter = BaseActionConverter()
    with pytest.raises(NotImplementedError, match="normalized-rot6d"):
        converter.to32_norm_6d(torch.zeros(1, 1, 20))
    with pytest.raises(NotImplementedError, match="normalized-rot6d"):
        converter.from32_norm_6d(torch.zeros(1, 1, 32))
