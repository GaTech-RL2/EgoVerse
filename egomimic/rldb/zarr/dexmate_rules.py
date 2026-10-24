"""
Array rules for dexmate_bimanual episodes: a Dexmate robot with two Sharpa Wave
hands. The episode validator runs check_dexmate on top of its general checks
(images.front_1, {side}.obs_ee_pose, pose quaternions, features); the SharpaDex
converter writes the keys and imports the constants below.

Every per-side key is f"{side}.{suffix}". Besides the arrays listed here an
episode stores {side}.obs_ee_pose (palm pose, required) and may store
{side}.cmd_ee_pose, images.{side}_wrist and {side}.obs_tactile_image (one RGB
JPEG strip per frame). All tracks are in the robot base frame.

Torch-free, like the validator. The FK residual and TCP side checks need the
URDFs and torch, so they stay in the converter's verify step.
"""

import math
from typing import Any

import numpy as np

from egomimic.rldb.zarr.episode_rules import QUAT_UNIT_TOL

EMBODIMENT = "dexmate_bimanual"
SIDES = ("left", "right")

ARM_DOF = 7
HAND_DOF = 22
# 5 fingertip force/torque sensors, thumb to pinky, each (fx, fy, fz, tx, ty, tz).
TACTILE_DIM = 30
# 21 hand keypoints in the MANO layout, xyz each.
KEYPOINT_DIM = 63

# Columns of {side}.{obs,cmd}_joints, as named in the Dexmate URDF.
ARM_JOINT_NAMES = {
    side: tuple(f"{side}_arm_joint_{i}" for i in range(1, ARM_DOF + 1))
    for side in SIDES
}
# Columns of {obs,cmd}_aux_joints: the lower body, then the neck.
AUX_JOINT_NAMES = (
    "lower_body_joint_1",
    "lower_body_joint_2",
    "lower_body_joint_3",
    "lower_body_joint_4",
    "lower_body_joint_5",
    "neck_joint_1",
    "neck_joint_2",
)
AUX_DOF = len(AUX_JOINT_NAMES)

# Columns of {side}.{obs,cmd}_hand_joints with their (lower, upper) limits in
# radians, from {side}_sharpa_wave_with_flange.urdf; both hands share them.
_HAND_JOINTS = (
    ("thumb_CMC_FE", -0.1745, 1.9199),
    ("thumb_CMC_AA", -0.3491, 0.3491),
    ("thumb_MCP_FE", -0.5236, 1.3963),
    ("thumb_MCP_AA", -0.3491, 0.3491),
    ("thumb_IP", 0.0, 1.7453),
    ("index_MCP_FE", -0.17453293, 1.5708),
    ("index_MCP_AA", -0.3491, 0.3491),
    ("index_PIP", 0.0, 1.7453),
    ("index_DIP", 0.0, 1.3963),
    ("middle_MCP_FE", -0.17453293, 1.5708),
    ("middle_MCP_AA", -0.3491, 0.3491),
    ("middle_PIP", 0.0, 1.7453),
    ("middle_DIP", 0.0, 1.3963),
    ("ring_MCP_FE", -0.17453293, 1.5708),
    ("ring_MCP_AA", -0.3491, 0.3491),
    ("ring_PIP", 0.0, 1.7453),
    ("ring_DIP", 0.0, 1.3963),
    ("pinky_CMC", 0.0, 0.2618),
    ("pinky_MCP_FE", -0.17453293, 1.5708),
    ("pinky_MCP_AA", -0.3491, 0.3491),
    ("pinky_PIP", 0.0, 1.7453),
    ("pinky_DIP", 0.0, 1.3963),
)
HAND_JOINT_NAMES = {
    side: tuple(f"{side}_{name}" for name, _, _ in _HAND_JOINTS) for side in SIDES
}
HAND_JOINT_LIMITS = {
    side: tuple((lower, upper) for _, lower, upper in _HAND_JOINTS) for side in SIDES
}

# Per-side suffix -> trailing width. Every one is (T, width) and finite.
REQUIRED_SIDE_ARRAYS = {
    "obs_joints": ARM_DOF,
    "cmd_joints": ARM_DOF,
    "obs_hand_joints": HAND_DOF,
    "cmd_hand_joints": HAND_DOF,
    "obs_hand_keypoints": KEYPOINT_DIM,
    "obs_tactile": TACTILE_DIM,
}
# FK of cmd_hand_joints, laid out like obs_hand_keypoints.
OPTIONAL_SIDE_ARRAYS = {"cmd_hand_keypoints": KEYPOINT_DIM}
# The head link pose (xyz + wxyz) is the frame training re-expresses the hand
# tracks in, so it is required. It is not obs_head_pose: that name is reserved
# for a calibrated front_1 camera pose, which Dexmate episodes do not have.
HEAD_LINK_POSE = "obs_head_link_pose"
REQUIRED_ARRAYS = {
    "obs_aux_joints": AUX_DOF,
    "cmd_aux_joints": AUX_DOF,
    HEAD_LINK_POSE: 7,
}

# Hand joints past HAND_JOINT_LIMITS, in radians. An observed joint more than
# ERROR_MARGIN_RAD (15 deg) past a limit is an error: that points at degrees,
# shifted columns or a corrupt frame, not at the hand. A smaller excursion past
# WARN_MARGIN_RAD (2 deg), or any commanded one, is a warning: the hand firmware
# stops near the limits and teleoperation commands overshoot them.
ERROR_MARGIN_RAD = 0.26
WARN_MARGIN_RAD = 0.035

# A joint value more than SPIKE_JUMP_RAD away from both neighbours (which agree
# with each other), or beyond SPIKE_MAX_ABS_RAD, marks a corrupt frame (warning).
SPIKE_JUMP_RAD = 1.0
SPIKE_MAX_ABS_RAD = 2 * math.pi
_SPIKE_SUFFIXES = ("obs_joints", "cmd_joints", "obs_hand_joints", "cmd_hand_joints")


def check_dexmate(store: Any, T: int, errors: list[str], warnings: list[str]) -> None:
    """Append the dexmate_bimanual array problems of ``store`` to ``errors`` and
    ``warnings``.

    ``store`` is the episode's zarr group and ``T`` its total_frames. Arrays are
    zero-padded to the chunk size, so only ``[:T]`` is read; whether each array
    has at least T rows is the validator's general check, not repeated here.
    """
    required = {
        f"{side}.{suffix}": width
        for side in SIDES
        for suffix, width in REQUIRED_SIDE_ARRAYS.items()
    }
    required.update(REQUIRED_ARRAYS)
    optional = {
        f"{side}.{suffix}": width
        for side in SIDES
        for suffix, width in OPTIONAL_SIDE_ARRAYS.items()
    }
    values = {}
    for key, width in {**required, **optional}.items():
        if key not in store:
            if key in required:
                errors.append(f"Missing required key: {key}")
            continue
        if (arr := _finite_values(store, key, width, T, errors)) is not None:
            values[key] = arr

    if (pose := values.get(HEAD_LINK_POSE)) is not None:
        norms = np.linalg.norm(pose[:, 3:7], axis=1)
        bad = np.where(np.abs(norms - 1.0) > QUAT_UNIT_TOL)[0]
        if len(bad):
            errors.append(
                f"{HEAD_LINK_POSE}: {len(bad)} frames with non-unit quaternions "
                f"(e.g. frame {bad[0]}, norm={norms[bad[0]]:.6f})"
            )
    for side in SIDES:
        for prefix in ("obs", "cmd"):
            key = f"{side}.{prefix}_hand_joints"
            if key in values:
                _check_hand_limits(
                    key, values[key], side, prefix == "obs", errors, warnings
                )
    spike_keys = [f"{side}.{s}" for side in SIDES for s in _SPIKE_SUFFIXES]
    for key in spike_keys + ["obs_aux_joints", "cmd_aux_joints"]:
        if key in values:
            _check_spikes(key, values[key], warnings)


def _finite_values(
    store: Any, key: str, width: int, T: int, errors: list[str]
) -> np.ndarray | None:
    """``store[key][:T]`` as float64 if it is (T, width) and finite, else None."""
    shape = tuple(store[key].shape)
    if len(shape) != 2 or shape[1] != width:
        errors.append(f"{key}: expected shape (T, {width}), got {shape}")
        return None
    values = np.asarray(store[key][:T], dtype=np.float64)
    if not np.isfinite(values).all():
        errors.append(f"{key}: contains NaN or inf")
        return None
    return values if len(values) else None


def _check_hand_limits(
    key: str,
    values: np.ndarray,
    side: str,
    observed: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    limits = np.asarray(HAND_JOINT_LIMITS[side], dtype=np.float64)
    # > 0 where a value is outside [lower, upper], by that much.
    excess = np.maximum(limits[:, 0] - values, values - limits[:, 1])
    if observed and (excess > ERROR_MARGIN_RAD).any():
        errors.append(
            _limit_message(key, side, excess, ERROR_MARGIN_RAD)
            + "; check units, joint order and corrupt frames"
        )
    elif (excess > WARN_MARGIN_RAD).any():
        warnings.append(
            _limit_message(key, side, excess, WARN_MARGIN_RAD)
            + "; expected near firmware stops and in teleoperation commands"
        )


def _limit_message(key: str, side: str, excess: np.ndarray, margin: float) -> str:
    outside = excess > margin
    columns = np.nonzero(outside.any(axis=0))[0]
    worst = int(columns[np.argmax(excess[:, columns].max(axis=0))])
    lower, upper = HAND_JOINT_LIMITS[side][worst]
    return (
        f"{key}: {len(columns)} joint(s) more than {margin:g} rad past the URDF "
        f"limits on {int(outside.any(axis=1).sum())} frame(s); worst "
        f"{HAND_JOINT_NAMES[side][worst]} by {float(excess[:, worst].max()):.4f} "
        f"outside [{lower:g}, {upper:g}]"
    )


def _check_spikes(key: str, values: np.ndarray, warnings: list[str]) -> None:
    bad = np.abs(values) > SPIKE_MAX_ABS_RAD
    if len(values) >= 3:
        middle = values[1:-1]
        bad[1:-1] |= (
            (np.abs(middle - values[:-2]) > SPIKE_JUMP_RAD)
            & (np.abs(values[2:] - middle) > SPIKE_JUMP_RAD)
            & (np.abs(values[2:] - values[:-2]) < SPIKE_JUMP_RAD / 2)
        )
    if not bad.any():
        return
    frames = np.nonzero(bad.any(axis=1))[0]
    column = int(np.nonzero(bad[frames[0]])[0][0])
    warnings.append(
        f"{key}: {len(frames)} frame(s) hold a one-frame jump of more than "
        f"{SPIKE_JUMP_RAD:g} rad or a value beyond {SPIKE_MAX_ABS_RAD:.4f} "
        f"(first at frame {int(frames[0])}, {_column_name(key, column)} = "
        f"{values[frames[0], column]:.4g}); treat them as corrupt frames"
    )


def _column_name(key: str, column: int) -> str:
    side, _, suffix = key.rpartition(".")
    if suffix.endswith("aux_joints"):
        return AUX_JOINT_NAMES[column]
    if suffix.endswith("hand_joints"):
        return HAND_JOINT_NAMES[side][column]
    return ARM_JOINT_NAMES[side][column]
