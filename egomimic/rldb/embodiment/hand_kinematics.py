"""Validate stored dexterous-hand keypoints against forward kinematics.

``obs_hand_joints`` stores the observed joint configuration in registry
``joint_names`` order. ``obs_hand_keypoints`` stores all slots in the declared
topology in the same episode coordinate frame as ``obs_ee_pose``. The registry
URDF and ``keypoint_links`` mapping convert the joint configuration into
hand-root-frame positions for the valid slots. This module transforms the
stored keypoints into that frame and returns their Euclidean distance from the
forward-kinematics positions.
"""

from __future__ import annotations

import functools

import numpy as np

from egomimic.rldb.embodiment.registry import EndEffectorSpec
from egomimic.rldb.embodiment.urdf import UrdfChain, UrdfError, load_urdf
from egomimic.utils.pose_utils import _xyzwxyz_to_matrix


@functools.lru_cache(maxsize=8)
def _chain(path: str) -> UrdfChain:
    """Load a URDF and cache the parsed chain by path."""
    return load_urdf(path)


def load_chain(spec: EndEffectorSpec) -> UrdfChain:
    """Load the URDF an end-effector declares.

    Args:
        spec: A registry entry whose ``urdf`` field is set.

    Returns:
        The parsed kinematic tree.

    Raises:
        UrdfError: If the entry declares no URDF or the file cannot be parsed
            as a supported kinematic tree.
    """
    path = spec.urdf_path
    if path is None:
        raise UrdfError(f"end-effector {spec.name!r} declares no `urdf`")
    return _chain(str(path))


def fk_keypoints(spec: EndEffectorSpec, joints: np.ndarray) -> np.ndarray:
    """Return the hand-root-frame keypoints implied by a joint track.

    Args:
        spec: A registry entry declaring ``urdf``, ``joint_names`` and
            ``keypoint_links``.
        joints: A ``(T, dof)`` array of joint values in radians or metres, in
            the entry's ``joint_names`` order.

    Returns:
        A ``(T, n_valid, 3)`` array of hand-root-frame positions. Axis 1 follows
        ``keypoint_links``, which the registry loader sorts by slot index.

    Raises:
        UrdfError: If the URDF is invalid, a declared joint is not actuated, or
            a declared keypoint link is not reachable from the URDF root.
        ValueError: If ``joints`` does not have the declared width.
    """
    chain = load_chain(spec)
    joints = np.asarray(joints, dtype=float)
    if joints.ndim != 2 or joints.shape[1] != len(spec.joint_names):
        raise ValueError(
            f"expected a (T, {len(spec.joint_names)}) joint array for "
            f"{spec.name!r}, got shape {joints.shape}"
        )
    known = set(chain.actuated_joint_names)
    unknown = [name for name in spec.joint_names if name not in known]
    if unknown:
        raise UrdfError(
            f"joint_names {unknown} are not actuated joints of {spec.urdf}; "
            f"it actuates {list(chain.actuated_joint_names)}"
        )
    links = [link for _, link in spec.keypoint_links]
    return np.stack(
        [chain.link_positions(dict(zip(spec.joint_names, row)), links) for row in joints]
    )


def keypoint_residuals(
    spec: EndEffectorSpec,
    joints: np.ndarray,
    keypoints: np.ndarray,
    ee_poses: np.ndarray,
) -> np.ndarray:
    """Return FK residuals for every frame and valid keypoint slot.

    Args:
        spec: The end-effector registry entry. It declares the topology, valid
            slots, joint order, URDF, and slot-to-link mapping.
        joints: A ``(T, dof)`` joint array in ``joint_names`` order.
        keypoints: A ``(T, 3 * n_slots)`` array of the vendor's keypoints, in
            the same frame as ``ee_poses``.
        ee_poses: A ``(T, 7)`` array of ``[x, y, z, qw, qx, qy, qz]`` hand-root
            poses in that frame.

    Returns:
        A ``(T, n_valid)`` array of Euclidean distances in metres. Axis 1 is in
        increasing valid-slot order.

    Raises:
        UrdfError: If the URDF cannot be read or does not match the entry.
        ValueError: If an input does not have the required rank or width.
    """
    keypoints = np.asarray(keypoints, dtype=float)
    ee_poses = np.asarray(ee_poses, dtype=float)
    n_slots = spec.keypoints.n_slots
    if keypoints.ndim != 2 or keypoints.shape[1] != 3 * n_slots:
        raise ValueError(
            f"expected a (T, {3 * n_slots}) keypoint array for {spec.name!r}, "
            f"got shape {keypoints.shape}"
        )
    if ee_poses.shape != (keypoints.shape[0], 7):
        raise ValueError(
            f"expected a ({keypoints.shape[0]}, 7) pose array, got shape "
            f"{ee_poses.shape}"
        )

    root_T_world = np.linalg.inv(_xyzwxyz_to_matrix(ee_poses))
    valid = list(spec.keypoints.valid)
    stored = keypoints.reshape(-1, n_slots, 3)[:, valid, :]
    # Convert episode-frame points to the hand-root frame returned by FK.
    stored_root = np.einsum("tij,tkj->tki", root_T_world[:, :3, :3], stored)
    stored_root += root_T_world[:, None, :3, 3]
    return np.linalg.norm(fk_keypoints(spec, joints) - stored_root, axis=-1)


__all__ = ["fk_keypoints", "keypoint_residuals", "load_chain"]
