"""Compare the joints a vendor shipped against the keypoints they shipped.

A dexterous end-effector stores both representations (`DEXTEROUS_EMBODIMENT_
DESIGN_V2.md` §2.4): ``obs_hand_joints`` is what the controller consumed, and
``obs_hand_keypoints`` is the MANO-topology tensor that trains through the
shared head. Neither one can check the other on its own. With the registry URDF
alongside them, "did the vendor wire this up correctly" becomes a scalar: run
forward kinematics on the joints and measure the distance to the keypoints.

The comparison happens in the hand-root frame, which ``obs_ee_pose`` defines,
so it does not depend on whether the episode expresses keypoints in a camera
frame or a base frame.
"""

from __future__ import annotations

import functools

import numpy as np

from egomimic.rldb.embodiment.registry import EndEffectorSpec
from egomimic.rldb.embodiment.urdf import UrdfChain, UrdfError, load_urdf
from egomimic.utils.pose_utils import _xyzwxyz_to_matrix


@functools.lru_cache(maxsize=8)
def _chain(path: str) -> UrdfChain:
    """Load and cache one URDF; a registry URDF is shared by many episodes."""
    return load_urdf(path)


def load_chain(spec: EndEffectorSpec) -> UrdfChain:
    """Load the URDF an end-effector declares.

    Args:
        spec: A registry entry whose ``urdf`` field is set.

    Returns:
        The parsed kinematic tree.

    Raises:
        UrdfError: If the entry declares no URDF, or the file is unreadable.
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
        A ``(T, n_valid, 3)`` array of root-frame positions, ordered by the
        entry's valid keypoint slots.

    Raises:
        UrdfError: If the URDF cannot be read, or names no joint or link the
            entry declares.
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
    """Return the per-frame, per-slot distance between the two representations.

    Args:
        spec: The registry entry for this end-effector.
        joints: A ``(T, dof)`` joint array in ``joint_names`` order.
        keypoints: A ``(T, 3 * n_slots)`` array of the vendor's keypoints, in
            the same frame as ``ee_poses``.
        ee_poses: A ``(T, 7)`` array of ``[x, y, z, qw, qx, qy, qz]`` hand-root
            poses in that frame.

    Returns:
        A ``(T, n_valid)`` array of metres. Each entry is the distance between
        the vendor's keypoint and the one forward kinematics places there.

    Raises:
        UrdfError: If the URDF cannot be read or does not match the entry.
        ValueError: If an array does not have the shape the entry implies.
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
    # Map the vendor's keypoints out of the episode frame and into the hand
    # root, which is the frame forward kinematics reports.
    stored_root = np.einsum("tij,tkj->tki", root_T_world[:, :3, :3], stored)
    stored_root += root_T_world[:, None, :3, 3]
    return np.linalg.norm(fk_keypoints(spec, joints) - stored_root, axis=-1)


__all__ = ["fk_keypoints", "keypoint_residuals", "load_chain"]
