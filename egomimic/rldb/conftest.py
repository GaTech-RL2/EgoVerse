"""Fixtures for dexterous-hand schema and kinematics tests.

The fixtures generate a 20-degree-of-freedom hand URDF and matching registry
entry at runtime. ``dexterous_registry`` temporarily makes the existing
``eva_bimanual`` identifier resolve to those generated specifications because
``ZarrWriter`` accepts only names in ``EMBODIMENT``. No synthetic hardware
entry is added to the production registry.
"""

from __future__ import annotations

import numpy as np
import pytest

from egomimic.rldb.embodiment.registry import (
    _parse_end_effector,
    _parse_platform,
    load_end_effectors,
)

#: Five serial finger chains with four revolute joints each produce 20 columns.
FINGERS = 5
JOINTS_PER_FINGER = 4
HAND_DOF = FINGERS * JOINTS_PER_FINGER
#: Column order for the generated ``(T, HAND_DOF)`` joint arrays.
HAND_JOINT_NAMES = [
    f"j_{finger}_{link}"
    for finger in range(FINGERS)
    for link in range(JOINTS_PER_FINGER)
]
#: MANO-21 slot 0 maps to the palm; slots 1 through 20 map to ``kp1``–``kp20``.
HAND_KEYPOINT_LINKS = {0: "palm"} | {slot: f"kp{slot}" for slot in range(1, 21)}
HAND_TOLERANCE_M = 1e-6


def _hand_urdf() -> str:
    """Return a URDF with five four-joint serial chains rooted at the palm."""
    parts = ['<robot name="test_hand">', '  <link name="palm"/>']
    for finger in range(FINGERS):
        parent = "palm"
        for link in range(JOINTS_PER_FINGER):
            slot = finger * JOINTS_PER_FINGER + link + 1
            # Spread the fingers across x so no two links share a position.
            offset = f"{0.02 * finger:.3f} 0 0.03" if link == 0 else "0 0 0.03"
            parts += [
                f'  <link name="kp{slot}"/>',
                f'  <joint name="j_{finger}_{link}" type="revolute">',
                f'    <parent link="{parent}"/>',
                f'    <child link="kp{slot}"/>',
                f'    <origin xyz="{offset}" rpy="0 0 0"/>',
                '    <axis xyz="1 0 0"/>',
                '    <limit lower="-1.5" upper="1.5" effort="1" velocity="1"/>',
                "  </joint>",
            ]
            parent = f"kp{slot}"
    parts.append("</robot>")
    return "\n".join(parts)


@pytest.fixture(scope="session")
def hand_urdf(tmp_path_factory) -> str:
    """Write the generated hand URDF and return its path."""
    path = tmp_path_factory.mktemp("registry") / "test_hand.urdf"
    path.write_text(_hand_urdf())
    return str(path)


@pytest.fixture(scope="session")
def hand_spec(hand_urdf):
    """Return the registry entry for the generated hand."""
    return _parse_end_effector(
        "test_hand_v1",
        {
            "class": "dexterous_hand",
            "dof": HAND_DOF,
            "action_space": "keypoints",
            "keypoints": {"topology": "mano21", "valid": "all"},
            "joint_names": HAND_JOINT_NAMES,
            "urdf": hand_urdf,
            "keypoint_links": HAND_KEYPOINT_LINKS,
            "fk_tolerance_m": HAND_TOLERANCE_M,
            "tactile": {"taxels": 30, "channels": 1, "units": "N"},
        },
    )


@pytest.fixture
def dexterous_registry(monkeypatch, hand_spec):
    """Make ``eva_bimanual`` resolve to the generated dexterous specifications.

    The monkeypatch covers every module that imported a registry loader by
    name. The fixture returns the generated platform specification.
    """
    end_effectors = load_end_effectors() | {hand_spec.name: hand_spec}
    platform = _parse_platform(
        "test_dexterous_platform",
        {
            "kind": "robot",
            "embodiment_prefix": "eva",
            "arity": ["bimanual"],
            "arm_dof": 7,
            "aux": {"dof": 7},
            "cameras": ["front_1", "left_wrist", "right_wrist"],
            "reference_frame": "camera:front_1",
            "default_end_effector": hand_spec.name,
            "embodiment_class": "egomimic.rldb.embodiment.eva.Eva",
        },
        end_effectors,
    )
    platforms = {platform.name: platform}
    by_embodiment = dict.fromkeys(platform.embodiments, platform)
    # Each module imported these by name, so rebind them where they are read.
    resolver = "egomimic.rldb.embodiment.embodiment"
    monkeypatch.setattr(f"{resolver}.load_end_effectors", lambda: end_effectors)
    monkeypatch.setattr(f"{resolver}.load_platforms", lambda: platforms)
    for module in (resolver, "egomimic.rldb.zarr.validate"):
        monkeypatch.setattr(
            f"{module}.load_embodiment_platforms", lambda: by_embodiment
        )
    return platform


@pytest.fixture
def hand_track(hand_spec):
    """Return a factory for mutually consistent hand tracks.

    The factory returns ``(joints, keypoints, ee_poses)``. It computes
    root-frame keypoints with forward kinematics, then applies each
    ``ee_poses`` transform so the keypoints and hand-root poses use the same
    episode coordinate frame.
    """
    from egomimic.rldb.embodiment.hand_kinematics import fk_keypoints
    from egomimic.utils.pose_utils import _xyzwxyz_to_matrix

    def build(frames: int = 6, *, joints=None) -> tuple[np.ndarray, ...]:
        rng = np.random.default_rng(0)
        if joints is None:
            joints = rng.uniform(-0.6, 0.6, size=(frames, HAND_DOF))
        angles = np.linspace(0.1, 0.5, frames)
        ee_poses = np.zeros((frames, 7))
        ee_poses[:, 0] = 0.3 + np.arange(frames) * 0.01
        ee_poses[:, 1] = 0.1
        ee_poses[:, 3] = np.cos(angles / 2)
        ee_poses[:, 6] = np.sin(angles / 2)
        world_T_root = _xyzwxyz_to_matrix(ee_poses)
        root_points = fk_keypoints(hand_spec, joints)
        world = np.einsum("tij,tkj->tki", world_T_root[:, :3, :3], root_points)
        world += world_T_root[:, None, :3, 3]
        return joints, world.reshape(frames, -1), ee_poses

    return build
