"""Read a URDF and compute link poses from joint values.

An end-effector registry entry may declare a ``urdf:`` file and a
``keypoint_links:`` mapping. Together they define the transform from the joint
vector a controller consumed to the keypoint positions a vendor shipped, which
is what makes the two representations checkable against each other.

This module reads the kinematic tree only: joint types, axes, origins, and
mimic couplings. It loads no meshes, no inertias, and no collision geometry, so
a vendor URDF that references mesh files it did not ship still resolves. The
declared ``pytorch-kinematics`` and ``mujoco`` dependencies both carry a heavier
runtime than an ingest-time check needs, and ``MinkKinematicsSolver`` raises
``NotImplementedError`` for a URDF path today.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Joint types whose value this module applies. Every other movable type is
#: rejected rather than silently treated as fixed.
_MOVABLE = frozenset({"revolute", "continuous", "prismatic"})
_SUPPORTED = _MOVABLE | {"fixed"}


class UrdfError(ValueError):
    """Report a URDF this module cannot read or evaluate."""


def _floats(text: str | None, default: Sequence[float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    values = [float(v) for v in text.split()]
    if len(values) != len(default):
        raise UrdfError(f"expected {len(default)} numbers, got {text!r}")
    return np.asarray(values, dtype=float)


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the URDF fixed-axis rotation ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _origin(element: ET.Element | None) -> np.ndarray:
    """Return the 4x4 transform declared by an ``<origin>`` child."""
    transform = np.eye(4)
    child = None if element is None else element.find("origin")
    if child is None:
        return transform
    xyz = _floats(child.get("xyz"), (0.0, 0.0, 0.0))
    rpy = _floats(child.get("rpy"), (0.0, 0.0, 0.0))
    transform[:3, :3] = _rpy_to_matrix(*rpy)
    transform[:3, 3] = xyz
    return transform


def _axis_transform(joint: "UrdfJoint", value: float) -> np.ndarray:
    """Return the transform a joint applies at ``value``."""
    transform = np.eye(4)
    if joint.joint_type == "prismatic":
        transform[:3, 3] = joint.axis * value
        return transform
    # Rodrigues' rotation about the joint axis.
    x, y, z = joint.axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    transform[:3, :3] = (
        np.eye(3) + math.sin(value) * skew + (1.0 - math.cos(value)) * (skew @ skew)
    )
    return transform


@dataclass(frozen=True)
class UrdfJoint:
    """One joint of the kinematic tree.

    Attributes:
        name: The URDF joint name.
        joint_type: ``revolute``, ``continuous``, ``prismatic``, or ``fixed``.
        parent: The parent link name.
        child: The child link name.
        origin: The 4x4 parent-to-joint transform at value zero.
        axis: The unit joint axis in the joint frame.
        mimic: ``(source joint, multiplier, offset)``, or ``None``.
    """

    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    mimic: tuple[str, float, float] | None = None

    @property
    def is_actuated(self) -> bool:
        """Return whether this joint takes a value of its own."""
        return self.joint_type in _MOVABLE and self.mimic is None


@dataclass(frozen=True)
class UrdfChain:
    """A URDF kinematic tree evaluated by ``link_transforms``.

    Attributes:
        root: The link that is no joint's child.
        joints: Every joint, in URDF document order.
        links: Every link name declared or referenced by a joint.
    """

    root: str
    joints: tuple[UrdfJoint, ...]
    links: frozenset[str]

    @property
    def actuated_joint_names(self) -> tuple[str, ...]:
        """Return the names of joints that take a value, in document order."""
        return tuple(j.name for j in self.joints if j.is_actuated)

    def link_transforms(
        self, joint_values: Mapping[str, float]
    ) -> dict[str, np.ndarray]:
        """Return every link's 4x4 pose in the root frame.

        Args:
            joint_values: Values in radians or metres for actuated joints. An
                omitted joint takes the value zero.

        Returns:
            A mapping from link name to its 4x4 pose in the root link frame.

        Raises:
            UrdfError: If a mimic joint names a source that does not exist.
        """
        values = {name: 0.0 for name in self.actuated_joint_names}
        for name, value in joint_values.items():
            values[name] = float(value)
        for joint in self.joints:
            if joint.mimic is None:
                continue
            source, multiplier, offset = joint.mimic
            if source not in values:
                raise UrdfError(
                    f"joint {joint.name!r} mimics {source!r}, which is not an "
                    "actuated joint of this URDF"
                )
            values[joint.name] = multiplier * values[source] + offset

        by_parent: dict[str, list[UrdfJoint]] = {}
        for joint in self.joints:
            by_parent.setdefault(joint.parent, []).append(joint)

        poses = {self.root: np.eye(4)}
        frontier = [self.root]
        while frontier:
            parent = frontier.pop()
            for joint in by_parent.get(parent, ()):
                local = joint.origin
                if joint.joint_type in _MOVABLE:
                    local = local @ _axis_transform(joint, values[joint.name])
                poses[joint.child] = poses[parent] @ local
                frontier.append(joint.child)
        return poses

    def link_positions(
        self, joint_values: Mapping[str, float], links: Sequence[str]
    ) -> np.ndarray:
        """Return the root-frame XYZ of each named link.

        Args:
            joint_values: Values for actuated joints, as in ``link_transforms``.
            links: Link names to read, in output order.

        Returns:
            An ``(len(links), 3)`` array of root-frame positions.

        Raises:
            UrdfError: If a name is not a link of this URDF.
        """
        poses = self.link_transforms(joint_values)
        missing = sorted({name for name in links if name not in poses})
        if missing:
            raise UrdfError(
                f"link(s) {missing} are not in this URDF; known: {sorted(poses)}"
            )
        return np.stack([poses[name][:3, 3] for name in links])


def _parse_joint(element: ET.Element) -> UrdfJoint:
    name = element.get("name")
    if not name:
        raise UrdfError("a <joint> element has no name")
    joint_type = element.get("type", "fixed")
    if joint_type not in _SUPPORTED:
        raise UrdfError(
            f"joint {name!r} has unsupported type {joint_type!r}; expected one of "
            f"{sorted(_SUPPORTED)}"
        )
    parent = element.find("parent")
    child = element.find("child")
    if parent is None or child is None:
        raise UrdfError(f"joint {name!r} declares no parent or no child link")

    axis_element = element.find("axis")
    axis = _floats(
        None if axis_element is None else axis_element.get("xyz"), (1.0, 0.0, 0.0)
    )
    norm = float(np.linalg.norm(axis))
    if joint_type in _MOVABLE and norm == 0.0:
        raise UrdfError(f"joint {name!r} declares a zero axis")
    axis = axis / norm if norm else axis

    mimic_element = element.find("mimic")
    mimic = None
    if mimic_element is not None:
        source = mimic_element.get("joint")
        if not source:
            raise UrdfError(f"joint {name!r} has a <mimic> with no `joint` attribute")
        mimic = (
            source,
            float(mimic_element.get("multiplier", 1.0)),
            float(mimic_element.get("offset", 0.0)),
        )

    return UrdfJoint(
        name=name,
        joint_type=joint_type,
        parent=parent.get("link", ""),
        child=child.get("link", ""),
        origin=_origin(element),
        axis=axis,
        mimic=mimic,
    )


def load_urdf(path: str | Path) -> UrdfChain:
    """Read a URDF file into an evaluable kinematic tree.

    Args:
        path: The ``.urdf`` file.

    Returns:
        The parsed chain.

    Raises:
        UrdfError: If the file is unreadable, is not a single-rooted tree, or
            contains a joint type this module does not evaluate.
    """
    path = Path(path)
    try:
        root_element = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise UrdfError(f"cannot read URDF {path}: {exc}") from exc

    joints = tuple(_parse_joint(e) for e in root_element.findall("joint"))
    names = [j.name for j in joints]
    if len(set(names)) != len(names):
        raise UrdfError(f"{path}: duplicate joint name(s) in the URDF")

    links = {e.get("name", "") for e in root_element.findall("link")}
    links |= {j.parent for j in joints} | {j.child for j in joints}
    links.discard("")

    children = {j.child for j in joints}
    roots = sorted(links - children)
    if len(roots) != 1:
        raise UrdfError(
            f"{path}: expected exactly one root link, found {roots or 'none'}"
        )
    if len(children) != len(joints):
        raise UrdfError(f"{path}: a link is the child of more than one joint")

    chain = UrdfChain(root=roots[0], joints=joints, links=frozenset(links))
    unreachable = sorted(links - set(chain.link_transforms({})))
    if unreachable:
        raise UrdfError(
            f"{path}: link(s) {unreachable} are not connected to root "
            f"{chain.root!r}; the joints form a cycle"
        )
    return chain


__all__ = ["UrdfChain", "UrdfError", "UrdfJoint", "load_urdf"]
