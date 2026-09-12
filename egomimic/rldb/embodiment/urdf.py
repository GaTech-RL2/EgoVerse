"""Mesh-free URDF kinematics using pytorch-kinematics.

Joint inputs use URDF document order, radians, and metres. Mimic joints are
derived from their sources; omitted actuated joints default to zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch

_MOVABLE = frozenset({"revolute", "continuous", "prismatic"})
_SUPPORTED = _MOVABLE | {"fixed"}


class UrdfError(ValueError):
    """Report an unreadable, invalid, or unsupported kinematic model."""


@dataclass(frozen=True)
class UrdfChain:
    """A single-rooted tree with actuated joint names in URDF document order."""

    root: str
    links: frozenset[str]
    actuated_joint_names: tuple[str, ...]
    _chain: pk.Chain = field(repr=False)
    _mimics: tuple[tuple[str, str, float, float], ...] = field(repr=False)

    def link_transforms(
        self, joint_values: Mapping[str, float]
    ) -> dict[str, np.ndarray]:
        """Return each link's 4×4 ``root_T_link`` pose at the supplied values."""
        values = {
            name: float(joint_values.get(name, 0.0))
            for name in self.actuated_joint_names
        }
        for name, source, multiplier, offset in self._mimics:
            values[name] = multiplier * values[source] + offset
        angles = torch.tensor(
            [[values[name] for name in self._chain.get_joint_parameter_names()]],
            dtype=torch.float64,
        )
        return {
            name: transform.get_matrix()[0].numpy()
            for name, transform in self._chain.forward_kinematics(angles).items()
        }

    def link_positions(
        self, joint_values: Mapping[str, float], links: Sequence[str]
    ) -> np.ndarray:
        """Return root-frame XYZ positions in the requested link order."""
        missing = sorted(set(links) - self.links)
        if missing:
            raise UrdfError(
                f"link(s) {missing} are not in this URDF; known: {sorted(self.links)}"
            )
        poses = self.link_transforms(joint_values)
        return np.stack([poses[name][:3, 3] for name in links])


def load_urdf(path: str | Path) -> UrdfChain:
    """Parse and validate a URDF without opening its referenced mesh files.

    Raises ``UrdfError`` for malformed trees, unsupported joint types, invalid
    axes, and unresolved mimic couplings.
    """
    path = Path(path)
    try:
        robot = pk.URDF.from_xml_string(path.read_bytes())
    except Exception as exc:
        raise UrdfError(f"cannot read URDF {path}: {exc}") from exc

    if len(robot.joint_map) != len(robot.joints):
        raise UrdfError(f"{path}: duplicate joint name(s) in the URDF")
    if len(robot.link_map) != len(robot.links):
        raise UrdfError(f"{path}: duplicate link name(s) in the URDF")
    links = frozenset(robot.link_map)
    children = {joint.child for joint in robot.joints}
    roots = sorted(links - children)
    if len(roots) != 1:
        raise UrdfError(
            f"{path}: expected exactly one root link, found {roots or 'none'}"
        )
    if len(children) != len(robot.joints):
        raise UrdfError(f"{path}: a link is the child of more than one joint")
    referenced = children | {joint.parent for joint in robot.joints}
    if referenced - links:
        raise UrdfError(f"{path}: undeclared link(s) {sorted(referenced - links)}")

    reachable = set()
    frontier = [roots[0]]
    while frontier:
        parent = frontier.pop()
        reachable.add(parent)
        frontier.extend(child for _, child in robot.child_map.get(parent, ()))
    if links - reachable:
        raise UrdfError(
            f"{path}: link(s) {sorted(links - reachable)} are not connected to root {roots[0]!r}"
        )

    actuated = tuple(
        joint.name
        for joint in robot.joints
        if joint.type in _MOVABLE and joint.mimic is None
    )
    pending = {
        joint.name: joint.mimic for joint in robot.joints if joint.mimic is not None
    }
    mimics = []
    available = set(actuated)
    while pending:
        resolved = [name for name, mimic in pending.items() if mimic.joint in available]
        if not resolved:
            raise UrdfError(
                f"{path}: unresolved mimic source(s) for joints {sorted(pending)}"
            )
        for name in resolved:
            mimic = pending.pop(name)
            mimics.append(
                (
                    name,
                    mimic.joint,
                    1.0 if mimic.multiplier is None else mimic.multiplier,
                    0.0 if mimic.offset is None else mimic.offset,
                )
            )
            available.add(name)

    frames = {name: pk.Frame(name, link=pk.Link(name)) for name in links}
    for joint in robot.joints:
        if joint.type not in _SUPPORTED:
            raise UrdfError(
                f"joint {joint.name!r} has unsupported type {joint.type!r}; expected one of {sorted(_SUPPORTED)}"
            )
        if joint.mimic is not None and joint.type not in _MOVABLE:
            raise UrdfError(f"fixed joint {joint.name!r} cannot mimic another joint")
        axis = joint.axis if joint.axis is not None else (1.0, 0.0, 0.0)
        if joint.type in _MOVABLE and (
            not np.isfinite(axis).all() or not np.linalg.norm(axis)
        ):
            raise UrdfError(f"joint {joint.name!r} declares a zero or non-finite axis")
        offset = None
        if joint.origin is not None:
            rpy = torch.tensor(joint.origin.rpy, dtype=torch.float64)
            offset = pk.Transform3d(
                rot=pk.quaternion_from_euler(rpy, "sxyz"),
                pos=joint.origin.xyz,
                dtype=torch.float64,
            )
        frames[joint.child].joint = pk.Joint(
            joint.name,
            offset=offset,
            axis=axis,
            dtype=torch.float64,
            joint_type="revolute" if joint.type == "continuous" else joint.type,
        )
        frames[joint.parent].children.append(frames[joint.child])

    return UrdfChain(
        root=roots[0],
        links=links,
        actuated_joint_names=actuated,
        _chain=pk.Chain(frames[roots[0]], dtype=torch.float64),
        _mimics=tuple(mimics),
    )


__all__ = ["UrdfChain", "UrdfError", "load_urdf"]
