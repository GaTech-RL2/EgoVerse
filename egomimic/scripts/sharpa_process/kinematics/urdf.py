"""Mesh-free URDF forward kinematics using pytorch-kinematics.

Only the kinematic tree is read, so a URDF whose mesh files are absent still
loads. Joint inputs are radians (metres for prismatic joints); omitted actuated
joints are zero. Mimic joints are not supported (the Sharpa models have none).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pytorch_kinematics as pk
import torch

_MOVABLE = frozenset({"revolute", "continuous", "prismatic"})
_SUPPORTED = _MOVABLE | {"fixed"}


class UrdfError(ValueError):
    """Report an unreadable, invalid, or unsupported kinematic model."""


@dataclass(frozen=True)
class UrdfChain:
    """A single-rooted tree with actuated joint names in URDF document order.

    ``joint_limits`` maps each actuated joint with a ``<limit>`` to
    ``(lower, upper)``.
    """

    root: str
    links: frozenset[str]
    actuated_joint_names: tuple[str, ...]
    joint_limits: Mapping[str, tuple[float, float]]
    _chain: pk.Chain = field(repr=False)

    def batch_link_transforms(
        self, joint_values: Mapping[str, np.ndarray], frames: int
    ) -> dict[str, np.ndarray]:
        """Return each link's ``root_T_link`` track with shape ``(frames, 4, 4)``.

        Args:
            joint_values: A mapping from actuated joint name to a ``(frames,)``
                track. Omitted actuated joints are zero; other names are ignored.
            frames: The number of configurations.
        """
        values = {
            name: np.broadcast_to(
                np.asarray(joint_values.get(name, 0.0), dtype=np.float64), (frames,)
            )
            for name in self.actuated_joint_names
        }
        names = self._chain.get_joint_parameter_names()
        angles = torch.from_numpy(
            np.stack([values[name] for name in names], axis=-1).reshape(frames, -1)
        )
        # A link behind fixed joints only can carry a batch of one.
        return {
            name: np.broadcast_to(transform.get_matrix().numpy(), (frames, 4, 4))
            for name, transform in self._chain.forward_kinematics(angles).items()
        }


def load_urdf(text: bytes | str, name: str = "URDF") -> UrdfChain:
    """Parse and validate a URDF without opening its referenced mesh files.

    Args:
        text: The URDF document.
        name: Used in error messages.

    Raises:
        UrdfError: For a malformed tree, an unsupported joint type, an invalid
            axis, or a mimic joint.
    """
    try:
        robot = pk.URDF.from_xml_string(text)
    except Exception as exc:
        raise UrdfError(f"cannot read {name}: {exc}") from exc

    if len(robot.joint_map) != len(robot.joints):
        raise UrdfError(f"{name}: duplicate joint name(s) in the URDF")
    if len(robot.link_map) != len(robot.links):
        raise UrdfError(f"{name}: duplicate link name(s) in the URDF")
    links = frozenset(robot.link_map)
    children = {joint.child for joint in robot.joints}
    roots = sorted(links - children)
    if len(roots) != 1:
        raise UrdfError(
            f"{name}: expected exactly one root link, found {roots or 'none'}"
        )
    if len(children) != len(robot.joints):
        raise UrdfError(f"{name}: a link is the child of more than one joint")
    referenced = children | {joint.parent for joint in robot.joints}
    if referenced - links:
        raise UrdfError(f"{name}: undeclared link(s) {sorted(referenced - links)}")

    reachable = set()
    frontier = [roots[0]]
    while frontier:
        parent = frontier.pop()
        reachable.add(parent)
        frontier.extend(child for _, child in robot.child_map.get(parent, ()))
    if links - reachable:
        raise UrdfError(
            f"{name}: link(s) {sorted(links - reachable)} are not connected to "
            f"root {roots[0]!r}"
        )

    frames = {link: pk.Frame(link, link=pk.Link(link)) for link in links}
    for joint in robot.joints:
        if joint.type not in _SUPPORTED:
            raise UrdfError(
                f"joint {joint.name!r} has unsupported type {joint.type!r}; "
                f"expected one of {sorted(_SUPPORTED)}"
            )
        if joint.mimic is not None:
            raise UrdfError(f"{name}: mimic joint {joint.name!r} is not supported")
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

    actuated = tuple(joint.name for joint in robot.joints if joint.type in _MOVABLE)
    limits = {
        joint.name: (float(joint.limit.lower), float(joint.limit.upper))
        for joint in robot.joints
        if joint.type in _MOVABLE and joint.limit is not None
    }
    return UrdfChain(
        root=roots[0],
        links=links,
        actuated_joint_names=actuated,
        joint_limits=limits,
        _chain=pk.Chain(frames[roots[0]], dtype=torch.float64),
    )


__all__ = ["UrdfChain", "UrdfError", "load_urdf"]
