"""Derive EgoVerse dexterous tracks from Sharpa's 65-column joint vectors.

Sharpa's ``observation.state`` and ``action`` vectors use this layout, in
radians:

=========  ==========================================================
Columns    Contents
=========  ==========================================================
``0:7``    ``left_arm_joint_1`` … ``left_arm_joint_7``
``7:29``   left hand, registry ``sharpa_wave_v1_left.joint_names`` order
``29:36``  ``right_arm_joint_1`` … ``right_arm_joint_7``
``36:58``  right hand, registry ``sharpa_wave_v1_right.joint_names`` order
``58:65``  platform ``aux.joint_names`` (lower body, then neck)
=========  ==========================================================

Forward kinematics over the registry URDFs gives palm poses, MANO-21 hand
keypoints and link poses in ``robot_base`` (the platform URDF root). The
platform's ``<side>_hand_base_link`` leaf is composed directly with the hand
URDF's flange root. This module takes and returns arrays only, so readers and
writers other than the Zarr converter can reuse it.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytorch_kinematics as pk

from egomimic.rldb.embodiment import Embodiment
from egomimic.rldb.embodiment.hand_kinematics import load_chain
from egomimic.rldb.embodiment.urdf import load_urdf
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz

EMBODIMENT = "dexmate_bimanual"
WIDTH = 65
ARM_DOF = 7
HAND_DOF = 22
SIDE_OFFSETS = {"left": 0, "right": ARM_DOF + HAND_DOF}
AUX_OFFSET = 2 * (ARM_DOF + HAND_DOF)
HEAD_LINK = "head_base_link"
#: Margin the validator allows around registry hand-joint limits.
LIMIT_TOLERANCE = 1e-6
#: Frames per forward-kinematics batch; bounds transient memory.
FK_BLOCK = 4096


def arm_joint_names(side: str) -> tuple[str, ...]:
    """Return the platform URDF's arm joint names for one side."""
    return tuple(f"{side}_arm_joint_{j + 1}" for j in range(ARM_DOF))


class SharpaKinematics:
    """Registry models and joint limits for ``dexmate_bimanual``.

    Hand limits come from the registry, which the validator enforces. Arm and
    auxiliary limits come from the platform URDF.
    """

    def __init__(self, resolved=None):
        self.resolved = resolved or Embodiment.resolve(EMBODIMENT)
        path = self.resolved.platform.asset_path(verify=True)
        self.platform = load_urdf(path)
        self.hands = {
            side: load_chain(spec) for side, spec in self.resolved.end_effectors.items()
        }
        robot = pk.URDF.from_xml_string(path.read_bytes())
        urdf_limits = {
            joint.name: (joint.limit.lower, joint.limit.upper)
            for joint in robot.joints
            if joint.type != "fixed" and joint.limit is not None
        }
        # A link is fixed in robot_base when every joint above it is fixed.
        self._parent = {joint.child: joint for joint in robot.joints}
        names, limits, sources = [], [], []
        for side in ("left", "right"):
            spec = self.resolved.end_effectors[side]
            for name in arm_joint_names(side):
                names.append(name)
                limits.append(urdf_limits[name])
                sources.append(f"urdf:{self.resolved.platform.name}")
            names.extend(spec.joint_names)
            limits.extend(spec.joint_limits)
            sources.extend([f"registry:{spec.name}"] * len(spec.joint_names))
        for name in self.resolved.platform.aux.joint_names:
            names.append(name)
            limits.append(urdf_limits[name])
            sources.append(f"urdf:{self.resolved.platform.name}")
        if len(names) != WIDTH:
            raise ValueError(
                f"expected {WIDTH} Sharpa columns, registry gives {len(names)}"
            )
        self.column_names = tuple(names)
        self.limits = np.asarray(limits, dtype=np.float64)
        self.limit_sources = tuple(sources)

    def is_fixed_in_base(self, link: str) -> bool:
        """Return whether ``link`` does not move relative to ``robot_base``."""
        if link not in self.platform.links:
            return False
        while link in self._parent:
            joint = self._parent[link]
            if joint.type != "fixed":
                return False
            link = joint.parent
        return True

    @property
    def links(self) -> frozenset[str]:
        """Return every platform and hand link that ``link_poses`` accepts."""
        return self.platform.links.union(*(c.links for c in self.hands.values()))

    def array_key(self, column: int, prefix: str) -> str:
        """Return the canonical array key that stores one Sharpa column."""
        if column >= AUX_OFFSET:
            return f"{prefix}_aux_joints"
        side = "left" if column < SIDE_OFFSETS["right"] else "right"
        local = column - SIDE_OFFSETS[side]
        return f"{side}.{prefix}_{'joints' if local < ARM_DOF else 'hand_joints'}"


def check_raw(raw, name: str = "joint vector") -> np.ndarray:
    """Return ``raw`` as an array after checking it is a finite ``(T, 65)``."""
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != WIDTH or not len(raw):
        raise ValueError(f"{name} must have shape (T, {WIDTH}), got {raw.shape}")
    if not np.isfinite(raw).all():
        rows = np.unique(np.nonzero(~np.isfinite(raw))[0])
        raise ValueError(f"{name} has non-finite values on {len(rows)} frame(s)")
    return raw


def split_columns(raw, prefix: str) -> dict[str, np.ndarray]:
    """Map one ``(T, 65)`` vector onto canonical joint arrays, keeping dtype."""
    raw = check_raw(raw)
    out = {f"{prefix}_aux_joints": np.ascontiguousarray(raw[:, AUX_OFFSET:WIDTH])}
    for side, offset in SIDE_OFFSETS.items():
        out[f"{side}.{prefix}_joints"] = np.ascontiguousarray(
            raw[:, offset : offset + ARM_DOF]
        )
        out[f"{side}.{prefix}_hand_joints"] = np.ascontiguousarray(
            raw[:, offset + ARM_DOF : offset + ARM_DOF + HAND_DOF]
        )
    return out


def _forward(kin: SharpaKinematics, values: np.ndarray):
    """Return platform and per-hand link tracks for a block of float64 rows."""
    frames = len(values)
    named = dict(zip(kin.column_names, values.T, strict=True))
    platform = kin.platform.batch_link_transforms(named, frames)
    hands = {
        side: chain.batch_link_transforms(named, frames)
        for side, chain in kin.hands.items()
    }
    return platform, hands


def _compose(link, platform, hands):
    """Return ``robot_base_T_link`` for a platform link or a hand link."""
    if link in platform:
        return platform[link]
    for side, poses in hands.items():
        if link in poses:
            return platform[f"{side}_hand_base_link"] @ poses[link]
    raise KeyError(link)


def derive_tracks(
    observed,
    commanded,
    *,
    kinematics: SharpaKinematics | None = None,
    links: Iterable[str] = (HEAD_LINK,),
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Derive canonical arrays from observed and commanded 65-column vectors.

    Args:
        observed: ``(T, 65)`` observed joint vector (``observation.state``).
        commanded: ``(T, 65)`` commanded joint vector (``action``).
        kinematics: Loaded models; ``None`` loads the registry models.
        links: Platform or hand links whose observed ``robot_base_T_link``
            tracks to return, such as camera mount parents.

    Returns:
        ``(tracks, link_poses)``. ``tracks`` holds ``{side}.{obs,cmd}_joints``,
        ``{side}.{obs,cmd}_hand_joints`` and ``{obs,cmd}_aux_joints`` in the
        input dtype; float64 ``{side}.{obs,cmd}_ee_pose`` palm poses
        ``[x, y, z, qw, qx, qy, qz]``; and float64 ``{side}.obs_hand_keypoints``
        ``(T, 63)``. ``link_poses`` maps each requested link to ``(T, 4, 4)``.
        Poses and keypoints are in ``robot_base``. FK uses the values exactly
        as given, after a float64 cast.

    Raises:
        ValueError: If an input is not a finite ``(T, 65)`` array, the inputs
            differ in length, or a requested link is unknown.
    """
    kin = kinematics or SharpaKinematics()
    observed = check_raw(observed, "observed joint vector")
    commanded = check_raw(commanded, "commanded joint vector")
    if len(observed) != len(commanded):
        raise ValueError(
            f"observed ({len(observed)}) and commanded ({len(commanded)}) differ in length"
        )
    links = list(dict.fromkeys(links))
    unknown = sorted(set(links) - kin.links)
    if unknown:
        raise ValueError(
            f"unknown link(s) {unknown}; not in the platform or hand URDFs"
        )
    total = len(observed)
    tracks = {**split_columns(observed, "obs"), **split_columns(commanded, "cmd")}
    specs = kin.resolved.end_effectors
    for side, spec in specs.items():
        for prefix in ("obs", "cmd"):
            tracks[f"{side}.{prefix}_ee_pose"] = np.empty((total, 7))
        tracks[f"{side}.obs_hand_keypoints"] = np.zeros(
            (total, 3 * spec.keypoints.n_slots)
        )
    link_poses = {link: np.empty((total, 4, 4)) for link in links}
    for start in range(0, total, FK_BLOCK):
        block = slice(start, min(total, start + FK_BLOCK))
        for prefix, raw in (("obs", observed), ("cmd", commanded)):
            platform, hands = _forward(kin, np.asarray(raw[block], dtype=np.float64))
            for side, spec in specs.items():
                base_T_flange = platform[f"{side}_hand_base_link"]
                palm = base_T_flange @ hands[side][spec.ee_pose_link]
                tracks[f"{side}.{prefix}_ee_pose"][block] = _matrix_to_xyzwxyz(palm)
                if prefix != "obs":
                    continue
                points = tracks[f"{side}.obs_hand_keypoints"][block]
                points = points.reshape(len(palm), spec.keypoints.n_slots, 3)
                for slot, link in spec.keypoint_links:
                    points[:, slot] = (base_T_flange @ hands[side][link])[:, :3, 3]
            if prefix == "obs":
                for link in links:
                    link_poses[link][block] = _compose(link, platform, hands)
    return tracks, link_poses


def joint_limit_excursions(
    raw,
    prefix: str,
    kinematics: SharpaKinematics,
    tolerance: float = LIMIT_TOLERANCE,
) -> list[dict]:
    """Summarize values outside each column's joint limits.

    Args:
        raw: A ``(T, 65)`` joint vector.
        prefix: ``obs`` or ``cmd``; selects the reported array keys.
        kinematics: Loaded models supplying the limits.
        tolerance: Margin outside the limits that is not counted.

    Returns:
        One record per column with an excursion, in column order.
        ``validator_checked`` marks hand columns, which the validator rejects.
    """
    values = check_raw(raw).astype(np.float64)
    lower = kinematics.limits[:, 0] - tolerance
    upper = kinematics.limits[:, 1] + tolerance
    below, above = values < lower, values > upper
    records = []
    for column in np.nonzero((below | above).any(axis=0))[0]:
        column = int(column)
        key = kinematics.array_key(column, prefix)
        under = kinematics.limits[column, 0] - values[below[:, column], column]
        over = values[above[:, column], column] - kinematics.limits[column, 1]
        records.append(
            {
                "array": key,
                "joint": kinematics.column_names[column],
                "source_column": column,
                "limit_source": kinematics.limit_sources[column],
                "lower": float(kinematics.limits[column, 0]),
                "upper": float(kinematics.limits[column, 1]),
                "frames": int((below[:, column] | above[:, column]).sum()),
                "frames_below": int(below[:, column].sum()),
                "frames_above": int(above[:, column].sum()),
                "max_below": float(under.max()) if len(under) else 0.0,
                "max_above": float(over.max()) if len(over) else 0.0,
                "validator_checked": key.endswith("hand_joints"),
            }
        )
    return records


def clip_to_limits(raw, kinematics: SharpaKinematics) -> np.ndarray:
    """Return ``raw`` clipped to each column's joint limits, keeping dtype."""
    raw = check_raw(raw)
    return np.clip(raw, kinematics.limits[:, 0], kinematics.limits[:, 1]).astype(
        raw.dtype
    )


__all__ = [
    "HEAD_LINK",
    "LIMIT_TOLERANCE",
    "SharpaKinematics",
    "arm_joint_names",
    "check_raw",
    "clip_to_limits",
    "derive_tracks",
    "joint_limit_excursions",
    "split_columns",
]
