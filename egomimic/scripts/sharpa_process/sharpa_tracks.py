"""Derive dexmate_bimanual tracks from Sharpa's 65-column joint vectors.

Sharpa's ``observation.state`` and ``action`` vectors use this layout, in
radians:

=========  ==========================================================
Columns    Contents
=========  ==========================================================
``0:7``    ``left_arm_joint_1`` … ``left_arm_joint_7``
``7:29``   left hand, ``dexmate_rules.HAND_JOINT_NAMES["left"]`` order
``29:36``  ``right_arm_joint_1`` … ``right_arm_joint_7``
``36:58``  right hand, ``dexmate_rules.HAND_JOINT_NAMES["right"]`` order
``58:65``  ``dexmate_rules.AUX_JOINT_NAMES`` (lower body, then neck)
=========  ==========================================================

Forward kinematics over the URDFs in ``kinematics/`` gives palm poses, MANO-21
hand keypoints and link poses in ``robot_base`` (the platform URDF root). The
platform's ``<side>_hand_base_link`` leaf is composed directly with the hand
URDF's flange root. This module takes and returns arrays only.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from egomimic.rldb.zarr import dexmate_rules
from egomimic.scripts.sharpa_process.kinematics import SIDES, load_chain, load_spec
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz, _xyzwxyz_to_matrix

EMBODIMENT = dexmate_rules.EMBODIMENT
ARM_DOF = dexmate_rules.ARM_DOF
HAND_DOF = dexmate_rules.HAND_DOF
WIDTH = 2 * (ARM_DOF + HAND_DOF) + dexmate_rules.AUX_DOF
SIDE_OFFSETS = {"left": 0, "right": ARM_DOF + HAND_DOF}
AUX_OFFSET = 2 * (ARM_DOF + HAND_DOF)
#: Margin below which a value outside its limits is not counted as an excursion.
LIMIT_TOLERANCE = 1e-6
#: Frames per forward-kinematics batch; bounds transient memory.
FK_BLOCK = 4096
#: Frames the FK residual check samples, evenly spaced, first and last included.
FK_SAMPLE_FRAMES = 64


class SharpaKinematics:
    """URDF models, column names and joint limits for ``dexmate_bimanual``.

    Hand limits are ``dexmate_rules.HAND_JOINT_LIMITS``, which the validator
    enforces (equal to the hand URDFs). Arm and auxiliary limits come from the
    platform URDF.
    """

    def __init__(self):
        spec = load_spec()
        platform = spec["platform"]
        self.head_link = platform["head_link"]
        self.platform = load_chain(platform["urdf"], platform["urdf_sha256"])
        self.hand_specs = spec["hands"]
        self.hands = {
            side: load_chain(hand["urdf"], hand["urdf_sha256"])
            for side, hand in self.hand_specs.items()
        }
        self.mounts = {
            side: platform["hand_mount_link"].format(side=side) for side in SIDES
        }
        self.aux_joint_names = tuple(platform["aux_joint_names"])
        if self.aux_joint_names != dexmate_rules.AUX_JOINT_NAMES:
            raise ValueError(
                f"spec aux joints {self.aux_joint_names} != dexmate_rules "
                f"{dexmate_rules.AUX_JOINT_NAMES}"
            )
        names, limits, sources = [], [], []
        for side in SIDES:
            for name in dexmate_rules.ARM_JOINT_NAMES[side]:
                names.append(name)
                limits.append(self.platform.joint_limits[name])
                sources.append(f"urdf:{platform['urdf']}")
            names.extend(dexmate_rules.HAND_JOINT_NAMES[side])
            limits.extend(dexmate_rules.HAND_JOINT_LIMITS[side])
            sources.extend(["dexmate_rules.HAND_JOINT_LIMITS"] * HAND_DOF)
        for name in self.aux_joint_names:
            names.append(name)
            limits.append(self.platform.joint_limits[name])
            sources.append(f"urdf:{platform['urdf']}")
        if len(names) != WIDTH:
            raise ValueError(f"expected {WIDTH} Sharpa columns, got {len(names)}")
        self.column_names = tuple(names)
        self.limits = np.asarray(limits, dtype=np.float64)
        self.limit_sources = tuple(sources)

    @property
    def links(self) -> frozenset[str]:
        """Return every platform and hand link that ``derive_tracks`` accepts."""
        return self.platform.links.union(*(c.links for c in self.hands.values()))

    def keypoint_links(self, side: str) -> list[tuple[int, str]]:
        """Return ``(slot, link)`` pairs in slot order."""
        return sorted(self.hand_specs[side]["keypoint_links"].items())

    def array_key(self, column: int, prefix: str) -> str:
        """Return the array key that stores one Sharpa column."""
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
    """Map one ``(T, 65)`` vector onto joint arrays, keeping dtype."""
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


def _compose(link, platform, hands, mounts):
    """Return ``robot_base_T_link`` for a platform link or a hand link."""
    if link in platform:
        return platform[link]
    for side, poses in hands.items():
        if link in poses:
            return platform[mounts[side]] @ poses[link]
    raise KeyError(link)


def derive_tracks(
    observed,
    commanded,
    *,
    kinematics: SharpaKinematics | None = None,
    links: Iterable[str] = (),
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Derive the stored arrays from observed and commanded 65-column vectors.

    Args:
        observed: ``(T, 65)`` observed joint vector (``observation.state``).
        commanded: ``(T, 65)`` commanded joint vector (``action``).
        kinematics: Loaded models; ``None`` loads them.
        links: Platform or hand links whose observed ``robot_base_T_link``
            tracks to return (the head link, the arm TCP links).

    Returns:
        ``(tracks, link_poses)``. ``tracks`` holds ``{side}.{obs,cmd}_joints``,
        ``{side}.{obs,cmd}_hand_joints`` and ``{obs,cmd}_aux_joints`` in the
        input dtype; float64 ``{side}.{obs,cmd}_ee_pose`` palm poses
        ``[x, y, z, qw, qx, qy, qz]``; and float64
        ``{side}.{obs,cmd}_hand_keypoints`` ``(T, 63)``. ``link_poses`` maps
        each requested link to ``(T, 4, 4)``. Poses and keypoints are in
        ``robot_base``. FK uses the values exactly as given, after a float64
        cast.

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
    for side in SIDES:
        for prefix in ("obs", "cmd"):
            tracks[f"{side}.{prefix}_ee_pose"] = np.empty((total, 7))
            tracks[f"{side}.{prefix}_hand_keypoints"] = np.zeros(
                (total, dexmate_rules.KEYPOINT_DIM)
            )
    link_poses = {link: np.empty((total, 4, 4)) for link in links}
    for start in range(0, total, FK_BLOCK):
        block = slice(start, min(total, start + FK_BLOCK))
        for prefix, raw in (("obs", observed), ("cmd", commanded)):
            platform, hands = _forward(kin, np.asarray(raw[block], dtype=np.float64))
            for side in SIDES:
                base_T_flange = platform[kin.mounts[side]]
                palm = base_T_flange @ hands[side][kin.hand_specs[side]["ee_pose_link"]]
                tracks[f"{side}.{prefix}_ee_pose"][block] = _matrix_to_xyzwxyz(palm)
                points = tracks[f"{side}.{prefix}_hand_keypoints"][block]
                points = points.reshape(len(palm), -1, 3)
                for slot, link in kin.keypoint_links(side):
                    points[:, slot] = (base_T_flange @ hands[side][link])[:, :3, 3]
            if prefix == "obs":
                for link in links:
                    link_poses[link][block] = _compose(
                        link, platform, hands, kin.mounts
                    )
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
        ``validator_check`` names what main's validator (``dexmate_rules``)
        reports for the column: ``error`` (an observed hand joint more than
        ``ERROR_MARGIN_RAD`` out), ``warning`` (more than ``WARN_MARGIN_RAD``),
        or ``None``. ``validator_checked`` marks the columns it rejects.
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
        worst = max(
            under.max() if len(under) else 0.0, over.max() if len(over) else 0.0
        )
        check = None
        if key.endswith("hand_joints"):
            if prefix == "obs" and worst > dexmate_rules.ERROR_MARGIN_RAD:
                check = "error"
            elif worst > dexmate_rules.WARN_MARGIN_RAD:
                check = "warning"
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
                "validator_check": check,
                "validator_checked": check == "error",
            }
        )
    return records


def sample_frames(count: int, limit: int = FK_SAMPLE_FRAMES) -> np.ndarray:
    """Up to ``limit`` evenly spaced indices of ``count``, ends included."""
    if limit <= 0 or count <= limit:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, limit).astype(int))


def fk_residuals(
    side: str, joints, keypoints, ee_poses, kinematics: SharpaKinematics
) -> np.ndarray:
    """Distances between stored keypoints and FK of the stored hand joints.

    Both are compared in the palm frame the stored ``ee_poses`` define, so the
    check needs only the hand URDF: a joint-order, unit or handedness slip, or
    keypoints that do not belong to the joints, shows up as one number.

    Args:
        side: ``left`` or ``right``.
        joints: ``(N, 22)`` hand joints in ``HAND_JOINT_NAMES[side]`` order.
        keypoints: ``(N, 63)`` keypoints in the frame of ``ee_poses``.
        ee_poses: ``(N, 7)`` palm poses ``[x, y, z, qw, qx, qy, qz]``.
        kinematics: Loaded models.

    Returns:
        ``(N, 21)`` Euclidean distances in metres, by MANO slot.
    """
    joints = np.asarray(joints, dtype=np.float64)
    keypoints = np.asarray(keypoints, dtype=np.float64)
    ee_poses = np.asarray(ee_poses, dtype=np.float64)
    frames = len(joints)
    if (
        joints.shape != (frames, HAND_DOF)
        or keypoints.shape != (frames, dexmate_rules.KEYPOINT_DIM)
        or ee_poses.shape != (frames, 7)
    ):
        raise ValueError(
            f"{side}: expected (N, {HAND_DOF}), (N, {dexmate_rules.KEYPOINT_DIM}) "
            f"and (N, 7), got {joints.shape}, {keypoints.shape}, {ee_poses.shape}"
        )
    named = dict(zip(dexmate_rules.HAND_JOINT_NAMES[side], joints.T, strict=True))
    poses = kinematics.hands[side].batch_link_transforms(named, frames)
    palm_T_root = np.linalg.inv(poses[kinematics.hand_specs[side]["ee_pose_link"]])
    links = kinematics.keypoint_links(side)
    fk = np.stack([(palm_T_root @ poses[link])[:, :3, 3] for _, link in links], 1)
    palm_T_base = np.linalg.inv(_xyzwxyz_to_matrix(ee_poses))
    stored = keypoints.reshape(frames, -1, 3)[:, [slot for slot, _ in links]]
    stored = np.einsum("tij,tkj->tki", palm_T_base[:, :3, :3], stored)
    stored += palm_T_base[:, None, :3, 3]
    return np.linalg.norm(fk - stored, axis=-1)


#: Sharpa's arm TCP in ``robot_base``, fitted over SharpaDex-v1.0 observed
#: joints: ``R_tcp = R_link7 @ R_tool`` (residual below 0.001 deg) and
#: ``p_tcp = p_link7 + R_link7 @ tool + base_shift`` (5 mm RMS; the rest is
#: lower-body geometry the URDF does not model).
TCP_MODEL = {
    "left": {
        "tool_rotvec_deg": (127.279, 0.0, 127.279),
        "tool_m": (0.0692, 0.0002, -0.0001),
        "base_shift_m": (0.0006, 0.0155, 0.071),
    },
    "right": {
        "tool_rotvec_deg": (0.0, 90.0, 0.0),
        "tool_m": (0.0693, 0.0004, -0.0002),
        "base_shift_m": (0.0012, -0.0169, 0.071),
    },
}
#: Link whose pose the TCP model composes.
TCP_LINK = "{side}_arm_Link_7"
#: Largest accepted median TCP rotation and 95th-percentile position errors.
TCP_ROTATION_TOLERANCE_DEG = 2.0
TCP_POSITION_TOLERANCE_M = 0.05


def tcp_side_check(tcp_pose, link_poses: dict[str, np.ndarray]) -> dict:
    """Compare Sharpa's observed TCP poses with forward kinematics per side.

    Args:
        tcp_pose: ``(T, 12)`` ``observation.state.tcp_pose``: per side
            ``[x, y, z, rotvec]``, left first, in ``robot_base``.
        link_poses: ``robot_base_T_link`` tracks for ``left_arm_Link_7`` and
            ``right_arm_Link_7`` over the same frames.

    Returns:
        ``{"passed": bool, "sides": {...}, "problems": [...]}``. Each side
        records the median rotation error against its own arm and against
        the other arm; a swapped pairing that fits better fails the check.
        Non-finite inputs or metrics fail it before any threshold is
        compared (NaN compares false), and no non-finite metric is returned.
    """
    from scipy.spatial.transform import Rotation

    tcp_pose = np.asarray(tcp_pose, dtype=np.float64)
    if tcp_pose.ndim != 2 or tcp_pose.shape[1] != 12:
        raise ValueError(f"tcp_pose must have shape (T, 12), got {tcp_pose.shape}")
    problems = []
    if not np.isfinite(tcp_pose).all():
        rows = len(np.unique(np.nonzero(~np.isfinite(tcp_pose))[0]))
        problems.append(f"tcp_pose has non-finite values on {rows} frame(s)")
    for side in SIDES:
        link = TCP_LINK.format(side=side)
        if not np.isfinite(np.asarray(link_poses[link], dtype=np.float64)).all():
            problems.append(f"forward kinematics of {link} is non-finite")
    if problems:
        return {"passed": False, "sides": {}, "problems": problems}
    vendor = {
        side: (
            tcp_pose[:, 6 * i : 6 * i + 3],
            Rotation.from_rotvec(tcp_pose[:, 6 * i + 3 : 6 * i + 6]),
        )
        for i, side in enumerate(SIDES)
    }
    predicted = {}
    for side, model in TCP_MODEL.items():
        link = np.asarray(link_poses[TCP_LINK.format(side=side)], dtype=np.float64)
        rotation = Rotation.from_matrix(link[:, :3, :3])
        tool = Rotation.from_rotvec(np.radians(model["tool_rotvec_deg"]))
        position = (
            link[:, :3, 3] + rotation.apply(model["tool_m"]) + model["base_shift_m"]
        )
        predicted[side] = (position, rotation * tool)
    sides = {}
    for side, other in (("left", "right"), ("right", "left")):
        position, rotation = vendor[side]

        def angle(fk_side):
            error = rotation.inv() * predicted[fk_side][1]
            return float(np.degrees(np.median(error.magnitude())))

        own, swapped = angle(side), angle(other)
        offset = np.linalg.norm(position - predicted[side][0], axis=1)
        sides[side] = {
            "rotation_error_deg_median": own,
            "swapped_rotation_error_deg_median": swapped,
            "position_error_m_p95": float(np.percentile(offset, 95)),
        }
        if not all(np.isfinite(v) for v in sides[side].values()):
            problems.append(f"{side} TCP metrics are non-finite")
            sides[side] = {k: None for k in sides[side]}
            continue
        if own > TCP_ROTATION_TOLERANCE_DEG:
            problems.append(
                f"{side} TCP rotation differs from forward kinematics by {own:.2f} deg "
                f"(median; tolerance {TCP_ROTATION_TOLERANCE_DEG:g})"
            )
        if swapped <= own:
            problems.append(
                f"{side} TCP matches the {other} arm better ({swapped:.2f} vs "
                f"{own:.2f} deg): the arm columns look swapped"
            )
        if sides[side]["position_error_m_p95"] > TCP_POSITION_TOLERANCE_M:
            problems.append(
                f"{side} TCP position differs from forward kinematics by "
                f"{sides[side]['position_error_m_p95']:.3f} m (p95; tolerance "
                f"{TCP_POSITION_TOLERANCE_M:g})"
            )
    return {"passed": not problems, "sides": sides, "problems": problems}


__all__ = [
    "SharpaKinematics",
    "TCP_LINK",
    "TCP_MODEL",
    "check_raw",
    "derive_tracks",
    "fk_residuals",
    "joint_limit_excursions",
    "sample_frames",
    "split_columns",
    "tcp_side_check",
]
