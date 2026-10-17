"""Minimal forward kinematics + wireframe rendering for the ORCA hand (v1 URDF).

Dependency-free (xml.etree + numpy) so it can run inside the training container
and on login nodes. Used by the real_mug validation viz to draw the GT /
predicted hand-joint chunk as a skeleton, either in a canonical side panel or
projected into the camera image when the hand-to-robot mount is known.

Joint order of the recorded 17-D hand vectors (orca_core v1 ``joint_ids``):
    thumb_mcp, thumb_abd, thumb_pip, thumb_dip,
    index_abd, index_mcp, index_pip,
    middle_abd, middle_mcp, middle_pip,
    ring_abd, ring_mcp, ring_pip,
    pinky_abd, pinky_mcp, pinky_pip,
    wrist
which maps 1:1 onto the URDF joints ``right_<name>``.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from functools import lru_cache

import cv2
import numpy as np

ORCA_V1_JOINT_ORDER = [
    "thumb_mcp", "thumb_abd", "thumb_pip", "thumb_dip",
    "index_abd", "index_mcp", "index_pip",
    "middle_abd", "middle_mcp", "middle_pip",
    "ring_abd", "ring_mcp", "ring_pip",
    "pinky_abd", "pinky_mcp", "pinky_pip",
    "wrist",
]
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
# Sign of each recorded joint relative to the URDF joint axis, per hand. The
# right hand matches orcahand_right.urdf as recorded. For the left hand the
# recorded wrist angle is opposite to orcahand_left.urdf's wrist axis (its
# motor map has `wrist: -17` vs `17` on the right); determined 2026-10-17 by
# overlaying the posed meshes on the v2 sanity_check demos
# (eth_clariden/mug/calib/left_hand_candidates.py).
JOINT_SIGNS = {
    "right": np.ones(17),
    "left": np.array([1] * 16 + [-1], dtype=float),
}
_ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "orcahand")
DEFAULT_URDF = os.path.join(_ASSET_DIR, "orcahand_right.urdf")
DEFAULT_URDF_LEFT = os.path.join(_ASSET_DIR, "orcahand_left.urdf")


def _rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _axis_angle(axis, th):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class OrcaHandFK:
    """FK over the URDF joint tree. ``side`` selects the ``right_``/``left_`` prefix."""

    def __init__(self, urdf_path: str | None = None, side: str = "right"):
        if urdf_path is None:
            urdf_path = os.environ.get("ORCA_URDF_LEFT", DEFAULT_URDF_LEFT) if side == "left" else os.environ.get("ORCA_URDF", DEFAULT_URDF)
        root = ET.parse(urdf_path).getroot()
        self.side = side
        self.joints = {}        # child_link -> (parent_link, origin T (4x4), type, axis, joint_name)
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            T = np.eye(4); T[:3, :3] = _rpy_to_mat(*rpy); T[:3, 3] = xyz
            a = j.find("axis")
            axis = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else np.array([1.0, 0, 0])
            self.joints[j.find("child").get("link")] = (j.find("parent").get("link"), T, j.get("type"), axis, j.get("name"))
        self.q_names = [f"{side}_{n}" for n in ORCA_V1_JOINT_ORDER]
        # keypoints: tower base (``{side}_tower``: in the stand-alone hand file
        # it sits 6 cm from the ``world`` root, in the combined FR3 files it is
        # the mount's child), wrist joint, palm, then per finger the abd/mcp
        # joint body, pip joint body, fingertip.
        self.keypoint_links = [f"{side}_tower", f"{side}_wrist_jointbody", f"{side}_palm"]
        self.edges = [(0, 1), (1, 2)]
        # "tower" = tower base -> wrist joint (the long segment along the arm),
        # "wrist" = wrist joint -> palm. Both can be hidden via draw_wireframe(skip=...).
        self.finger_of_edge = ["tower", "wrist"]
        palm_idx = 2
        for f in FINGERS:
            if f == "thumb":
                chain = [f"{side}_thumb_mcp_jointbody", f"{side}_thumb_pip_jointbody", f"{side}_thumb_dip_jointbody", f"{side}_thumb_fingertip"]
            else:
                chain = [f"{side}_{f}_abd_jointbody", f"{side}_{f}_pip_jointbody", f"{side}_{f}_fingertip"]
            prev = palm_idx
            for link in chain:
                self.keypoint_links.append(link)
                self.edges.append((prev, len(self.keypoint_links) - 1))
                self.finger_of_edge.append(f)
                prev = len(self.keypoint_links) - 1

    def link_poses(self, q17) -> dict[str, np.ndarray]:
        """All link frames (4x4) in the URDF root frame.

        q17: 17-D hand joint vector (orca_core v1 order) or a dict
        {joint_name: value} covering any revolute joints of the tree (used by
        FR3OrcaFK to drive the arm + hand together)."""
        qmap = dict(q17) if isinstance(q17, dict) else dict(zip(self.q_names, np.asarray(q17, float) * JOINT_SIGNS[self.side]))
        children = set(self.joints)
        roots = {p for p, *_ in self.joints.values()} - children
        poses = {r: np.eye(4) for r in roots}

        def pose(link):
            if link in poses:
                return poses[link]
            parent, T, typ, axis, name = self.joints[link]
            Tp = pose(parent) @ T
            if typ in ("revolute", "continuous") and name in qmap:
                Tj = np.eye(4); Tj[:3, :3] = _axis_angle(axis, qmap[name]); Tp = Tp @ Tj
            poses[link] = Tp
            return Tp

        for link in self.joints:
            pose(link)
        return poses

    def keypoints(self, q17: np.ndarray, T_ref_world: np.ndarray | None = None) -> np.ndarray:
        """(K, 3) keypoint positions; optionally re-expressed in another frame."""
        poses = self.link_poses(q17)
        P = np.stack([poses[l][:3, 3] for l in self.keypoint_links])
        if T_ref_world is not None:
            P = (T_ref_world[:3, :3] @ P.T).T + T_ref_world[:3, 3]
        return P


# Hand mounts: pose of the URDF ``world`` (tower base) frame relative to a robot
# TCP frame, T_tcp_hand. "mug_tcp": Franka Panda + ORCA right hand on the
# real_mug rig, TCP = Franka GUI flange->EE (0.13, 0, 0.07) with identity
# rotation. Exact value from cvg-dex-teleop's combined URDF
# (assets/orcahand/fr3_mount_orcahand.urdf, MOUNT_ASSEMBLY_NOTES.md):
#   T_tcp_world = inv(T_flange_tcp) @ T(mount_joint) @ T(mount_to_hand) @ inv(T_world_tower)
# with mount_joint = fr3_link8 -> mount, mount_to_hand = mount -> right_tower,
# and T_world_tower = (0.04, 0, 0.04575) from orcahand_right.urdf. The TCP sits
# 2.5 cm above the palm centre along the fingers. Verified visually on held-out
# episodes (2026-10-07).
HAND_MOUNTS = {
    "mug_tcp": np.array([[-0.009331, 0.009791, 0.999909, -0.213471],
                         [0.999956, 0.000179, 0.00933, -0.081349],
                         [-0.000087, 0.999952, -0.009792, 0.001015],
                         [0.0, 0.0, 0.0, 1.0]]),
    # Left hand on the bimanual v2 rig: mirrored Gavin mount (notes: mount_joint
    # left + mount_to_hand left) and orcahand_left.urdf's world->left_tower
    # offset (-0.04, 0, 0.04575). Comes out as the y-mirror of the right mount.
    "mug_tcp_left": np.array([[0.009332, 0.009791, 0.999909, -0.213471],
                              [0.999956, -0.000178, -0.009330, 0.081349],
                              [0.000087, 0.999952, -0.009792, 0.001015],
                              [0.0, 0.0, 0.0, 1.0]]),
}
# Convention used by Eve.draw_hand_chunk: the mount for the left hand is the
# right mount's key + "_left" when such an entry exists.


def mount_for(side: str, key: str) -> np.ndarray:
    if side == "left" and f"{key}_left" in HAND_MOUNTS:
        return HAND_MOUNTS[f"{key}_left"]
    return HAND_MOUNTS[key]

_FINGER_RGB = {"tower": (235, 235, 235), "wrist": (235, 235, 235), "thumb": (255, 170, 40), "index": (80, 200, 255), "middle": (160, 120, 255), "ring": (255, 110, 180), "pinky": (120, 255, 120)}


def draw_wireframe(image: np.ndarray, pts2d: np.ndarray, edges, finger_of_edge, color=None, thickness=2, radius=3, skip=()):
    """Draw a hand skeleton; ``color`` overrides the per-finger palette (e.g. GT green / pred red).
    ``skip``: edge groups to hide, e.g. ("tower",) hides the tower-base -> wrist segment that
    runs along the forearm; keypoints only used by hidden edges are hidden too."""
    out = image
    H, W = out.shape[:2]
    def ok(p):
        return np.isfinite(p).all() and -W < p[0] < 2 * W and -H < p[1] < 2 * H
    used = set()
    for (a, b), f in zip(edges, finger_of_edge):
        if f in skip:
            continue
        used.update((a, b))
        pa, pb = pts2d[a], pts2d[b]
        if ok(pa) and ok(pb):
            cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color or _FINGER_RGB[f], thickness, cv2.LINE_AA)
    for i, p in enumerate(pts2d):
        if i in used and ok(p):
            cv2.circle(out, (int(p[0]), int(p[1])), radius, color or (255, 255, 255), -1, cv2.LINE_AA)
    return out


DEFAULT_FR3_URDF = {"right": os.path.join(_ASSET_DIR, "fr3_mount_orcahand.urdf"),
                    "left": os.path.join(_ASSET_DIR, "fr3_mount_orcahand_left.urdf")}
FR3_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
FR3_ARM_LINKS = ["fr3_link0", "fr3_link1", "fr3_link2", "fr3_link3", "fr3_link4", "fr3_link5", "fr3_link6", "fr3_link7", "fr3_link8"]


class FR3OrcaFK(OrcaHandFK):
    """FK through cvg-dex-teleop's combined FR3 + mount + ORCA URDF
    (fr3_mount_orcahand{,_left}.urdf), driven by the recorded 7 arm joints and
    17 hand joints. Frames are in the FR3 base (``base`` link) of that arm.
    Independent of the hand-coded Panda DH + TCP + HAND_MOUNTS path."""

    def __init__(self, side: str = "right", urdf_path: str | None = None):
        super().__init__(urdf_path or os.environ.get(f"FR3_ORCA_URDF_{side.upper()}", DEFAULT_FR3_URDF[side]), side=side)

    def poses(self, q_arm7, q_hand17) -> dict[str, np.ndarray]:
        qd = dict(zip(FR3_JOINTS, np.asarray(q_arm7, float)[:7]))
        qd.update(zip(self.q_names, np.asarray(q_hand17, float) * JOINT_SIGNS[self.side]))
        return self.link_poses(qd)

    def fk(self, q_arm7, q_hand17):
        """Returns (hand keypoints (K,3), arm link origins (9,3), T_base_link8) in the arm base frame."""
        poses = self.poses(q_arm7, q_hand17)
        hand = np.stack([poses[l][:3, 3] for l in self.keypoint_links])
        arm = np.stack([poses[l][:3, 3] for l in FR3_ARM_LINKS])
        return hand, arm, poses["fr3_link8"]


@lru_cache(maxsize=2)
def get_fr3_fk(side: str = "right") -> FR3OrcaFK:
    return FR3OrcaFK(side=side)


@lru_cache(maxsize=2)
def get_fk(side: str = "right") -> OrcaHandFK:
    # Resolved at call time so tests / scripts can point the ORCA_URDF /
    # ORCA_URDF_LEFT env vars at another copy of the description.
    return OrcaHandFK(side=side)


def canonical_view_panel(q_gt: np.ndarray, q_pred: np.ndarray | None, size=(240, 240), steps=(0, 33, 66, 99)) -> np.ndarray:
    """Render GT (green) and predicted (red) hand skeletons for a few timesteps of a
    (T, 17) chunk, side by side in a fixed palm-facing orthographic view. No camera or
    mount calibration needed — this is for judging predicted finger motion."""
    fk = get_fk()
    H, W = size
    n = len(steps)
    panel = np.full((H, W * n, 3), 24, np.uint8)
    # orthographic view down the palm normal: URDF palm frame x-y plane, scaled to the tile
    scale = (min(H, W) * 0.8) / 0.17
    for k, t in enumerate(steps):
        tile = panel[:, k * W:(k + 1) * W]
        for q, color in ((q_pred, (220, 40, 40)), (q_gt, (40, 220, 40))):
            if q is None:
                continue
            P = fk.keypoints(q[min(t, len(q) - 1)])
            # express in palm frame so the wrist angle doesn't swing the whole hand
            poses = fk.link_poses(q[min(t, len(q) - 1)])
            Tp = np.linalg.inv(poses[f"{fk.side}_palm"])
            P = (Tp[:3, :3] @ P.T).T + Tp[:3, 3]
            pts = np.stack([W / 2 + P[:, 0] * scale, H * 0.7 - P[:, 2] * scale], 1)
            draw_wireframe(tile, pts, fk.edges, fk.finger_of_edge, color=color, thickness=1, radius=2)
        cv2.putText(tile, f"t+{t}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    return panel
