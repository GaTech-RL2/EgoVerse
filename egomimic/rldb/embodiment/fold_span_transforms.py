"""Span-safe (per-frame, variable-length) keymaps + transforms for the fold
cross-embodiment cotrain SMOKE.

WHY THIS EXISTS: the canonical ``Eva.get_transform_list('cartesian')`` /
``Aria.get_transform_list('keypoints_headframe_ypr')`` pipelines build a
FIXED-horizon windowed action chunk per obs-frame (``InterpolatePose
chunk_length=100`` / ``Reshape (30,21,3)``). Those assume the WINDOWED padded
MultiDataset reader. The span-as-episode packed reader
(``ZarrAnnotationSpanPackedDataset``) reads a whole variable-length span at
native per-frame resolution, so the fixed reshape/interp blows up
(``cannot reshape array of size ... into (30,21,3)``).

These minimal transforms instead produce PER-FRAME actions + proprio over the
whole span by concatenating the raw per-frame arrays (frame conventions are the
raw quaternion / head-frame-relative-nothing forms Ã¢ÂÂ good enough to verify the
ARCH runs + loss descends, which is the smoke's only goal):

  eva  : actions_cartesian (14) = [left.cmd_ee_pose(7), right.cmd_ee_pose(7)]
         observations.state.ee_pose (14) = [left.obs_ee_pose(7), right.obs_ee_pose(7)]
  human: actions_keypoints (132) = [left wrist_xyz(3), left kp(63),
                                     right wrist_xyz(3), right kp(63)]
         observations.keypoints (132) = same as the action (teacher-forced)
         (was 138 with wrist ypr; dropped -- see HeadFrameWristPos)
         obs_head_pose (7) = raw head pose
"""

import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    ConcatKeys,
    NumpyToTensor,
    Transform,
)


# --------------------------------------------------------------------------- #
# Span-safe PER-FRAME head-frame conversion (fold2 FIX 2).
#
# The canonical Aria.get_transform_list("keypoints_headframe_ypr") brings
# keypoints + wrist into the head's local frame (relative to obs_head_pose),
# but does it via ActionChunkCoordinateFrameTransform, which uses ONE head pose
# for a whole windowed chunk, wrapped in InterpolatePose/Reshape((30,21,3)) that
# assume a FIXED window length -> breaks on variable-length span reads.
#
# The head-frame math is inherently per-frame: frame t's points are expressed
# in frame t's head frame. These ops do EXACTLY that, vectorised over the whole
# span AND robust to the single-frame (D,) reads that the WINDOWED ZarrDataset
# probe (norm_stats.populate_from_datasets) issues. obs_head_pose / wrist_pose
# are xyz + quat(wxyz) (matches _xyzwxyz_to_matrix in the canonical pipeline).
# --------------------------------------------------------------------------- #
def _wxyz_to_matrix(q):
    """(N,4) wxyz quats -> (N,3,3) rotation matrices (scipy wants xyzw)."""
    xyzw = np.concatenate([q[:, 1:4], q[:, 0:1]], axis=-1)
    # Aria arrays are zero-padded past attrs["total_frames"], and an action
    # window near the end of an episode gets clamped short then padded back to
    # the horizon -- either way some rows arrive as [0,0,0,0]. A zero-norm
    # quaternion has NO rotation: scipy raises, and any silent normalize turns
    # numerical dust into a garbage frame. Substitute identity there so the pad
    # is a no-op rotation instead of corrupting the whole chunk.
    n = np.linalg.norm(xyzw, axis=-1)
    bad = n < 1e-8
    if bad.any():
        xyzw = np.asarray(xyzw, dtype=np.float64).copy()
        xyzw[bad] = np.array([0.0, 0.0, 0.0, 1.0])   # identity, xyzw order
    return R.from_quat(xyzw).as_matrix()


class HeadFrameKeypoints(Transform):
    """Per-frame head-frame conversion of a flat keypoint array.

    ``kp`` (T, n_kp*3) [or (n_kp*3,) single frame] -> same shape, each frame's
    keypoints expressed in that frame's head frame:
        kp_head[t,i] = R_head[t]^T @ (kp_world[t,i] - t_head[t])
    """

    def __init__(self, head_key: str, kp_key: str, out_key: str, n_kp: int = 21):
        self.head_key = head_key
        self.kp_key = kp_key
        self.out_key = out_key
        self.n_kp = int(n_kp)

    def transform(self, batch: dict) -> dict:
        head = np.asarray(batch[self.head_key], dtype=np.float64)
        kp = np.asarray(batch[self.kp_key], dtype=np.float64)
        single = head.ndim == 1
        if single:
            head, kp = head[None, :], kp[None, :]
        T = head.shape[0]
        t_head = head[:, :3]
        R_head = _wxyz_to_matrix(head[:, 3:7])             # (T,3,3)
        pts = kp.reshape(T, self.n_kp, 3)
        rel = pts - t_head[:, None, :]                     # (T,n,3)
        kp_head = np.einsum("tij,tnj->tni", np.transpose(R_head, (0, 2, 1)), rel)
        out = kp_head.reshape(T, self.n_kp * 3)
        batch[self.out_key] = out[0] if single else out
        return batch


class HeadFrameWristYPR(Transform):
    """Per-frame head-frame wrist pose -> xyz + ypr (ZYX euler).

    ``wrist`` (T,7) [or (7,)] xyz+quat(wxyz) -> (T,6) [or (6,)] xyz+ypr, each
    frame in that frame's head frame:
        R_wh = R_head^T @ R_wrist ;  t_wh = R_head^T @ (t_wrist - t_head)
    """

    def __init__(self, head_key: str, wrist_key: str, out_key: str):
        self.head_key = head_key
        self.wrist_key = wrist_key
        self.out_key = out_key

    def transform(self, batch: dict) -> dict:
        head = np.asarray(batch[self.head_key], dtype=np.float64)
        wr = np.asarray(batch[self.wrist_key], dtype=np.float64)
        single = head.ndim == 1
        if single:
            head, wr = head[None, :], wr[None, :]
        t_head = head[:, :3]
        R_head = _wxyz_to_matrix(head[:, 3:7])             # (T,3,3)
        t_wr = wr[:, :3]
        R_wr = _wxyz_to_matrix(wr[:, 3:7])                 # (T,3,3)
        R_headT = np.transpose(R_head, (0, 2, 1))
        t_wh = np.einsum("tij,tj->ti", R_headT, t_wr - t_head)      # (T,3)
        R_wh = np.einsum("tij,tjk->tik", R_headT, R_wr)            # (T,3,3)
        ypr = R.from_matrix(R_wh).as_euler("ZYX", degrees=False)   # (T,3)
        out = np.concatenate([t_wh, ypr], axis=-1)                 # (T,6)
        batch[self.out_key] = out[0] if single else out
        return batch


class HeadFrameWristPos(Transform):
    """Head-frame wrist POSITION only -- xyz, no orientation (option B).

    ``wrist`` (T,7) [or (7,)] xyz+quat(wxyz) -> (T,3) [or (3,)]:
        t_wh = R_head^T @ (t_wrist - t_head)

    Deliberately drops the ZYX-euler orientation that HeadFrameWristYPR emits:
    those 3 dims/wrist are discontinuous (wrap at +-pi, gimbal lock at pitch
    +-pi/2) and a diffusion model cannot represent the branch cut, which showed
    up as ~80%% of the human action error concentrated in 12 of 138 dims.
    Orientation remains recoverable from the 21 head-frame keypoints.
    """

    def __init__(self, head_key: str, wrist_key: str, out_key: str):
        self.head_key = head_key
        self.wrist_key = wrist_key
        self.out_key = out_key

    def transform(self, batch: dict) -> dict:
        head = np.asarray(batch[self.head_key], dtype=np.float64)
        wr = np.asarray(batch[self.wrist_key], dtype=np.float64)
        single = head.ndim == 1
        if single:
            head, wr = head[None, :], wr[None, :]
        t_head = head[:, :3]
        R_head = _wxyz_to_matrix(head[:, 3:7])
        R_headT = np.transpose(R_head, (0, 2, 1))
        t_wh = np.einsum("tij,tj->ti", R_headT, wr[:, :3] - t_head)   # (T,3)
        batch[self.out_key] = t_wh[0] if single else t_wh
        return batch


# --------------------------------------------------------------------------- #
# eva_bimanual
# --------------------------------------------------------------------------- #
def _drop_camera_keys(km):
    """norm_mode: drop image/annotation keys so the norm-stats dataset reads only
    the numeric (proprio/action) arrays (mirrors Embodiment.get_keymap)."""
    return {
        k: v
        for k, v in km.items()
        if v.get("key_type") not in (
            "camera_keys", "annotation_keys", "metadata_keys"
        )
    }


def eva_span_keymap(norm_mode: bool = False, annotation_key=None):
    """Minimal keymap: front image + per-arm ee pose (proprio) + per-arm cmd
    ee pose (action). Deliberately omits wrist images + grippers to keep the
    span read light. ``norm_mode`` (set by trainHydra when building the
    norm-stats dataset) drops the image key."""
    km = {
        "front_img_1": {"key_type": "camera_keys", "zarr_key": "images.front_1"},
        "left.cmd_ee_pose": {"key_type": "action_keys", "zarr_key": "left.cmd_ee_pose"},
        "right.cmd_ee_pose": {"key_type": "action_keys", "zarr_key": "right.cmd_ee_pose"},
        "left.obs_ee_pose": {"key_type": "proprio_keys", "zarr_key": "left.obs_ee_pose"},
        "right.obs_ee_pose": {"key_type": "proprio_keys", "zarr_key": "right.obs_ee_pose"},
    }
    return _drop_camera_keys(km) if norm_mode else km


class CanonicalizeQuatSign(Transform):
    """Force the quaternion hemisphere to w >= 0 on an (..., 7) xyz+quat(wxyz).

    q and -q are the SAME rotation but OPPOSITE regression targets. eva's raw
    cmd_ee_pose/obs_ee_pose quats are bimodal -- 53% of frames sit at w<0 with
    443 mid-episode sign flips per 20 episodes -- so a denoiser fitting the
    conditional mean blends the two hemispheres and carries an irreducible
    error. Human poses were already canonicalised inside HeadFramePose; this
    is the eva-side mirror. Stateless (per frame), so it also holds for the
    single-frame reads norm_stats issues.
    """

    def __init__(self, keys, quat_slice=(3, 7)):
        self.keys = list(keys)
        self.qs = tuple(quat_slice)

    def transform(self, batch: dict) -> dict:
        a, b = self.qs
        for k in self.keys:
            if k not in batch:
                continue
            v = np.array(batch[k], dtype=np.float64, copy=True)
            single = v.ndim == 1
            if single:
                v = v[None, :]
            w = v[:, a:a + 1]
            v[:, a:b] = np.where(w < 0.0, -v[:, a:b], v[:, a:b])
            batch[k] = (v[0] if single else v).astype(np.float32)
        return batch


def eva_span_transforms_quat14():
    return [
        ConcatKeys(
            key_list=["left.cmd_ee_pose", "right.cmd_ee_pose"],
            new_key_name="actions_cartesian",
            delete_old_keys=True,
        ),
        ConcatKeys(
            key_list=["left.obs_ee_pose", "right.obs_ee_pose"],
            new_key_name="state_ee_pose",
            delete_old_keys=True,
        ),
        NumpyToTensor(keys=["actions_cartesian", "state_ee_pose"]),
    ]


# --------------------------------------------------------------------------- #
# human_bimanual
# --------------------------------------------------------------------------- #
def human_span_keymap(norm_mode: bool = False, annotation_key=None):
    km = {
        "front_img_1": {"key_type": "camera_keys", "zarr_key": "images.front_1"},
        "left.obs_keypoints": {"key_type": "proprio_keys", "zarr_key": "left.obs_keypoints"},
        "right.obs_keypoints": {"key_type": "proprio_keys", "zarr_key": "right.obs_keypoints"},
        "left.obs_wrist_pose": {"key_type": "proprio_keys", "zarr_key": "left.obs_wrist_pose"},
        "right.obs_wrist_pose": {"key_type": "proprio_keys", "zarr_key": "right.obs_wrist_pose"},
        "obs_head_pose": {"key_type": "proprio_keys", "zarr_key": "obs_head_pose"},
    }
    return _drop_camera_keys(km) if norm_mode else km


_WRIST_MODE = os.environ.get("RH_WRIST_MODE", "pos").lower()
#: "ypr" -> 138 (legacy, wrist xyz+ypr) | "pos" -> 132 (wrist xyz) |
#: "none" -> 126 (KEYPOINTS ONLY, no wrist pose)
_WRIST = (HeadFrameWristYPR if _WRIST_MODE == "ypr"
          else (None if _WRIST_MODE == "none" else HeadFrameWristPos))


def human_span_transforms():
    # SPAN-SAFE HEAD-FRAME (fold2 FIX 2): keypoints + wrist expressed in each
    # frame's head frame (relative to obs_head_pose), matching the canonical
    # Aria "keypoints_headframe_ypr" representation but computed PER-FRAME so it
    # works on variable-length span reads AND the single-frame windowed probe.
    #   action_keypoints (132) = [Lwrist_xyz(3), Lkp_hf(63), Rwrist_xyz(3), Rkp_hf(63)]
    # (order matches _build_aria_keypoints_bimanual_transform_list). The action
    # is the teacher-forced per-frame head-frame state (obs == action here).
    # RH_WRIST_MODE: ypr -> 138 (legacy wrist xyz+ypr) | pos -> 132 (wrist xyz)
    # | none -> 126 (KEYPOINTS ONLY). 138 is needed to load pre-2026-07-30
    # checkpoints, whose norm stats and per-emb codecs are keyed to it.
    _wrist_ops = ([] if _WRIST is None else [
        _WRIST("obs_head_pose", "left.obs_wrist_pose", "L_wrist_hf"),
        _WRIST("obs_head_pose", "right.obs_wrist_pose", "R_wrist_hf"),
    ])
    _keys = (["L_kp_hf", "R_kp_hf"] if _WRIST is None
             else ["L_wrist_hf", "L_kp_hf", "R_wrist_hf", "R_kp_hf"])
    return [
        *_wrist_ops,
        HeadFrameKeypoints("obs_head_pose", "left.obs_keypoints", "L_kp_hf"),
        HeadFrameKeypoints("obs_head_pose", "right.obs_keypoints", "R_kp_hf"),
        # action target (132): head-frame wrist-xyz + head-frame keypoints per hand.
        ConcatKeys(
            key_list=_keys,
            new_key_name="actions_keypoints",
            delete_old_keys=False,
        ),
        # proprio obs (132): same head-frame layout; consumes the components.
        ConcatKeys(
            key_list=_keys,
            new_key_name="state_keypoints",
            delete_old_keys=True,
        ),
        NumpyToTensor(keys=["actions_keypoints", "state_keypoints", "obs_head_pose"]),
    ]


# --------------------------------------------------------------------------- #
# human_bimanual -- CARTESIAN (action space MATCHED to eva)
#
# "Same action space" cotrain (user, 2026-08-12): both embodiments emit a 14-D
# cartesian end-effector action [left(7), right(7)], each 7 = xyz(3) + quat
# wxyz(4). Because the two spaces are IDENTICAL in dim AND semantics, a
# STANDARD Diffusion Policy works unchanged -- one scalar action_dim, one
# shared head, no per-embodiment codec, no per-emb obs encoder.
#
#   eva  : actions_cartesian(14) = [left.cmd_ee_pose(7), right.cmd_ee_pose(7)]
#          state_ee_pose(14)     = [left.obs_ee_pose(7),  right.obs_ee_pose(7)]
#   human: actions_cartesian(14) = [L_ee_hf(7), R_ee_hf(7)]
#          state_ee_pose(14)     = the same tensor (teacher-forced: the human
#                                  has no separately commanded pose, exactly as
#                                  the keypoints feed already uses obs as its
#                                  own target)
#
# FRAME -- a representation choice, stated rather than buried: the human's
# left/right.obs_ee_pose live in the Aria WORLD frame, whose origin+yaw are
# arbitrary PER RECORDING, so raw world coordinates are not a learnable action
# space (the same fold would sit at a different absolute xyz in every episode).
# They are converted PER FRAME into that frame's HEAD frame -- the convention
# every other human transform in this file already uses, and the closest
# analogue to eva's fixed robot-base frame.
# --------------------------------------------------------------------------- #
class HeadFramePose(Transform):
    """Per-frame head-frame conversion of a full 7-D pose.

    ``pose`` (T, 7) [or (7,) single frame] as xyz + quat(wxyz) -> same shape,
    expressed in that frame's head frame::

        t_hf[t] = R_head[t]^T @ (t_pose[t] - t_head[t])
        R_hf[t] = R_head[t]^T @ R_pose[t]

    The output quaternion is sign-canonicalised to ``w >= 0``. q and -q are the
    same rotation but NOT the same regression target: without this the sign
    flips arbitrarily between frames and the normalizer sees a bimodal
    distribution straddling zero. Canonicalising is stateless, so it also holds
    for the single-frame ``(7,)`` reads that ``norm_stats.populate_from_datasets``
    issues (a previous-frame continuity fix could not).
    """

    def __init__(self, head_key: str, pose_key: str, out_key: str):
        self.head_key = head_key
        self.pose_key = pose_key
        self.out_key = out_key

    def transform(self, batch: dict) -> dict:
        head = np.asarray(batch[self.head_key], dtype=np.float64)
        pose = np.asarray(batch[self.pose_key], dtype=np.float64)
        single = pose.ndim == 1                      # decided by POSE, not head
        if single:
            pose = pose[None, :]
        if head.ndim == 1:
            # one observation frame's head pose applies to every action in that
            # frame's chunk -> broadcast, do not unsqueeze in lockstep. Keying
            # "single" off head.ndim gave head (1,7) vs pose (1,H,7).
            head = np.broadcast_to(head[None, :], (pose.shape[0], head.shape[-1]))
        R_head = _wxyz_to_matrix(head[:, 3:7])                  # (T,3,3)
        R_pose = _wxyz_to_matrix(pose[:, 3:7])
        t_hf = np.einsum("tij,tj->ti", R_head.transpose(0, 2, 1),
                         pose[:, 0:3] - head[:, 0:3])           # (T,3)
        R_hf = np.einsum("tij,tjk->tik", R_head.transpose(0, 2, 1), R_pose)
        xyzw = R.from_matrix(R_hf).as_quat()                    # scipy: xyzw
        q = np.concatenate([xyzw[:, 3:4], xyzw[:, 0:3]], axis=-1)   # -> wxyz
        q = np.where(q[:, 0:1] < 0.0, -q, q)                    # canonical w>=0
        out = np.concatenate([t_hf, q], axis=-1).astype(np.float32)  # (T,7)
        batch[self.out_key] = out[0] if single else out
        return batch


def human_span_cart_keymap(norm_mode: bool = False, annotation_key=None):
    """Cartesian counterpart of :func:`human_span_keymap`: no keypoints, just
    the per-hand end-effector pose + the head pose that defines the frame."""
    km = {
        "front_img_1": {"key_type": "camera_keys", "zarr_key": "images.front_1"},
        "left.obs_ee_pose": {"key_type": "proprio_keys", "zarr_key": "left.obs_ee_pose"},
        "right.obs_ee_pose": {"key_type": "proprio_keys", "zarr_key": "right.obs_ee_pose"},
        "obs_head_pose": {"key_type": "proprio_keys", "zarr_key": "obs_head_pose"},
    }
    return _drop_camera_keys(km) if norm_mode else km


def human_span_cart_transforms_quat14():
    """14-D head-frame cartesian action+state, key names IDENTICAL to eva's so
    the model config needs no per-embodiment branching at all.

    NOTE: unaffected by ``RH_WRIST_MODE`` -- that env var only sizes the
    keypoints feed (126/132/138). This feed is always 14.
    """
    return [
        HeadFramePose("obs_head_pose", "left.obs_ee_pose", "L_ee_hf"),
        HeadFramePose("obs_head_pose", "right.obs_ee_pose", "R_ee_hf"),
        ConcatKeys(
            key_list=["L_ee_hf", "R_ee_hf"],
            new_key_name="actions_cartesian",
            delete_old_keys=False,
        ),
        ConcatKeys(
            key_list=["L_ee_hf", "R_ee_hf"],
            new_key_name="state_ee_pose",
            delete_old_keys=True,
        ),
        NumpyToTensor(keys=["actions_cartesian", "state_ee_pose", "obs_head_pose"]),
    ]

# --------------------------------------------------------------------------- #
# 6D ROTATION ACTION SPACE (user, 2026-08-15)
#
# Quaternions double-cover SO(3): q and -q are the same rotation but opposite
# regression targets. Measured on eva cmd_ee_pose: 53% of frames at w<0 with
# 443 mid-episode sign flips per 20 episodes -- so a mean-seeking denoiser
# blends hemispheres and carries an irreducible error floor. Canonicalising to
# w>=0 only MOVES the seam; the 6D representation (first two columns of R,
# Zhou et al. 2019 "On the Continuity of Rotation Representations") removes it,
# being continuous on SO(3).
#
#   per hand  xyz(3) + rot6d(6) = 9      bimanual = 18   (quat version: 14)
#
# Both embodiments use it, so the matched action space is preserved. Recover R
# with Gram-Schmidt on the two columns; the decoder side is
# rot6d_to_matrix below.
# --------------------------------------------------------------------------- #
def _rot6d_from_matrix(R_):
    """(N,3,3) -> (N,6): first TWO COLUMNS, flattened. Column order matters --
    the Gram-Schmidt inverse must read them back in the same order."""
    return np.concatenate([R_[:, :, 0], R_[:, :, 1]], axis=-1)


def rot6d_to_matrix(d6):
    """(N,6) -> (N,3,3) via Gram-Schmidt. Inverse of _rot6d_from_matrix."""
    a1, a2 = d6[:, 0:3], d6[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    a2p = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2p / (np.linalg.norm(a2p, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


class PoseToRot6D(Transform):
    """xyz + quat(wxyz) (..., 7) -> xyz + rot6d (..., 9). Stateless per frame,
    so it also holds for the single-frame reads norm_stats issues."""

    def __init__(self, in_key: str, out_key: str = None):
        self.in_key = in_key
        self.out_key = out_key or in_key

    def transform(self, batch: dict) -> dict:
        v = np.asarray(batch[self.in_key], dtype=np.float64)
        single = v.ndim == 1
        if single:
            v = v[None, :]
        R_ = _wxyz_to_matrix(v[:, 3:7])
        out = np.concatenate([v[:, 0:3], _rot6d_from_matrix(R_)], axis=-1)
        out = out.astype(np.float32)
        batch[self.out_key] = out[0] if single else out
        return batch


def eva_span_transforms():
    """eva: actions_cartesian(18) / state_ee_pose(18) in xyz+rot6d."""
    return [
        PoseToRot6D("left.cmd_ee_pose"), PoseToRot6D("right.cmd_ee_pose"),
        PoseToRot6D("left.obs_ee_pose"), PoseToRot6D("right.obs_ee_pose"),
        ConcatKeys(key_list=["left.cmd_ee_pose", "right.cmd_ee_pose"],
                   new_key_name="actions_cartesian", delete_old_keys=True),
        ConcatKeys(key_list=["left.obs_ee_pose", "right.obs_ee_pose"],
                   new_key_name="state_ee_pose", delete_old_keys=True),
        NumpyToTensor(keys=["actions_cartesian", "state_ee_pose"]),
    ]


def eva_dfot_keymap(norm_mode: bool = False, annotation_key=None):
    """Robot-only DFoT feed: front RGB plus the commanded 20-D action."""
    km = {
        "front_img_1": {
            "key_type": "camera_keys", "zarr_key": "images.front_1"},
        "left.cmd_ee_pose": {
            "key_type": "action_keys", "zarr_key": "left.cmd_ee_pose"},
        "left.cmd_gripper": {
            "key_type": "action_keys", "zarr_key": "left.cmd_gripper"},
        "right.cmd_ee_pose": {
            "key_type": "action_keys", "zarr_key": "right.cmd_ee_pose"},
        "right.cmd_gripper": {
            "key_type": "action_keys", "zarr_key": "right.cmd_gripper"},
    }
    return _drop_camera_keys(km) if norm_mode else km


def eva_dfot_transforms():
    """Per-frame ``[L xyz+rot6d+grip, R xyz+rot6d+grip]`` action."""
    return [
        PoseToRot6D("left.cmd_ee_pose"),
        PoseToRot6D("right.cmd_ee_pose"),
        ConcatKeys(
            key_list=["left.cmd_ee_pose", "left.cmd_gripper",
                      "right.cmd_ee_pose", "right.cmd_gripper"],
            new_key_name="actions_cartesian", delete_old_keys=True,
        ),
        NumpyToTensor(keys=["actions_cartesian"]),
    ]


def human_span_cart_transforms():
    """human: same 18-D xyz+rot6d, in the per-frame HEAD frame (HeadFramePose
    first, then 6D) -- so eva and human remain a MATCHED action space."""
    return [
        HeadFramePose("obs_head_pose", "left.obs_ee_pose", "L_ee_hf"),
        HeadFramePose("obs_head_pose", "right.obs_ee_pose", "R_ee_hf"),
        PoseToRot6D("L_ee_hf"), PoseToRot6D("R_ee_hf"),
        ConcatKeys(key_list=["L_ee_hf", "R_ee_hf"],
                   new_key_name="actions_cartesian", delete_old_keys=False),
        ConcatKeys(key_list=["L_ee_hf", "R_ee_hf"],
                   new_key_name="state_ee_pose", delete_old_keys=True),
        NumpyToTensor(keys=["actions_cartesian", "state_ee_pose",
                            "obs_head_pose"]),
    ]


# Back-compat aliases: the 6D builders keep their explicit names too, so a
# config can ask for either representation by name rather than by default.
eva_span_rot6d_transforms = eva_span_transforms
human_span_cart_rot6d_transforms = human_span_cart_transforms

# --------------------------------------------------------------------------- #
# NORMAL-DATALOADER (MultiDataset) keymaps + transforms for fold.
#
# This is the STANDARD reader (one sample per FRAME, annotation_collate), which
# is what stock Diffusion Policy trains on. It decodes only the frames it reads
# -- no img_decode_stride, no zero-filled placeholders -- so the black-image
# class of bug the packed h264 path had cannot occur here.
#
# Obs keys carry horizon=N_OBS_STEPS so DP's observation history is REAL data
# rather than the current frame duplicated. Action keys are fetched one frame
# longer (ACTION_HORIZON + N_OBS_STEPS - 1) and then sliced, so the chunk starts
# at the LAST obs frame:
#
#     obs frames      t ........ t+N-1          (N_OBS_STEPS of them)
#     action chunk           t+N-1 ........ t+N-2+ACTION_HORIZON
#
# i.e. the policy sees N frames up to and including the current one and predicts
# ACTION_HORIZON actions starting at the current one. Fetching without the +N-1
# and skipping the slice would silently shift the chunk one frame into the past.
#
# Both embodiments emit the SAME 18-D action (xyz + rot6d per hand), so the
# matched action space carries over from the packed configs unchanged.
#
# Tail frames are repeat-last padded to exactly the requested horizon by
# ZarrDataset._pad_sequences, which is what keeps the chunk fixed-size.
# --------------------------------------------------------------------------- #
ACTION_HORIZON = 100
N_OBS_STEPS = 2
_ACT_FETCH = ACTION_HORIZON + N_OBS_STEPS - 1


class ZeroOut(Transform):
    """DIAGNOSTIC ONLY: overwrite a key with zeros, preserving shape/dtype.
    Used to hand the model a trivially-learnable target: with x0==0 the DDPM
    forward gives x_t = sqrt(1-abar)*eps, so eps is EXACTLY recoverable from
    x_t alone. A model that cannot fit this has a wiring fault, not a data one."""

    def __init__(self, key):
        self.key = str(key)

    def transform(self, batch):
        v = batch[self.key]
        batch[self.key] = np.zeros_like(v)
        return batch


class SliceFrames(Transform):
    """Keep ``[start:stop]`` along the leading (time) axis of ``key``.

    Used to drop the lead-in frames of an over-fetched action chunk so the
    chunk starts at the current obs frame.
    """

    def __init__(self, key, start=0, stop=None, new_key=None):
        self.key = key
        self.start = start
        self.stop = stop
        self.new_key = new_key or key

    def transform(self, batch):
        v = np.asarray(batch[self.key])
        batch[self.new_key] = v[self.start:self.stop]
        return batch


class SelectFrame(Transform):
    """Copy one frame from a time-major array into a standalone pose key."""

    def __init__(self, key, index=-1, new_key=None):
        self.key = key
        self.index = int(index)
        self.new_key = new_key or key

    def transform(self, batch):
        batch[self.new_key] = np.asarray(batch[self.key])[self.index]
        return batch


class ReshapePoints(Transform):
    """Switch between flattened ``(..., N*3)`` and ``(..., N, 3)`` points."""

    def __init__(self, key, n_points=21, flatten=False, new_key=None):
        self.key = key
        self.n_points = int(n_points)
        self.flatten = bool(flatten)
        self.new_key = new_key or key

    def transform(self, batch):
        value = np.asarray(batch[self.key])
        if self.flatten:
            value = value.reshape(*value.shape[:-2], self.n_points * 3)
        else:
            value = value.reshape(*value.shape[:-1], self.n_points, 3)
        batch[self.new_key] = value
        return batch


class PoseXYZ(Transform):
    """Keep only xyz from an ``(..., 7)`` pose, preserving leading axes."""

    def __init__(self, key, new_key=None):
        self.key = key
        self.new_key = new_key or key

    def transform(self, batch):
        batch[self.new_key] = np.asarray(batch[self.key])[..., :3]
        return batch


class ZerosLike(Transform):
    """(T, W) zeros with the leading axis of ``ref_key``.

    Used to pad an embodiment that lacks a channel the shared action layout
    reserves -- Aria has no gripper. The pad is NEVER scored: MaskedActionLoss
    excludes those dims for that embodiment.
    """

    def __init__(self, ref_key, new_key, width=1):
        self.ref_key, self.new_key, self.width = ref_key, new_key, int(width)

    def transform(self, batch):
        ref = np.asarray(batch[self.ref_key])
        batch[self.new_key] = np.zeros((*ref.shape[:-1], self.width),
                                       dtype=np.float32)
        return batch


class DropKeys(Transform):
    """Remove scratch keys so they never reach the collate / norm-stat layer."""

    def __init__(self, keys):
        self.keys = list(keys)

    def transform(self, batch):
        for k in self.keys:
            batch.pop(k, None)
        return batch


def eva_normal_keymap(norm_mode: bool = False, annotation_key=None):
    km = {
        "front_img_1": {"key_type": "camera_keys", "zarr_key": "images.front_1",
                        "horizon": N_OBS_STEPS},
        "front_intrinsics": {"key_type": "metadata_keys",
                             "zarr_key": "intrinsics.front_1"},
        "left_camera_extrinsics": {"key_type": "metadata_keys",
                                   "zarr_key": "extrinsics.left"},
        "right_camera_extrinsics": {"key_type": "metadata_keys",
                                    "zarr_key": "extrinsics.right"},
        # eva HAS a commanded stream -> use it as the action target
        "left.cmd_ee_pose": {"key_type": "action_keys", "zarr_key": "left.cmd_ee_pose",
                             "horizon": _ACT_FETCH},
        "right.cmd_ee_pose": {"key_type": "action_keys", "zarr_key": "right.cmd_ee_pose",
                              "horizon": _ACT_FETCH},
        "left.obs_ee_pose": {"key_type": "proprio_keys", "zarr_key": "left.obs_ee_pose",
                             "horizon": N_OBS_STEPS},
        "right.obs_ee_pose": {"key_type": "proprio_keys", "zarr_key": "right.obs_ee_pose",
                              "horizon": N_OBS_STEPS},
        # GRIPPER (2026-08-16): the fold task is grasp/release, so a policy
        # without this predicts arm motion it can never act on.
        "left.cmd_gripper": {"key_type": "action_keys", "zarr_key": "left.cmd_gripper",
                             "horizon": _ACT_FETCH},
        "right.cmd_gripper": {"key_type": "action_keys", "zarr_key": "right.cmd_gripper",
                              "horizon": _ACT_FETCH},
        "left.obs_gripper": {"key_type": "proprio_keys", "zarr_key": "left.obs_gripper",
                             "horizon": N_OBS_STEPS},
        "right.obs_gripper": {"key_type": "proprio_keys", "zarr_key": "right.obs_gripper",
                              "horizon": N_OBS_STEPS},
        # WRIST CAMERAS -- eva only; Aria has no wrist views, so these belong in
        # the per-embodiment (specific) obs branch, never the agnostic one.
        "left_wrist_img": {"key_type": "camera_keys", "zarr_key": "images.left_wrist",
                           "horizon": N_OBS_STEPS},
        "right_wrist_img": {"key_type": "camera_keys", "zarr_key": "images.right_wrist",
                            "horizon": N_OBS_STEPS},
    }
    return _drop_camera_keys(km) if norm_mode else km


def human_normal_keymap(norm_mode: bool = False, annotation_key=None):
    km = {
        "front_img_1": {"key_type": "camera_keys", "zarr_key": "images.front_1",
                        "horizon": N_OBS_STEPS},
        "front_intrinsics": {"key_type": "metadata_keys",
                             "zarr_key": "intrinsics.front_1"},
        # human has NO commanded stream; obs IS the target (teacher-forced),
        # exactly as the packed human feed already does.
        "left.act_ee_pose": {"key_type": "action_keys", "zarr_key": "left.obs_ee_pose",
                             "horizon": _ACT_FETCH},
        "right.act_ee_pose": {"key_type": "action_keys", "zarr_key": "right.obs_ee_pose",
                              "horizon": _ACT_FETCH},
        "left.obs_ee_pose": {"key_type": "proprio_keys", "zarr_key": "left.obs_ee_pose",
                             "horizon": N_OBS_STEPS},
        "right.obs_ee_pose": {"key_type": "proprio_keys", "zarr_key": "right.obs_ee_pose",
                              "horizon": N_OBS_STEPS},
        "obs_head_pose": {"key_type": "proprio_keys", "zarr_key": "obs_head_pose",
                          "horizon": N_OBS_STEPS},
    }
    return _drop_camera_keys(km) if norm_mode else km


def eva_normal_transforms():
    """Wrist-relative actions + base-frame proprio, both in rot6d layout.

    Each arm's future command is expressed relative to that arm's CURRENT
    observed EEF pose.  This is the same action-frame contract used by the
    human transforms below; only the observation frames remain embodiment
    native (robot base vs human head camera).
    """
    return [
        SelectFrame("left.obs_ee_pose", -1, "left.current_ee_pose"),
        SelectFrame("right.obs_ee_pose", -1, "right.current_ee_pose"),
        SliceFrames("left.cmd_ee_pose", start=N_OBS_STEPS - 1),
        SliceFrames("right.cmd_ee_pose", start=N_OBS_STEPS - 1),
        SliceFrames("left.cmd_gripper", start=N_OBS_STEPS - 1),
        SliceFrames("right.cmd_gripper", start=N_OBS_STEPS - 1),
        ActionChunkCoordinateFrameTransform(
            "left.current_ee_pose", "left.cmd_ee_pose", "left.cmd_ee_wrist",
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            "right.current_ee_pose", "right.cmd_ee_pose", "right.cmd_ee_wrist",
            mode="xyzwxyz",
        ),
        PoseToRot6D("left.cmd_ee_wrist"), PoseToRot6D("right.cmd_ee_wrist"),
        PoseToRot6D("left.obs_ee_pose"), PoseToRot6D("right.obs_ee_pose"),
        # 20-D: [L xyz3 rot6d6 grip1, R xyz3 rot6d6 grip1] -> gripper at 9 and 19
        ConcatKeys(key_list=["left.cmd_ee_wrist", "left.cmd_gripper",
                             "right.cmd_ee_wrist", "right.cmd_gripper"],
                   new_key_name="actions_cartesian", delete_old_keys=True),
        ConcatKeys(key_list=["left.obs_ee_pose", "left.obs_gripper",
                             "right.obs_ee_pose", "right.obs_gripper"],
                   new_key_name="state_ee_pose", delete_old_keys=True),
        DropKeys(["left.cmd_ee_pose", "right.cmd_ee_pose",
                  "left.current_ee_pose", "right.current_ee_pose"]),
        NumpyToTensor(keys=["actions_cartesian", "state_ee_pose"]),
    ]


def human_normal_transforms():
    """Wrist-relative actions + head-frame proprio in the shared 20-D layout."""
    return [
        SelectFrame("left.obs_ee_pose", -1, "left.current_ee_pose"),
        SelectFrame("right.obs_ee_pose", -1, "right.current_ee_pose"),
        SliceFrames("left.act_ee_pose", start=N_OBS_STEPS - 1),
        SliceFrames("right.act_ee_pose", start=N_OBS_STEPS - 1),
        ActionChunkCoordinateFrameTransform(
            "left.current_ee_pose", "left.act_ee_pose", "L_act_wrist",
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            "right.current_ee_pose", "right.act_ee_pose", "R_act_wrist",
            mode="xyzwxyz",
        ),
        HeadFramePose("obs_head_pose", "left.obs_ee_pose", "L_obs_hf"),
        HeadFramePose("obs_head_pose", "right.obs_ee_pose", "R_obs_hf"),
        PoseToRot6D("L_act_wrist"), PoseToRot6D("R_act_wrist"),
        PoseToRot6D("L_obs_hf"), PoseToRot6D("R_obs_hf"),
        # Match eva's 20-D layout. Aria has no gripper, so those two slots are
        # zero pads -- MaskedActionLoss excludes dims 9 and 19 for this
        # embodiment, so they are never scored and never learned.
        ZerosLike("L_act_wrist", "L_grip_pad", 1), ZerosLike("R_act_wrist", "R_grip_pad", 1),
        ConcatKeys(key_list=["L_act_wrist", "L_grip_pad", "R_act_wrist", "R_grip_pad"],
                   new_key_name="actions_cartesian", delete_old_keys=True),
        ZerosLike("L_obs_hf", "L_grip_pad_o", 1), ZerosLike("R_obs_hf", "R_grip_pad_o", 1),
        ConcatKeys(key_list=["L_obs_hf", "L_grip_pad_o", "R_obs_hf", "R_grip_pad_o"],
                   new_key_name="state_ee_pose", delete_old_keys=True),
        # HeadFramePose writes NEW keys, so the raw streams survive the
        # concats (unlike eva, where ConcatKeys consumes them directly).
        # Drop them: nothing downstream reads them, but they would be
        # collated, normalized and shipped to the GPU every step.
        DropKeys(["left.act_ee_pose", "right.act_ee_pose",
                  "left.current_ee_pose", "right.current_ee_pose",
                  "left.obs_ee_pose", "right.obs_ee_pose"]),
        NumpyToTensor(keys=["actions_cartesian", "state_ee_pose", "obs_head_pose"]),
    ]


# --------------------------------------------------------------------------- #
# NORMAL-DATALOADER keypoint variant (human hetero action space).
#
# eva is UNCHANGED -- it reuses eva_normal_{keymap,transforms} (cartesian 18).
# A robot has no body keypoints, so only the human side switches:
#     actions_keypoints (ACTION_HORIZON, 132)   state_keypoints (N_OBS, 132)
#
# The human action is teacher-forced (obs IS the action), so each keypoint /
# wrist array is fetched TWICE from the same zarr key at two horizons: once at
# _ACT_FETCH for the action chunk, once at N_OBS_STEPS for the observation. Each
# copy is paired with a head pose of matching length, because HeadFrameKeypoints
# / HeadFrameWristPos convert per-frame and need equal leading lengths.
# --------------------------------------------------------------------------- #
def human_normal_transforms_synth():
    """human_normal_transforms + the action target ZEROED. Diagnostic: separates
    'the model cannot fit this target' from 'the model cannot fit anything here'."""
    tl = list(human_normal_transforms())
    out = []
    for t in tl:
        out.append(t)
        if isinstance(t, ConcatKeys) and getattr(t, "new_key_name", None) == "actions_cartesian":
            out.append(ZeroOut("actions_cartesian"))
    return out


def human_normal_keymap_kp(norm_mode: bool = False, annotation_key=None):
    P = "proprio_keys"
    km = {
        "front_img_1": {"key_type": "camera_keys", "zarr_key": "images.front_1",
                        "horizon": N_OBS_STEPS},
        "front_intrinsics": {"key_type": "metadata_keys",
                             "zarr_key": "intrinsics.front_1"},
        # ---- action copies (teacher-forced), fetched at the action horizon ----
        "left.act_keypoints":   {"key_type": P, "zarr_key": "left.obs_keypoints",
                                 "horizon": _ACT_FETCH},
        "right.act_keypoints":  {"key_type": P, "zarr_key": "right.obs_keypoints",
                                 "horizon": _ACT_FETCH},
        "left.act_wrist_pose":  {"key_type": P, "zarr_key": "left.obs_wrist_pose",
                                 "horizon": _ACT_FETCH},
        "right.act_wrist_pose": {"key_type": P, "zarr_key": "right.obs_wrist_pose",
                                 "horizon": _ACT_FETCH},
        # ---- observation copies, fetched at the obs-history horizon ----
        "left.obs_keypoints":   {"key_type": P, "zarr_key": "left.obs_keypoints",
                                 "horizon": N_OBS_STEPS},
        "right.obs_keypoints":  {"key_type": P, "zarr_key": "right.obs_keypoints",
                                 "horizon": N_OBS_STEPS},
        "left.obs_wrist_pose":  {"key_type": P, "zarr_key": "left.obs_wrist_pose",
                                 "horizon": N_OBS_STEPS},
        "right.obs_wrist_pose": {"key_type": P, "zarr_key": "right.obs_wrist_pose",
                                 "horizon": N_OBS_STEPS},
        "obs_head_pose":        {"key_type": P, "zarr_key": "obs_head_pose",
                                 "horizon": N_OBS_STEPS},
    }
    return _drop_camera_keys(km) if norm_mode else km


def human_normal_transforms_kp():
    """132 = [Lwrist_xyz(3), Lkp_hf(63), Rwrist_xyz(3), Rkp_hf(63)] per frame,
    same layout and same RH_WRIST_MODE sizing as the packed human_span_transforms
    -- so kp-vs-cart is not confounded by a different human representation."""
    _wrist_act = ([] if _WRIST is None else [
        _WRIST("act_head_pose", "left.act_wrist_pose", "L_wrist_act"),
        _WRIST("act_head_pose", "right.act_wrist_pose", "R_wrist_act"),
    ])
    _wrist_obs = ([] if _WRIST is None else [
        _WRIST("obs_head_pose", "left.obs_wrist_pose", "L_wrist_obs"),
        _WRIST("obs_head_pose", "right.obs_wrist_pose", "R_wrist_obs"),
    ])
    # The action contract is fixed at 132-D regardless of the legacy
    # RH_WRIST_MODE switch: per arm = relative wrist xyz3 + relative kp63.
    act_keys = ["L_wrist_act", "L_kp_act", "R_wrist_act", "R_kp_act"]
    obs_keys = (["L_kp_obs", "R_kp_obs"] if _WRIST is None
                else ["L_wrist_obs", "L_kp_obs", "R_wrist_obs", "R_kp_obs"])
    return [
        SelectFrame("left.obs_wrist_pose", -1, "left.current_wrist_pose"),
        SelectFrame("right.obs_wrist_pose", -1, "right.current_wrist_pose"),
        SliceFrames("left.act_wrist_pose", start=N_OBS_STEPS - 1),
        SliceFrames("right.act_wrist_pose", start=N_OBS_STEPS - 1),
        SliceFrames("left.act_keypoints", start=N_OBS_STEPS - 1),
        SliceFrames("right.act_keypoints", start=N_OBS_STEPS - 1),
        ReshapePoints("left.act_keypoints"),
        ReshapePoints("right.act_keypoints"),
        ActionChunkCoordinateFrameTransform(
            "left.current_wrist_pose", "left.act_wrist_pose", "L_wrist_act",
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            "right.current_wrist_pose", "right.act_wrist_pose", "R_wrist_act",
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            "left.current_wrist_pose", "left.act_keypoints", "L_kp_act",
            mode="xyz",
        ),
        ActionChunkCoordinateFrameTransform(
            "right.current_wrist_pose", "right.act_keypoints", "R_kp_act",
            mode="xyz",
        ),
        PoseXYZ("L_wrist_act"), PoseXYZ("R_wrist_act"),
        ReshapePoints("L_kp_act", flatten=True),
        ReshapePoints("R_kp_act", flatten=True),
        ConcatKeys(key_list=act_keys, new_key_name="actions_keypoints",
                   delete_old_keys=True),
        *_wrist_obs,
        HeadFramePose("obs_head_pose", "left.obs_wrist_pose", "L_wrist_obs_pose_hf"),
        HeadFramePose("obs_head_pose", "right.obs_wrist_pose", "R_wrist_obs_pose_hf"),
        SelectFrame("L_wrist_obs_pose_hf", -1, "L_current_wrist_hf"),
        SelectFrame("R_wrist_obs_pose_hf", -1, "R_current_wrist_hf"),
        ConcatKeys(
            key_list=["L_current_wrist_hf", "R_current_wrist_hf"],
            new_key_name="viz_current_wrist_poses",
            delete_old_keys=True,
        ),
        HeadFrameKeypoints("obs_head_pose", "left.obs_keypoints", "L_kp_obs"),
        HeadFrameKeypoints("obs_head_pose", "right.obs_keypoints", "R_kp_obs"),
        ConcatKeys(key_list=obs_keys, new_key_name="state_keypoints",
                   delete_old_keys=True),
        DropKeys(["left.current_wrist_pose", "right.current_wrist_pose",
                  "left.act_keypoints", "right.act_keypoints",
                  "left.act_wrist_pose", "right.act_wrist_pose",
                  "left.obs_keypoints", "right.obs_keypoints",
                  "left.obs_wrist_pose", "right.obs_wrist_pose"]),
        NumpyToTensor(keys=["actions_keypoints", "state_keypoints", "obs_head_pose",
                            "viz_current_wrist_poses"]),
    ]


# --------------------------------------------------------------------------- #
# ROLLOUT transform lists (2026-08-16).
#
# Deploy must preprocess obs EXACTLY as training did, and must undo the action
# encoding exactly, or the policy is fed/read in a different space than it was
# fitted in -- the classic silent train/deploy skew. These live beside the
# forward lists on purpose: if eva_normal_transforms changes, the mismatch is
# visible in the same file rather than in robot code nobody re-reads.
# --------------------------------------------------------------------------- #
class SplitConcat(Transform):
    """Inverse of ConcatKeys: split one array into named parts along the last
    axis. ``parts`` is [(name, width), ...] and must sum to the array width."""

    def __init__(self, in_key, parts, delete_old_key=True):
        self.in_key = in_key
        self.parts = [(str(n), int(w)) for n, w in parts]
        self.delete_old_key = bool(delete_old_key)

    def transform(self, batch):
        v = np.asarray(batch[self.in_key])
        total = sum(w for _, w in self.parts)
        if v.shape[-1] != total:
            raise ValueError(
                f"SplitConcat: {self.in_key!r} is {v.shape[-1]}-D but parts "
                f"{self.parts} sum to {total}. The action space changed -- "
                f"update the split rather than letting it mis-slice.")
        off = 0
        for name, w in self.parts:
            batch[name] = v[..., off:off + w]
            off += w
        if self.delete_old_key:
            batch.pop(self.in_key, None)
        return batch


class Rot6DToPoseYPR(Transform):
    """xyz + rot6d (..., 9) -> xyz + ypr (..., 6).

    Inverts PoseToRot6D (rot6d_to_matrix is its documented inverse) and then
    converts to the ZYX euler the robot interface speaks -- NOT back to the
    quaternion the raw zarr held, because rollout.py's
    cam_frame_to_base_frame / rot_ee_frame_to_ee_pose_batch both read
    pose[..., 3:6] as ypr.
    """

    def __init__(self, in_key, out_key=None):
        self.in_key = in_key
        self.out_key = out_key or in_key

    def transform(self, batch):
        v = np.asarray(batch[self.in_key], dtype=np.float64)
        single = v.ndim == 1
        if single:
            v = v[None, :]
        if v.shape[-1] != 9:
            raise ValueError(
                f"Rot6DToPoseYPR: {self.in_key!r} is {v.shape[-1]}-D, expected 9 "
                f"(xyz3 + rot6d6).")
        ypr = R.from_matrix(rot6d_to_matrix(v[:, 3:9])).as_euler("ZYX")
        out = np.concatenate([v[:, 0:3], ypr], axis=-1).astype(np.float32)
        batch[self.out_key] = out[0] if single else out
        return batch


def build_bimanual_rot6d_wrist_revert_transforms(
    action_key="actions_cartesian", state_key="state_ee_pose"
):
    """20-D fold wrist chunk -> 14-D parent-frame pose chunk.

    This is only composition: rotation conversion and SE(3) application stay in
    the existing ``Rot6DToPoseYPR`` and ``ActionChunkCoordinateFrameTransform``
    utilities used by the training/rollout pipelines.
    """
    return [
        SplitConcat(
            action_key,
            [("L_action_pose", 9), ("L_action_grip", 1),
             ("R_action_pose", 9), ("R_action_grip", 1)],
        ),
        SplitConcat(
            state_key,
            [("L_state_pose", 9), ("L_state_grip", 1),
             ("R_state_pose", 9), ("R_state_grip", 1)],
            delete_old_key=False,
        ),
        SelectFrame("L_state_pose", -1, "L_current_pose"),
        SelectFrame("R_state_pose", -1, "R_current_pose"),
        Rot6DToPoseYPR("L_action_pose"),
        Rot6DToPoseYPR("R_action_pose"),
        Rot6DToPoseYPR("L_current_pose"),
        Rot6DToPoseYPR("R_current_pose"),
        ActionChunkCoordinateFrameTransform(
            "L_current_pose", "L_action_pose", "L_action_parent",
            mode="xyzypr", inverse=False,
        ),
        ActionChunkCoordinateFrameTransform(
            "R_current_pose", "R_action_pose", "R_action_parent",
            mode="xyzypr", inverse=False,
        ),
        ConcatKeys(
            ["L_action_parent", "L_action_grip",
             "R_action_parent", "R_action_grip"],
            action_key,
            delete_old_keys=True,
        ),
    ]


def build_bimanual_keypoint_wrist_revert_transforms(
    action_key="actions_keypoints", wrist_pose_key="viz_current_wrist_poses"
):
    """132-D fold wrist/keypoint chunk -> head-camera coordinates."""
    return [
        SplitConcat(
            action_key,
            [("L_wrist_xyz", 3), ("L_keypoints", 63),
             ("R_wrist_xyz", 3), ("R_keypoints", 63)],
        ),
        SplitConcat(
            wrist_pose_key,
            [("L_current_wrist", 7), ("R_current_wrist", 7)],
            delete_old_key=False,
        ),
        ReshapePoints("L_keypoints"),
        ReshapePoints("R_keypoints"),
        ActionChunkCoordinateFrameTransform(
            "L_current_wrist", "L_wrist_xyz", "L_wrist_head",
            mode="xyz", inverse=False,
        ),
        ActionChunkCoordinateFrameTransform(
            "R_current_wrist", "R_wrist_xyz", "R_wrist_head",
            mode="xyz", inverse=False,
        ),
        ActionChunkCoordinateFrameTransform(
            "L_current_wrist", "L_keypoints", "L_keypoints_head",
            mode="xyz", inverse=False,
        ),
        ActionChunkCoordinateFrameTransform(
            "R_current_wrist", "R_keypoints", "R_keypoints_head",
            mode="xyz", inverse=False,
        ),
        ReshapePoints("L_keypoints_head", flatten=True),
        ReshapePoints("R_keypoints_head", flatten=True),
        ConcatKeys(
            ["L_wrist_head", "L_keypoints_head",
             "R_wrist_head", "R_keypoints_head"],
            action_key,
            delete_old_keys=True,
        ),
    ]


def eva_rollout_obs_transforms():
    """OBS-ONLY subset of eva_normal_transforms.

    Identical ops on the same keys, minus everything touching cmd_* -- those
    keys do not exist at rollout time. state_ee_pose comes out 20-D, matching
    what the encoder was trained on.
    """
    return [
        PoseToRot6D("left.obs_ee_pose"), PoseToRot6D("right.obs_ee_pose"),
        ConcatKeys(key_list=["left.obs_ee_pose", "left.obs_gripper",
                             "right.obs_ee_pose", "right.obs_gripper"],
                   new_key_name="state_ee_pose", delete_old_keys=True),
        NumpyToTensor(keys=["state_ee_pose"]),
    ]


def eva_action_revert_transforms(in_key="actions_cartesian",
                                 out_key="robot_action"):
    """MODEL action (..., 20) -> ROBOT action (..., 14).

        model : [L xyz3 rot6d6 grip1 | R xyz3 rot6d6 grip1]  = 20
        robot : [L xyz3 ypr3    grip1 | R xyz3 ypr3    grip1] = 14

    The exact inverse of the action path in eva_normal_transforms, re-expressed
    in the robot's ypr convention. Frame transforms (cam->base, rot-ee->ee) are
    NOT here: they need the per-arm extrinsics and live in the rollout node, so
    this list stays pure and testable.
    """
    return [
        SplitConcat(in_key, parts=[("L_pose", 9), ("L_grip", 1),
                                   ("R_pose", 9), ("R_grip", 1)]),
        Rot6DToPoseYPR("L_pose"), Rot6DToPoseYPR("R_pose"),
        ConcatKeys(key_list=["L_pose", "L_grip", "R_pose", "R_grip"],
                   new_key_name=out_key, delete_old_keys=True),
    ]
