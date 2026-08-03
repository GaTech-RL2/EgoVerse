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
raw quaternion / head-frame-relative-nothing forms — good enough to verify the
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
        if v.get("key_type") not in ("camera_keys", "annotation_keys")
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


def eva_span_transforms():
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
