"""Canonical Fold transforms for normal windowed 28-to-100 training.

Human actions contain 28 consecutive source poses (27 intervals) and are
interpolated onto the shared 100-step policy horizon (99 intervals). Robot
actions already use the policy rate. The canonical human keypoint action is
126-D: 21 xyz keypoints per hand, expressed relative to the current wrist.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    ConcatKeys,
    InterpolatePose,
    NumpyToTensor,
    Transform,
)

ACTION_HORIZON = 100
N_OBS_STEPS = 2
HUMAN_ACTION_SOURCE_HORIZON = 28
HUMAN_ACTION_SLOWDOWN = (ACTION_HORIZON - 1) / (HUMAN_ACTION_SOURCE_HORIZON - 1)
_ROBOT_ACTION_FETCH = ACTION_HORIZON + N_OBS_STEPS - 1
_HUMAN_ACTION_FETCH = HUMAN_ACTION_SOURCE_HORIZON + N_OBS_STEPS - 1


def _wxyz_to_matrix(quaternions):
    quaternions = np.asarray(quaternions, dtype=np.float64)
    xyzw = np.concatenate([quaternions[..., 1:4], quaternions[..., 0:1]], axis=-1)
    invalid = np.linalg.norm(xyzw, axis=-1) < 1e-8
    if np.any(invalid):
        xyzw = xyzw.copy()
        xyzw[invalid] = np.array([0.0, 0.0, 0.0, 1.0])
    return Rotation.from_quat(xyzw).as_matrix()


def _matrix_to_wxyz(matrices):
    xyzw = Rotation.from_matrix(matrices).as_quat()
    wxyz = np.concatenate([xyzw[..., 3:4], xyzw[..., :3]], axis=-1)
    return np.where(wxyz[..., :1] < 0.0, -wxyz, wxyz)


def _drop_camera_keys(keymap):
    return {
        key: value
        for key, value in keymap.items()
        if value["key_type"] not in {"camera_keys", "annotation_keys"}
    }


class SliceFrames(Transform):
    def __init__(self, key, start=0, stop=None, new_key=None):
        self.key = key
        self.start = start
        self.stop = stop
        self.new_key = new_key or key

    def transform(self, batch):
        batch[self.new_key] = np.asarray(batch[self.key])[self.start : self.stop]
        return batch


class SelectFrame(Transform):
    def __init__(self, key, index=-1, new_key=None):
        self.key = key
        self.index = int(index)
        self.new_key = new_key or key

    def transform(self, batch):
        batch[self.new_key] = np.asarray(batch[self.key])[self.index]
        return batch


class ReshapePoints(Transform):
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


class ZerosLike(Transform):
    def __init__(self, reference_key, new_key, width=1):
        self.reference_key = reference_key
        self.new_key = new_key
        self.width = int(width)

    def transform(self, batch):
        reference = np.asarray(batch[self.reference_key])
        batch[self.new_key] = np.zeros(
            (*reference.shape[:-1], self.width), dtype=np.float32
        )
        return batch


class DropKeys(Transform):
    def __init__(self, keys):
        self.keys = list(keys)

    def transform(self, batch):
        for key in self.keys:
            batch.pop(key, None)
        return batch


class HeadFramePose(Transform):
    """Express xyz+wxyz poses in the corresponding head frame."""

    def __init__(self, head_key, pose_key, out_key):
        self.head_key = head_key
        self.pose_key = pose_key
        self.out_key = out_key

    def transform(self, batch):
        head = np.asarray(batch[self.head_key], dtype=np.float64)
        pose = np.asarray(batch[self.pose_key], dtype=np.float64)
        single = pose.ndim == 1
        if single:
            pose = pose[None]
        if head.ndim == 1:
            head = np.broadcast_to(head[None], (pose.shape[0], head.shape[-1]))
        head_rotation = _wxyz_to_matrix(head[:, 3:7])
        inverse_head = head_rotation.transpose(0, 2, 1)
        position = np.einsum("tij,tj->ti", inverse_head, pose[:, :3] - head[:, :3])
        rotation = np.einsum(
            "tij,tjk->tik", inverse_head, _wxyz_to_matrix(pose[:, 3:7])
        )
        output = np.concatenate([position, _matrix_to_wxyz(rotation)], axis=-1)
        batch[self.out_key] = output[0] if single else output
        return batch


class HeadFrameKeypoints(Transform):
    """Express flattened xyz keypoints in their corresponding head frames."""

    def __init__(self, head_key, keypoint_key, out_key, n_points=21):
        self.head_key = head_key
        self.keypoint_key = keypoint_key
        self.out_key = out_key
        self.n_points = int(n_points)

    def transform(self, batch):
        head = np.asarray(batch[self.head_key], dtype=np.float64)
        points = np.asarray(batch[self.keypoint_key], dtype=np.float64)
        single = points.ndim == 1
        if single:
            points = points[None]
        if head.ndim == 1:
            head = np.broadcast_to(head[None], (points.shape[0], head.shape[-1]))
        points = points.reshape(points.shape[0], self.n_points, 3)
        inverse_head = _wxyz_to_matrix(head[:, 3:7]).transpose(0, 2, 1)
        output = np.einsum(
            "tij,tnj->tni", inverse_head, points - head[:, None, :3]
        ).reshape(points.shape[0], -1)
        batch[self.out_key] = output[0] if single else output
        return batch


def _rot6d_from_matrix(matrices):
    return np.concatenate([matrices[..., :, 0], matrices[..., :, 1]], axis=-1)


def rot6d_to_matrix(rotation_6d):
    rotation_6d = np.asarray(rotation_6d)
    first, second = rotation_6d[..., :3], rotation_6d[..., 3:6]
    first = first / np.linalg.norm(first, axis=-1, keepdims=True).clip(1e-8)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.linalg.norm(second, axis=-1, keepdims=True).clip(1e-8)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=-1)


class PoseToRot6D(Transform):
    def __init__(self, in_key, out_key=None):
        self.in_key = in_key
        self.out_key = out_key or in_key

    def transform(self, batch):
        value = np.asarray(batch[self.in_key], dtype=np.float64)
        output = np.concatenate(
            [value[..., :3], _rot6d_from_matrix(_wxyz_to_matrix(value[..., 3:7]))],
            axis=-1,
        ).astype(np.float32)
        batch[self.out_key] = output
        return batch


def eva_normal_keymap(norm_mode=False, annotation_key=None):
    keymap = {
        "front_img_1": {
            "key_type": "camera_keys",
            "zarr_key": "images.front_1",
            "horizon": N_OBS_STEPS,
        },
        "front_intrinsics": {
            "key_type": "metadata_keys",
            "zarr_key": "intrinsics.front_1",
        },
        "left_camera_extrinsics": {
            "key_type": "metadata_keys",
            "zarr_key": "extrinsics.left",
        },
        "right_camera_extrinsics": {
            "key_type": "metadata_keys",
            "zarr_key": "extrinsics.right",
        },
        "left.cmd_ee_pose": {
            "key_type": "action_keys",
            "zarr_key": "left.cmd_ee_pose",
            "horizon": _ROBOT_ACTION_FETCH,
        },
        "right.cmd_ee_pose": {
            "key_type": "action_keys",
            "zarr_key": "right.cmd_ee_pose",
            "horizon": _ROBOT_ACTION_FETCH,
        },
        "left.cmd_gripper": {
            "key_type": "action_keys",
            "zarr_key": "left.cmd_gripper",
            "horizon": _ROBOT_ACTION_FETCH,
        },
        "right.cmd_gripper": {
            "key_type": "action_keys",
            "zarr_key": "right.cmd_gripper",
            "horizon": _ROBOT_ACTION_FETCH,
        },
        "left.obs_ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "left.obs_ee_pose",
            "horizon": N_OBS_STEPS,
        },
        "right.obs_ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "right.obs_ee_pose",
            "horizon": N_OBS_STEPS,
        },
        "left.obs_gripper": {
            "key_type": "proprio_keys",
            "zarr_key": "left.obs_gripper",
            "horizon": N_OBS_STEPS,
        },
        "right.obs_gripper": {
            "key_type": "proprio_keys",
            "zarr_key": "right.obs_gripper",
            "horizon": N_OBS_STEPS,
        },
        "left_wrist_img": {
            "key_type": "camera_keys",
            "zarr_key": "images.left_wrist",
            "horizon": N_OBS_STEPS,
        },
        "right_wrist_img": {
            "key_type": "camera_keys",
            "zarr_key": "images.right_wrist",
            "horizon": N_OBS_STEPS,
        },
    }
    return _drop_camera_keys(keymap) if norm_mode else keymap


def human_normal_keymap(norm_mode=False, annotation_key=None):
    keymap = {
        "front_img_1": {
            "key_type": "camera_keys",
            "zarr_key": "images.front_1",
            "horizon": N_OBS_STEPS,
        },
        "front_intrinsics": {
            "key_type": "metadata_keys",
            "zarr_key": "intrinsics.front_1",
        },
        "left.act_ee_pose": {
            "key_type": "action_keys",
            "zarr_key": "left.obs_ee_pose",
            "horizon": _HUMAN_ACTION_FETCH,
        },
        "right.act_ee_pose": {
            "key_type": "action_keys",
            "zarr_key": "right.obs_ee_pose",
            "horizon": _HUMAN_ACTION_FETCH,
        },
        "left.obs_ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "left.obs_ee_pose",
            "horizon": N_OBS_STEPS,
        },
        "right.obs_ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "right.obs_ee_pose",
            "horizon": N_OBS_STEPS,
        },
        "obs_head_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "obs_head_pose",
            "horizon": N_OBS_STEPS,
        },
    }
    return _drop_camera_keys(keymap) if norm_mode else keymap


def human_normal_keymap_kp(norm_mode=False, annotation_key=None):
    keymap = {
        "front_img_1": {
            "key_type": "camera_keys",
            "zarr_key": "images.front_1",
            "horizon": N_OBS_STEPS,
        },
        "front_intrinsics": {
            "key_type": "metadata_keys",
            "zarr_key": "intrinsics.front_1",
        },
        "left.act_keypoints": {
            "key_type": "proprio_keys",
            "zarr_key": "left.obs_keypoints",
            "horizon": _HUMAN_ACTION_FETCH,
        },
        "right.act_keypoints": {
            "key_type": "proprio_keys",
            "zarr_key": "right.obs_keypoints",
            "horizon": _HUMAN_ACTION_FETCH,
        },
        "left.obs_keypoints": {
            "key_type": "proprio_keys",
            "zarr_key": "left.obs_keypoints",
            "horizon": N_OBS_STEPS,
        },
        "right.obs_keypoints": {
            "key_type": "proprio_keys",
            "zarr_key": "right.obs_keypoints",
            "horizon": N_OBS_STEPS,
        },
        "left.obs_wrist_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "left.obs_wrist_pose",
            "horizon": N_OBS_STEPS,
        },
        "right.obs_wrist_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "right.obs_wrist_pose",
            "horizon": N_OBS_STEPS,
        },
        "obs_head_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "obs_head_pose",
            "horizon": N_OBS_STEPS,
        },
    }
    return _drop_camera_keys(keymap) if norm_mode else keymap


def eva_normal_transforms():
    return [
        SelectFrame("left.obs_ee_pose", -1, "left.current_ee_pose"),
        SelectFrame("right.obs_ee_pose", -1, "right.current_ee_pose"),
        SliceFrames("left.cmd_ee_pose", N_OBS_STEPS - 1),
        SliceFrames("right.cmd_ee_pose", N_OBS_STEPS - 1),
        SliceFrames("left.cmd_gripper", N_OBS_STEPS - 1),
        SliceFrames("right.cmd_gripper", N_OBS_STEPS - 1),
        ActionChunkCoordinateFrameTransform(
            "left.current_ee_pose",
            "left.cmd_ee_pose",
            "left.cmd_ee_wrist",
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            "right.current_ee_pose",
            "right.cmd_ee_pose",
            "right.cmd_ee_wrist",
            mode="xyzwxyz",
        ),
        PoseToRot6D("left.cmd_ee_wrist"),
        PoseToRot6D("right.cmd_ee_wrist"),
        PoseToRot6D("left.obs_ee_pose"),
        PoseToRot6D("right.obs_ee_pose"),
        ConcatKeys(
            [
                "left.cmd_ee_wrist",
                "left.cmd_gripper",
                "right.cmd_ee_wrist",
                "right.cmd_gripper",
            ],
            "actions_cartesian",
            delete_old_keys=True,
        ),
        ConcatKeys(
            [
                "left.obs_ee_pose",
                "left.obs_gripper",
                "right.obs_ee_pose",
                "right.obs_gripper",
            ],
            "state_ee_pose",
            delete_old_keys=True,
        ),
        DropKeys(
            [
                "left.cmd_ee_pose",
                "right.cmd_ee_pose",
                "left.current_ee_pose",
                "right.current_ee_pose",
            ]
        ),
        NumpyToTensor(keys=["actions_cartesian", "state_ee_pose"]),
    ]


def human_normal_transforms():
    transforms = [
        SelectFrame("left.obs_ee_pose", -1, "left.current_ee_pose"),
        SelectFrame("right.obs_ee_pose", -1, "right.current_ee_pose"),
        SliceFrames("left.act_ee_pose", N_OBS_STEPS - 1),
        SliceFrames("right.act_ee_pose", N_OBS_STEPS - 1),
        ActionChunkCoordinateFrameTransform(
            "left.current_ee_pose",
            "left.act_ee_pose",
            "left.action_wrist",
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            "right.current_ee_pose",
            "right.act_ee_pose",
            "right.action_wrist",
            mode="xyzwxyz",
        ),
        InterpolatePose(
            ACTION_HORIZON,
            "left.action_wrist",
            "left.action_wrist",
            stride=1,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            ACTION_HORIZON,
            "right.action_wrist",
            "right.action_wrist",
            stride=1,
            mode="xyzwxyz",
        ),
        HeadFramePose("obs_head_pose", "left.obs_ee_pose", "left.obs_head"),
        HeadFramePose("obs_head_pose", "right.obs_ee_pose", "right.obs_head"),
        PoseToRot6D("left.action_wrist"),
        PoseToRot6D("right.action_wrist"),
        PoseToRot6D("left.obs_head"),
        PoseToRot6D("right.obs_head"),
        ZerosLike("left.action_wrist", "left.action_gripper"),
        ZerosLike("right.action_wrist", "right.action_gripper"),
        ConcatKeys(
            [
                "left.action_wrist",
                "left.action_gripper",
                "right.action_wrist",
                "right.action_gripper",
            ],
            "actions_cartesian",
            delete_old_keys=True,
        ),
        ZerosLike("left.obs_head", "left.obs_gripper"),
        ZerosLike("right.obs_head", "right.obs_gripper"),
        ConcatKeys(
            [
                "left.obs_head",
                "left.obs_gripper",
                "right.obs_head",
                "right.obs_gripper",
            ],
            "state_ee_pose",
            delete_old_keys=True,
        ),
        DropKeys(
            [
                "left.act_ee_pose",
                "right.act_ee_pose",
                "left.current_ee_pose",
                "right.current_ee_pose",
                "left.obs_ee_pose",
                "right.obs_ee_pose",
            ]
        ),
        NumpyToTensor(keys=["actions_cartesian", "state_ee_pose", "obs_head_pose"]),
    ]
    return transforms


def human_normal_transforms_kp():
    return [
        SelectFrame("left.obs_wrist_pose", -1, "left.current_wrist_pose"),
        SelectFrame("right.obs_wrist_pose", -1, "right.current_wrist_pose"),
        SliceFrames("left.act_keypoints", N_OBS_STEPS - 1),
        SliceFrames("right.act_keypoints", N_OBS_STEPS - 1),
        ReshapePoints("left.act_keypoints"),
        ReshapePoints("right.act_keypoints"),
        ActionChunkCoordinateFrameTransform(
            "left.current_wrist_pose",
            "left.act_keypoints",
            "left.action_keypoints",
            mode="xyz",
        ),
        ActionChunkCoordinateFrameTransform(
            "right.current_wrist_pose",
            "right.act_keypoints",
            "right.action_keypoints",
            mode="xyz",
        ),
        InterpolatePose(
            ACTION_HORIZON,
            "left.action_keypoints",
            "left.action_keypoints",
            stride=1,
            mode="xyz",
        ),
        InterpolatePose(
            ACTION_HORIZON,
            "right.action_keypoints",
            "right.action_keypoints",
            stride=1,
            mode="xyz",
        ),
        ReshapePoints("left.action_keypoints", flatten=True),
        ReshapePoints("right.action_keypoints", flatten=True),
        ConcatKeys(
            ["left.action_keypoints", "right.action_keypoints"],
            "actions_keypoints",
            delete_old_keys=True,
        ),
        HeadFramePose("obs_head_pose", "left.obs_wrist_pose", "left.obs_wrist_head"),
        HeadFramePose("obs_head_pose", "right.obs_wrist_pose", "right.obs_wrist_head"),
        SelectFrame("left.obs_wrist_head", -1, "left.current_wrist_head"),
        SelectFrame("right.obs_wrist_head", -1, "right.current_wrist_head"),
        ConcatKeys(
            ["left.current_wrist_head", "right.current_wrist_head"],
            "viz_current_wrist_poses",
            delete_old_keys=True,
        ),
        HeadFrameKeypoints(
            "obs_head_pose", "left.obs_keypoints", "left.obs_keypoints_head"
        ),
        HeadFrameKeypoints(
            "obs_head_pose", "right.obs_keypoints", "right.obs_keypoints_head"
        ),
        ConcatKeys(
            ["left.obs_keypoints_head", "right.obs_keypoints_head"],
            "state_keypoints",
            delete_old_keys=True,
        ),
        DropKeys(
            [
                "left.current_wrist_pose",
                "right.current_wrist_pose",
                "left.act_keypoints",
                "right.act_keypoints",
                "left.obs_keypoints",
                "right.obs_keypoints",
                "left.obs_wrist_pose",
                "right.obs_wrist_pose",
            ]
        ),
        NumpyToTensor(
            keys=[
                "actions_keypoints",
                "state_keypoints",
                "obs_head_pose",
                "viz_current_wrist_poses",
            ]
        ),
    ]


class SplitConcat(Transform):
    def __init__(self, in_key, parts, delete_old_key=True):
        self.in_key = in_key
        self.parts = [(str(name), int(width)) for name, width in parts]
        self.delete_old_key = bool(delete_old_key)

    def transform(self, batch):
        value = np.asarray(batch[self.in_key])
        if value.shape[-1] != sum(width for _, width in self.parts):
            raise ValueError(f"{self.in_key!r} has unexpected width {value.shape[-1]}")
        offset = 0
        for name, width in self.parts:
            batch[name] = value[..., offset : offset + width]
            offset += width
        if self.delete_old_key:
            batch.pop(self.in_key, None)
        return batch


class Rot6DToPoseYPR(Transform):
    def __init__(self, in_key, out_key=None):
        self.in_key = in_key
        self.out_key = out_key or in_key

    def transform(self, batch):
        value = np.asarray(batch[self.in_key], dtype=np.float64)
        if value.shape[-1] != 9:
            raise ValueError(f"{self.in_key!r} must be xyz3 + rot6d6")
        ypr = Rotation.from_matrix(rot6d_to_matrix(value[..., 3:9])).as_euler("ZYX")
        batch[self.out_key] = np.concatenate([value[..., :3], ypr], axis=-1).astype(
            np.float32
        )
        return batch


def build_bimanual_rot6d_wrist_revert_transforms(
    action_key="actions_cartesian", state_key="state_ee_pose"
):
    return [
        SplitConcat(
            action_key,
            [
                ("left.action_pose", 9),
                ("left.action_grip", 1),
                ("right.action_pose", 9),
                ("right.action_grip", 1),
            ],
        ),
        SplitConcat(
            state_key,
            [
                ("left.state_pose", 9),
                ("left.state_grip", 1),
                ("right.state_pose", 9),
                ("right.state_grip", 1),
            ],
            delete_old_key=False,
        ),
        SelectFrame("left.state_pose", -1, "left.current_pose"),
        SelectFrame("right.state_pose", -1, "right.current_pose"),
        Rot6DToPoseYPR("left.action_pose"),
        Rot6DToPoseYPR("right.action_pose"),
        Rot6DToPoseYPR("left.current_pose"),
        Rot6DToPoseYPR("right.current_pose"),
        ActionChunkCoordinateFrameTransform(
            "left.current_pose",
            "left.action_pose",
            "left.action_parent",
            mode="xyzypr",
            inverse=False,
        ),
        ActionChunkCoordinateFrameTransform(
            "right.current_pose",
            "right.action_pose",
            "right.action_parent",
            mode="xyzypr",
            inverse=False,
        ),
        ConcatKeys(
            [
                "left.action_parent",
                "left.action_grip",
                "right.action_parent",
                "right.action_grip",
            ],
            action_key,
            delete_old_keys=True,
        ),
    ]


class BimanualKeypointWristRevert(Transform):
    def __init__(self, action_key, wrist_pose_key):
        self.action_key = action_key
        self.wrist_pose_key = wrist_pose_key

    @staticmethod
    def _apply(points, pose):
        rotation = _wxyz_to_matrix(np.asarray(pose)[None, 3:7])[0]
        return np.einsum("ij,tnj->tni", rotation, points) + pose[:3]

    def transform(self, batch):
        actions = np.asarray(batch[self.action_key], dtype=np.float64)
        wrists = np.asarray(batch[self.wrist_pose_key], dtype=np.float64)
        if wrists.shape != (14,):
            raise ValueError(f"{self.wrist_pose_key!r} must contain two 7-D poses")
        if actions.shape[-1] == 126:
            left = actions[..., :63].reshape(-1, 21, 3)
            right = actions[..., 63:].reshape(-1, 21, 3)
            pieces = [
                self._apply(left, wrists[:7]).reshape(-1, 63),
                self._apply(right, wrists[7:]).reshape(-1, 63),
            ]
        elif actions.shape[-1] == 132:
            left_wrist = actions[..., :3].reshape(-1, 1, 3)
            left = actions[..., 3:66].reshape(-1, 21, 3)
            right_wrist = actions[..., 66:69].reshape(-1, 1, 3)
            right = actions[..., 69:].reshape(-1, 21, 3)
            pieces = [
                self._apply(left_wrist, wrists[:7]).reshape(-1, 3),
                self._apply(left, wrists[:7]).reshape(-1, 63),
                self._apply(right_wrist, wrists[7:]).reshape(-1, 3),
                self._apply(right, wrists[7:]).reshape(-1, 63),
            ]
        else:
            raise ValueError(
                "keypoint actions must use canonical 126-D or legacy 132-D"
            )
        batch[self.action_key] = np.concatenate(pieces, axis=-1).astype(np.float32)
        return batch


def build_bimanual_keypoint_wrist_revert_transforms(
    action_key="actions_keypoints", wrist_pose_key="viz_current_wrist_poses"
):
    return [BimanualKeypointWristRevert(action_key, wrist_pose_key)]


def eva_rollout_obs_transforms():
    return [
        PoseToRot6D("left.obs_ee_pose"),
        PoseToRot6D("right.obs_ee_pose"),
        ConcatKeys(
            [
                "left.obs_ee_pose",
                "left.obs_gripper",
                "right.obs_ee_pose",
                "right.obs_gripper",
            ],
            "state_ee_pose",
            delete_old_keys=True,
        ),
        NumpyToTensor(keys=["state_ee_pose"]),
    ]


def eva_action_revert_transforms(in_key="actions_cartesian", out_key="robot_action"):
    return [
        SplitConcat(
            in_key,
            [("left.pose", 9), ("left.grip", 1), ("right.pose", 9), ("right.grip", 1)],
        ),
        Rot6DToPoseYPR("left.pose"),
        Rot6DToPoseYPR("right.pose"),
        ConcatKeys(
            ["left.pose", "left.grip", "right.pose", "right.grip"],
            out_key,
            delete_old_keys=True,
        ),
    ]
