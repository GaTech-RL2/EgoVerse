import os
from datetime import datetime, timezone
from typing import Dict, List

import cv2
import numpy as np
import projectaria_tools.core.sophus as sp
import torch
import torch.nn.functional as F
from projectaria_tools.core import calibration, data_provider, mps
from projectaria_tools.core.mps.utils import (
    get_nearest_hand_tracking_result,
    get_nearest_pose,
)
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId

from egomimic.utils.egomimicUtils import (
    INTRINSICS,
    cam_frame_to_cam_pixels,
)
from egomimic.utils.pose_utils import T_rot_orientation

ROTATION_MATRIX = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
T_ROT_CAM = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

HORIZON_DEFAULT = 10
STEP_DEFAULT = 3.0
EPISODE_LENGTH = 100
CHUNK_LENGTH_ACT = 100


def undistort_to_linear(
    provider, stream_ids, raw_image, camera_label="rgb", height=480, width=640
):
    camera_label = provider.get_label_from_stream_id(stream_ids[camera_label])
    calib = provider.get_device_calibration().get_camera_calib(camera_label)
    focal_length = 133.25430222 * (height / 240)
    warped = calibration.get_linear_camera_calibration(
        height, width, focal_length, camera_label, calib.get_transform_device_camera()
    )
    warped_image = calibration.distort_by_calibration(raw_image, warped, calib)
    warped_rot = np.rot90(warped_image, k=3)
    return warped_rot


def slam_to_rgb(provider, height=480, width=640):
    """
    Get slam camera to rgb camera transform
    provider: vrs data provider
    """
    focal_length = 133.25430222 * (height / 240)
    device_calibration = provider.get_device_calibration()

    slam_id = StreamId("1201-1")
    slam_label = provider.get_label_from_stream_id(slam_id)
    slam_calib = device_calibration.get_camera_calib(slam_label)
    slam_camera = calibration.get_linear_camera_calibration(
        height,
        width,
        focal_length,
        slam_label,
        slam_calib.get_transform_device_camera(),
    )
    T_device_slam_camera = (
        slam_camera.get_transform_device_camera()
    )  # slam to device frame

    rgb_id = StreamId("214-1")
    rgb_label = provider.get_label_from_stream_id(rgb_id)
    rgb_calib = device_calibration.get_camera_calib(rgb_label)
    rgb_camera = calibration.get_linear_camera_calibration(
        height, width, focal_length, rgb_label, rgb_calib.get_transform_device_camera()
    )
    T_device_rgb_camera = (
        rgb_camera.get_transform_device_camera()
    )  # rgb to device frame

    transform = T_device_rgb_camera.inverse() @ T_device_slam_camera

    return transform


def compute_coordinate_frame(palm_pose, wrist_pose, palm_normal):
    x_axis = wrist_pose - palm_pose
    x_axis = np.ravel(x_axis) / np.linalg.norm(x_axis)
    z_axis = np.ravel(palm_normal) / np.linalg.norm(palm_normal)
    y_axis = np.cross(x_axis, z_axis)
    y_axis = np.ravel(y_axis) / np.linalg.norm(y_axis)

    x_axis = np.cross(z_axis, y_axis)
    x_axis = np.ravel(x_axis) / np.linalg.norm(x_axis)

    return -1 * x_axis, y_axis, z_axis


def compute_orientation_rotation_matrix(palm_pose, wrist_pose, palm_normal):
    x_axis = wrist_pose - palm_pose
    x_axis = np.ravel(x_axis) / np.linalg.norm(x_axis)
    z_axis = np.ravel(palm_normal) / np.linalg.norm(palm_normal)
    y_axis = np.cross(x_axis, z_axis)
    y_axis = np.ravel(y_axis) / np.linalg.norm(y_axis)

    x_axis = np.cross(z_axis, y_axis)
    x_axis = np.ravel(x_axis) / np.linalg.norm(x_axis)

    rot_matrix = np.column_stack([-1 * x_axis, y_axis, z_axis])
    return rot_matrix


def timestamp_ms_to_episode_hash(timestamp_ms: int) -> str:
    """
    Convert UTC epoch milliseconds -> string like "2026-01-12-03-47-29-664000".
    (microseconds are always 6 digits; last 3 digits will be 000 because input is ms)
    """
    if not isinstance(timestamp_ms, int):
        raise TypeError("timestamp_ms must be an int (UTC epoch milliseconds).")

    dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d-%H-%M-%S-%f")


def downsample_hwc_uint8_in_chunks(
    images: np.ndarray,  # (T,H,W,3) uint8
    out_hw=(240, 320),
    chunk: int = 256,
) -> np.ndarray:
    assert images.dtype == np.uint8 and images.ndim == 4 and images.shape[-1] == 3
    T, H, W, C = images.shape
    outH, outW = out_hw

    out = np.empty((T, outH, outW, 3), dtype=np.uint8)

    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        x = (
            torch.from_numpy(images[s:e]).permute(0, 3, 1, 2).to(torch.float32) / 255.0
        )  # (B,3,H,W)
        x = F.interpolate(x, size=(outH, outW), mode="bilinear", align_corners=False)
        x = (x * 255.0).clamp(0, 255).to(torch.uint8)  # (B,3,outH,outW)
        out[s:e] = x.permute(0, 2, 3, 1).cpu().numpy()
        del x

    return out


def quat_translation_swap(quat_translation: np.ndarray) -> np.ndarray:
    """
    Swap the quaternion and translation in a (N, 7) array.
    Parameters
    ----------
    quat_translation : np.ndarray
        (N, 7) array of quaternion and translation
    Returns
    -------
    np.ndarray:
        (N, 7) array of translation and quaternion
    """
    return np.concatenate(
        (quat_translation[..., 4:7], quat_translation[..., 0:4]), axis=-1
    )


class AriaVRSExtractor:
    TAGS = ["aria", "robotics", "vrs"]

    @staticmethod
    def process_episode(episode_path, arm: str, low_res=False, height=480, width=640):
        """
        Extracts all feature keys from a given episode and returns as a dictionary
        Parameters
        ----------
        episode_path : str or Path
            Path to the VRS file containing the episode data.
        arm : str
            String for which arm to add data for
        Returns
        -------
        episode_feats : dict
            Dictionary mapping keys in the episode to episode features, for example:
                hand.<cartesian>   : (world frame) (6D per arm)
                hand.<keypoints>   : (world frame) (3 cartesian + 4 quaternion + 63 dim (21 keypoints) per arm)
                images.<camera_key>    :
                head_pose              : (world frame)

            #TODO: Add metadata to be a nested dict

        """
        episode_feats = dict()

        # file setup and opening
        root_dir = episode_path.parent

        mps_sample_path = os.path.join(root_dir, ("mps_" + episode_path.stem + "_vrs"))

        hand_tracking_results_path = os.path.join(
            mps_sample_path, "hand_tracking", "hand_tracking_results.csv"
        )

        closed_loop_pose_path = os.path.join(
            mps_sample_path, "slam", "closed_loop_trajectory.csv"
        )
        # TODO: in the future might write to sql on the failure due to mps processing failures
        if not os.path.exists(hand_tracking_results_path):
            raise FileNotFoundError(
                f"Hand tracking results file not found at {hand_tracking_results_path}"
            )
        if not os.path.exists(closed_loop_pose_path):
            raise FileNotFoundError(
                f"Closed loop pose file not found at {closed_loop_pose_path}"
            )

        vrs_reader = data_provider.create_vrs_data_provider(str(episode_path))

        hand_tracking_results = mps.hand_tracking.read_hand_tracking_results(
            hand_tracking_results_path
        )

        closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_pose_path)

        time_domain: TimeDomain = TimeDomain.DEVICE_TIME

        stream_ids: Dict[str, StreamId] = {
            "rgb": StreamId("214-1"),
            "slam-left": StreamId("1201-1"),
            "slam-right": StreamId("1201-2"),
        }
        stream_timestamps_ns: Dict[str, List[int]] = {
            key: vrs_reader.get_timestamps_ns(stream_id, time_domain)
            for key, stream_id in stream_ids.items()
        }

        rgb_to_device_T = slam_to_rgb(
            vrs_reader, height=height, width=width
        )  # aria sophus SE3

        hand_cartesian_pose = AriaVRSExtractor.get_ee_pose(
            world_device_T=closed_loop_traj,
            stream_timestamps_ns=stream_timestamps_ns,
            hand_tracking_results=hand_tracking_results,
            arm=arm,
        )

        hand_keypoints_pose = AriaVRSExtractor.get_hand_keypoints(
            world_device_T=closed_loop_traj,
            stream_timestamps_ns=stream_timestamps_ns,
            hand_tracking_results=hand_tracking_results,
            arm=arm,
        )

        head_pose = AriaVRSExtractor.get_head_pose(
            world_device_T=closed_loop_traj,
            device_rgb_T=rgb_to_device_T.inverse(),
            stream_timestamps_ns=stream_timestamps_ns,
        )

        images = AriaVRSExtractor.get_images(
            vrs_reader=vrs_reader,
            stream_ids=stream_ids,
            stream_timestamps_ns=stream_timestamps_ns,
            height=height,
            width=width,
        )

        if low_res:
            images = downsample_hwc_uint8_in_chunks(
                images, out_hw=(240, 320), chunk=256
            )

        print(f"[DEBUG] LENGTH BEFORE CLEANING: {len(hand_cartesian_pose)}")
        [hand_cartesian_pose, hand_keypoints_pose, head_pose], images = (
            AriaVRSExtractor.clean_data(
                poses=[hand_cartesian_pose, hand_keypoints_pose, head_pose],
                images=images,
            )
        )
        # actions, pose, images = AriaVRSExtractor.clean_data_projection(actions=actions, pose=pose, images=images, arm=arm)
        print(f"[DEBUG] LENGTH AFTER CLEANING: {len(hand_cartesian_pose)}")

        episode_feats["left.obs_ee_pose"] = hand_cartesian_pose[..., :7]
        episode_feats["right.obs_ee_pose"] = hand_cartesian_pose[..., 7:]
        episode_feats["left.obs_keypoints"] = hand_keypoints_pose[..., 7 : 7 + 21 * 3]
        episode_feats["right.obs_keypoints"] = hand_keypoints_pose[
            ..., 7 + 21 * 3 + 7 : 7 + 21 * 3 + 7 + 21 * 3
        ]
        episode_feats["left.obs_wrist_pose"] = hand_keypoints_pose[..., :7]
        episode_feats["right.obs_wrist_pose"] = hand_keypoints_pose[
            ..., 7 + 21 * 3 : 7 + 21 * 3 + 7
        ]
        episode_feats["images.front_1"] = images
        episode_feats["obs_head_pose"] = head_pose

        return episode_feats

    @staticmethod
    def clean_data(poses, images):
        """
        Clean data
        Parameters
        ----------
        actions : np.array
        pose : np.array
        images : np.array
        Returns
        -------
        actions, pose, images : tuple of np.array
            cleaned data
        """
        mask_poses = np.ones(len(poses[0]), dtype=bool)
        for pose in poses:
            bad_data_mask = np.any(pose >= 1e8, axis=1)
            mask_poses = mask_poses & ~bad_data_mask

        for i in range(len(poses)):
            poses[i] = poses[i][mask_poses]
        clean_images = images[mask_poses]

        return poses, clean_images

    @staticmethod
    def clean_data_projection(
        actions, pose, images, arm, CHUNK_LENGTH=CHUNK_LENGTH_ACT
    ):
        """
        Clean data
        Parameters
        ----------
        actions : np.array
        pose : np.array
        images : np.array
        Returns
        -------
        actions, pose, images : tuple of np.array
            cleaned data
        """
        actions_copy = actions.copy()
        if arm == "bimanual":
            actions_left = actions_copy[..., :3]
            actions_right = actions_copy[..., 6:9]
            actions_copy = np.concatenate((actions_left, actions_right), axis=-1)
        else:
            actions_copy = actions_copy[..., :3]

        ac_dim = actions_copy.shape[-1]
        actions_flat = actions_copy.reshape(-1, 3)

        N, C, H, W = images.shape

        if H == 480:
            intrinsics = INTRINSICS["base"]
        elif H == 240:
            intrinsics = INTRINSICS["base_half"]
        px = cam_frame_to_cam_pixels(actions_flat, intrinsics)
        px = px.reshape((-1, CHUNK_LENGTH, ac_dim))
        if ac_dim == 3:
            bad_data_mask = (
                (px[:, :, 0] < 0)
                | (px[:, :, 0] > (W))
                | (px[:, :, 1] < 0)
                | (px[:, :, 1] > (H))
            )
        elif ac_dim == 6:
            BUFFER = 0
            bad_data_mask = (
                (px[:, :, 0] < 0 - BUFFER)
                | (px[:, :, 0] > (W) + BUFFER)
                | (px[:, :, 1] < 0)
                # | (px[:, :, 1] > 480 + BUFFER)
                | (px[:, :, 3] < 0 - BUFFER)
                | (px[:, :, 3] > (H) + BUFFER)
                | (px[:, :, 4] < 0)
                # | (px[:, :, 4] > 480 + BUFFER)
            )

            px_diff = np.diff(px, axis=1)
            px_diff = np.concatenate(
                (px_diff, np.zeros((px_diff.shape[0], 1, px_diff.shape[-1]))), axis=1
            )
            px_diff = np.abs(px_diff)
            bad_data_mask = bad_data_mask | np.any(px_diff > 100, axis=2)

        bad_data_mask = np.any(bad_data_mask, axis=1)

        actions = actions[~bad_data_mask]
        images = images[~bad_data_mask]
        pose = pose[~bad_data_mask]

        return actions, pose, images

    @staticmethod
    def get_images(
        vrs_reader,
        stream_ids: dict,
        stream_timestamps_ns: dict,
        height=480,
        width=640,
    ):
        """
        Get RGB Image from VRS
        Parameters
        ----------
        vrs_reader : VRS Data Provider
            Object that reads and obtains data from VRS
        stream_ids : dict
            maps sensor keys to a list of ids for Aria
        stream_timestamps_ns : dict
            dict that maps sensor keys to a list of nanosecond timestamps in device time
        Returns
        -------
        images : np.array
            rgb images undistorted to 480x640x3
        """
        images = []
        frame_length = len(stream_timestamps_ns["rgb"])

        time_domain = TimeDomain.DEVICE_TIME
        time_query_closest = TimeQueryOptions.CLOSEST

        for t in range(frame_length):
            query_timestamp = stream_timestamps_ns["rgb"][t]

            sample_frame = vrs_reader.get_image_data_by_time_ns(
                stream_ids["rgb"],
                query_timestamp,
                time_domain,
                time_query_closest,
            )

            image_t = undistort_to_linear(
                vrs_reader,
                stream_ids,
                raw_image=sample_frame[0].to_numpy_array(),
                height=height,
                width=width,
            )

            images.append(image_t)
        images = np.array(images)
        return images

    @staticmethod
    def get_hand_keypoints(
        world_device_T,
        stream_timestamps_ns: dict,
        hand_tracking_results,
        arm: str,
    ):
        """
        Get Hand Keypoints from VRS
        Parameters
        ----------
        world_device_T : np.array
            Transform from world coordinates to ARIA camera frame
        stream_timestamps_ns : dict
        hand_tracking_results : dict
        arm : str
            arm to get hand keypoints for
        Returns
        -------
        hand_keypoints : np.array
            hand_keypoints
        """
        frame_length = len(stream_timestamps_ns["rgb"])

        keypoints = []

        use_left_hand = arm == "left" or arm == "bimanual"
        use_right_hand = arm == "right" or arm == "bimanual"
        for t in range(frame_length):
            query_timestamp = stream_timestamps_ns["rgb"][t]
            hand_tracking_result_t = get_nearest_hand_tracking_result(
                hand_tracking_results, query_timestamp
            )
            world_device_T_t = get_nearest_pose(world_device_T, query_timestamp)
            if world_device_T_t is not None:
                world_device_T_t = world_device_T_t.transform_world_device

            right_confidence = getattr(
                getattr(hand_tracking_result_t, "right_hand", None), "confidence", -1
            )
            left_confidence = getattr(
                getattr(hand_tracking_result_t, "left_hand", None), "confidence", -1
            )
            left_obs_t = np.full(7 + 21 * 3, 1e9)
            if (
                use_left_hand
                and not left_confidence < 0
                and world_device_T_t is not None
            ):
                left_hand_keypoints = np.stack(
                    hand_tracking_result_t.left_hand.landmark_positions_device, axis=0
                )
                wrist_T = (
                    hand_tracking_result_t.left_hand.transform_device_wrist
                )  # Sophus SE3

                world_wrist_T = world_device_T_t @ wrist_T
                world_keypoints = (
                    world_device_T_t @ left_hand_keypoints.T
                ).T  # keypoints are in device frame

                world_wrist_T = sp.SE3.from_matrix(
                    T_rot_orientation(world_wrist_T.to_matrix(), T_ROT_CAM)
                )
                wrist_quat_and_translation = quat_translation_swap(
                    world_wrist_T.to_quat_and_translation()
                )
                if wrist_quat_and_translation.ndim == 2:
                    wrist_quat_and_translation = wrist_quat_and_translation[0]
                left_obs_t[:7] = wrist_quat_and_translation
                left_obs_t[7:] = world_keypoints.flatten()

            right_obs_t = np.full(7 + 21 * 3, 1e9)
            if (
                use_right_hand
                and not right_confidence < 0
                and world_device_T_t is not None
            ):
                right_hand_keypoints = np.stack(
                    hand_tracking_result_t.right_hand.landmark_positions_device, axis=0
                )
                wrist_T = (
                    hand_tracking_result_t.right_hand.transform_device_wrist
                )  # Sophus SE3

                world_wrist_T = world_device_T_t @ wrist_T
                world_keypoints = (
                    world_device_T_t @ right_hand_keypoints.T
                ).T  # keypoints are in device frame

                world_wrist_T = sp.SE3.from_matrix(
                    T_rot_orientation(world_wrist_T.to_matrix(), T_ROT_CAM)
                )
                wrist_quat_and_translation = quat_translation_swap(
                    world_wrist_T.to_quat_and_translation()
                )
                if wrist_quat_and_translation.ndim == 2:
                    wrist_quat_and_translation = wrist_quat_and_translation[0]
                right_obs_t[:7] = wrist_quat_and_translation
                right_obs_t[7:] = world_keypoints.flatten()

            if use_left_hand and use_right_hand:
                keypoints_obs_t = np.concatenate((left_obs_t, right_obs_t), axis=-1)
            elif use_left_hand:
                keypoints_obs_t = left_obs_t
            elif use_right_hand:
                keypoints_obs_t = right_obs_t
            else:
                raise ValueError(f"Incorrect arm provided: {arm}")
            keypoints.append(np.ravel(keypoints_obs_t))
        keypoints = np.array(keypoints)
        return keypoints

    @staticmethod
    def get_head_pose(
        world_device_T,
        device_rgb_T,
        stream_timestamps_ns: dict,
    ):
        """
        Get Head Pose from VRS
        Parameters
        ----------
        world_device_T : np.array
            Transform from world coordinates to ARIA camera frame
        stream_timestamps_ns : dict
            dict that maps sensor keys to a list of nanosecond timestamps in device time

        Returns
        -------
        head_pose : np.array
            head_pose
        """
        head_pose = []
        frame_length = len(stream_timestamps_ns["rgb"])

        rgb_to_rgbprime_rot = np.eye(4)
        rgb_to_rgbprime_rot[:3, :3] = ROTATION_MATRIX.T
        rgb_to_rgbprime_T = sp.SE3.from_matrix(rgb_to_rgbprime_rot)
        rgbprime_to_rgb_T = rgb_to_rgbprime_T.inverse()
        for t in range(frame_length):
            query_timestamp = stream_timestamps_ns["rgb"][t]
            world_device_T_t = get_nearest_pose(world_device_T, query_timestamp)
            if world_device_T_t is not None:
                world_device_T_t = world_device_T_t.transform_world_device
            head_pose_obs_t = np.full(7, 1e9)
            if world_device_T_t is not None:
                world_rgb_T_t = world_device_T_t @ device_rgb_T @ rgbprime_to_rgb_T
                head_pose_quat_and_translation = quat_translation_swap(
                    world_rgb_T_t.to_quat_and_translation()
                )
                if head_pose_quat_and_translation.ndim == 2:
                    head_pose_quat_and_translation = head_pose_quat_and_translation[0]
                head_pose_obs_t[:7] = head_pose_quat_and_translation
            head_pose.append(np.ravel(head_pose_obs_t))
        head_pose = np.array(head_pose)
        return head_pose

    @staticmethod
    def get_ee_pose(
        world_device_T,
        stream_timestamps_ns: dict,
        hand_tracking_results,
        arm: str,
    ):
        """
        Get EE Pose from VRS
        Parameters
        ----------
        world_device_T : np.array
            Transform from world coordinates to ARIA camera frame
        stream_timestamps_ns : dict
            dict that maps sensor keys to a list of nanosecond timestamps in device time
        hand_tracking_results : dict
            dict that maps sensor keys to a list of hand tracking results
        arm : str
            arm to get hand keypoints for
        Returns
        -------
        ee_pose : np.array
            ee_pose (6D per arm)
            -1 if no hand tracking data is available
        """
        ee_pose = []
        frame_length = len(stream_timestamps_ns["rgb"])

        use_left_hand = arm == "left" or arm == "bimanual"
        use_right_hand = arm == "right" or arm == "bimanual"

        for t in range(frame_length):
            query_timestamp = stream_timestamps_ns["rgb"][t]
            hand_tracking_result_t = get_nearest_hand_tracking_result(
                hand_tracking_results, query_timestamp
            )
            world_device_T_t = get_nearest_pose(world_device_T, query_timestamp)
            if world_device_T_t is not None:
                world_device_T_t = world_device_T_t.transform_world_device

            right_confidence = getattr(
                getattr(hand_tracking_result_t, "right_hand", None), "confidence", -1
            )
            left_confidence = getattr(
                getattr(hand_tracking_result_t, "left_hand", None), "confidence", -1
            )

            left_obs_t = np.full(7, 1e9)
            if (
                use_left_hand
                and not left_confidence < 0
                and world_device_T_t is not None
            ):
                left_palm_pose = (
                    hand_tracking_result_t.left_hand.get_palm_position_device()
                )
                left_wrist_pose = (
                    hand_tracking_result_t.left_hand.get_wrist_position_device()
                )
                left_palm_normal = hand_tracking_result_t.left_hand.wrist_and_palm_normal_device.palm_normal_device

                left_rot_matrix = compute_orientation_rotation_matrix(
                    palm_pose=left_palm_pose,
                    wrist_pose=left_wrist_pose,
                    palm_normal=left_palm_normal,
                )
                left_T_t = np.eye(4)
                left_T_t[:3, :3] = left_rot_matrix
                left_T_t[:3, 3] = left_palm_pose
                left_T_t = sp.SE3.from_matrix(left_T_t)
                left_T_t = world_device_T_t @ left_T_t
                left_T_t = sp.SE3.from_matrix(
                    T_rot_orientation(left_T_t.to_matrix(), T_ROT_CAM)
                )

                left_quat_and_translation = quat_translation_swap(
                    left_T_t.to_quat_and_translation()
                )
                if left_quat_and_translation.ndim == 2:
                    left_quat_and_translation = left_quat_and_translation[0]
                left_obs_t[:7] = left_quat_and_translation

            right_obs_t = np.full(7, 1e9)
            if (
                use_right_hand
                and not right_confidence < 0
                and world_device_T_t is not None
            ):
                right_palm_pose = (
                    hand_tracking_result_t.right_hand.get_palm_position_device()
                )
                right_wrist_pose = (
                    hand_tracking_result_t.right_hand.get_wrist_position_device()
                )
                right_palm_normal = hand_tracking_result_t.right_hand.wrist_and_palm_normal_device.palm_normal_device

                right_rot_matrix = compute_orientation_rotation_matrix(
                    palm_pose=right_palm_pose,
                    wrist_pose=right_wrist_pose,
                    palm_normal=right_palm_normal,
                )
                right_T_t = np.eye(4)
                right_T_t[:3, :3] = right_rot_matrix
                right_T_t[:3, 3] = right_palm_pose
                right_T_t = sp.SE3.from_matrix(right_T_t)
                right_T_t = world_device_T_t @ right_T_t
                right_T_t = sp.SE3.from_matrix(
                    T_rot_orientation(right_T_t.to_matrix(), T_ROT_CAM)
                )
                right_quat_and_translation = quat_translation_swap(
                    right_T_t.to_quat_and_translation()
                )
                if right_quat_and_translation.ndim == 2:
                    right_quat_and_translation = right_quat_and_translation[0]
                right_obs_t[:7] = right_quat_and_translation

            if use_left_hand and use_right_hand:
                ee_pose_obs_t = np.concatenate((left_obs_t, right_obs_t), axis=-1)
            elif use_left_hand:
                ee_pose_obs_t = left_obs_t
            elif use_right_hand:
                ee_pose_obs_t = right_obs_t
            else:
                raise ValueError(f"Incorrect arm provided: {arm}")
            ee_pose.append(np.ravel(ee_pose_obs_t))
        ee_pose = np.array(ee_pose)
        return ee_pose

    @staticmethod
    def get_cameras(rgb_camera_key: str):
        """
        Returns a list of rgb keys
        Parameters
        ----------
        rgb_camera_key : str

        Returns
        -------
        rgb_cameras : list of str
            A list of keys corresponding to rgb_cameras in the dataset.
        """

        rgb_cameras = [rgb_camera_key]
        return rgb_cameras

    @staticmethod
    def get_state(state_key: str):
        """
        Returns a list of state keys
        Parameters
        ----------
        state_key : str

        Returns
        -------
        states : list of str
            A list of keys corresponding to states in the dataset.
        """

        states = [state_key]
        return states

    @staticmethod
    def define_features(
        episode_feats: dict, image_compressed: bool = True, encode_as_video: bool = True
    ) -> tuple:
        """
        Define features from episode_feats (output of process_episode), including a metadata section.

        Parameters
        ----------
        episode_feats : dict
            The output of the process_episode method, containing feature data.
        image_compressed : bool, optional
            Whether the images are compressed, by default True.
        encode_as_video : bool, optional
            Whether to encode images as video or as images, by default True.

        Returns
        -------
        tuple of dict[str, dict]
            A dictionary where keys are feature names and values are dictionaries
            containing feature information such as dtype, shape, and dimension names,
            and a separate dictionary for metadata (unused for now)
        """
        features = {}
        metadata = {}
        for key, value in episode_feats.items():
            if isinstance(value, dict):  # Handle nested dictionaries recursively
                nested_features, nested_metadata = AriaVRSExtractor.define_features(
                    value, image_compressed, encode_as_video
                )
                features.update(
                    {
                        f"{key}.{nested_key}": nested_value
                        for nested_key, nested_value in nested_features.items()
                    }
                )
                features.update(
                    {
                        f"{key}.{nested_key}": nested_value
                        for nested_key, nested_value in nested_metadata.items()
                    }
                )
            elif isinstance(value, np.ndarray):
                dtype = str(value.dtype)
                if "images" in key:
                    dtype = "video" if encode_as_video else "image"
                    if image_compressed:
                        decompressed_sample = cv2.imdecode(value[0], 1)
                        shape = (
                            decompressed_sample.shape[1],
                            decompressed_sample.shape[0],
                            decompressed_sample.shape[2],
                        )
                    else:
                        shape = value.shape[1:]  # Skip the frame count dimension
                    dim_names = ["channel", "height", "width"]
                elif "actions" in key and len(value[0].shape) > 1:
                    shape = value[0].shape
                    dim_names = ["chunk_length", "action_dim"]
                    dtype = f"prestacked_{str(value.dtype)}"
                else:
                    shape = value[0].shape
                    dim_names = [f"dim_{i}" for i in range(len(shape))]
                features[key] = {
                    "dtype": dtype,
                    "shape": shape,
                    "names": dim_names,
                }
            elif isinstance(value, torch.Tensor):
                dtype = str(value.dtype)
                if "actions" in key and len(tuple(value[0].size())) > 1:
                    dim_names = ["chunk_length", "action_dim"]
                    dtype = f"prestacked_{str(value.dtype)}"
                else:
                    dim_names = [f"dim_{i}" for i in range(len(shape))]
                shape = tuple(value[0].size())
                dim_names = [f"dim_{i}" for i in range(len(shape))]
                features[key] = {
                    "dtype": dtype,
                    "shape": shape,
                    "names": dim_names,
                }
            else:
                metadata[key] = {
                    "dtype": "metadata",
                    "value": value,
                }

        return features, metadata
