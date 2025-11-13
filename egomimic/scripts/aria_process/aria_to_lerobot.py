import argparse
import logging
import os
from pathlib import Path
import shutil
import traceback
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torch
from enum import Enum

from egomimic.utils.egomimicUtils import (
    str2bool,
    cam_frame_to_cam_pixels,
    INTRINSICS,
    interpolate_keys,
    interpolate_arr,
    interpolate_arr_euler,
    transform_to_pose,
    pose_to_transform,
)

from projectaria_tools.core.calibration import CameraCalibration, DeviceCalibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

from projectaria_tools.core import data_provider, mps

from projectaria_tools.core.mps.utils import get_nearest_hand_tracking_result

from projectaria_tools.core.mps.utils import (
    filter_points_from_confidence,
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze,
    get_nearest_pose,
)
from projectaria_tools.core.stream_id import StreamId

from aria_utils import (
    build_camera_matrix,
    undistort_to_linear,
    slam_to_rgb,
    compute_coordinate_frame,
    transform_coordinates,
    coordinate_frame_to_ypr,
)

from egomimic.rldb.utils import EMBODIMENT

import time

import numpy as np

import torch
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R


## CHANGE THIS TO YOUR DESIRED CACHE FOR HF
os.environ["HF_HOME"] = "~/.cache/huggingface"

HORIZON_DEFAULT = 10
STEP_DEFAULT = 3.0
EPISODE_LENGTH = 100
CHUNK_LENGTH_ACT = 100

ROTATION_MATRIX = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])


# NOTE: Replaced by transform ee_pose
# def transform_actions(actions):
#     if actions.shape[-1] == 3:
#         actions[..., 0] *= -1  # Multiply x by -1
#         actions[..., 1] *= -1  # Multiply y by -1
#     elif actions.shape[-1] == 6:
#         actions[..., 0] *= -1  # Multiply x by -1 for first set
#         actions[..., 1] *= -1  # Multiply y by -1 for first set
#         actions[..., 3] *= -1  # Multiply x by -1 for second set
#         actions[..., 4] *= -1  # Multiply y by -1 for second set
#     return actions


def compute_camera_relative_pose(pose, cam_t_inv, cam_offset):
    """
    pose (6,) : np.array
        x y z y p r
    cam_t_inv (4, 4) : np.array
        camera intrinsics inverse of timestep t
    cam_offset (4, 4) : np.array
        camera intrinsics of offset

    returns pose_t (6,) : np.array
        future pose in camera t frame x y z y p r
    """
    T_offset_pose = pose_to_transform(pose)
    undo_rotation = np.eye(4)
    undo_rotation[:3, :3] = ROTATION_MATRIX

    T_unrotated = undo_rotation @ T_offset_pose
    T_world = np.dot(cam_offset, T_unrotated)
    T_camera = np.dot(cam_t_inv, T_world)

    redo_rotation = np.eye(4)
    redo_rotation[:3, :3] = ROTATION_MATRIX.T
    T_final = redo_rotation @ T_camera

    pose_t = transform_to_pose(T_final)
    return pose_t


def get_hand_pose_in_camera_frame(hand_data, cam_t_inv, cam_offset, transform):
    """
    Process a single hand's data to compute the 6-dof pose in the camera-t frame.

    Args:
        hand_data: hand data from mps:
            - palm_position_device
            - wrist_position_device
            - wrist_and_palm_normal_device.palm_normal_device
        cam_t_inv (np.ndarray): Inverse transformation matrix for the camera at timestep t.
        cam_offset (np.ndarray): Transformation matrix for the camera offset.
        transform: The transform used in transform_coordinates.

    Returns:
        np.ndarray: 6-dof pose (translation + Euler angles) in the camera-t frame.
                    Returns np.full(6, 1e9) if the palm position is not detected.
    """
    if hand_data is None or not np.any(hand_data.get_palm_position_device()):
        return np.full(6, 1e9)

    palm_pose = hand_data.get_palm_position_device()
    wrist_pose = hand_data.get_wrist_position_device()
    palm_normal = hand_data.wrist_and_palm_normal_device.palm_normal_device

    if hand_data.confidence < 0:
        pose_offset = np.full(6, 1e9)
        return pose_offset

    x_axis, y_axis, z_axis = compute_coordinate_frame(
        palm_pose=palm_pose, wrist_pose=wrist_pose, palm_normal=palm_normal
    )

    palm_pose, x, y, z = transform_coordinates(
        palm_pose=palm_pose,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        transform=transform,
    )

    palm_euler = coordinate_frame_to_ypr(x, y, z)
    pose_offset = np.concatenate((palm_pose, palm_euler), axis=None)

    pose_offset_in_camera_t = compute_camera_relative_pose(
        pose_offset, cam_t_inv=cam_t_inv, cam_offset=cam_offset
    )
    return pose_offset_in_camera_t


class AriaVRSExtractor:
    TAGS = ["aria", "robotics", "vrs"]

    @staticmethod
    def process_episode(episode_path, arm, prestack=False, low_res=False):
        """
        Extracts all feature keys from a given episode and returns as a dictionary
        Parameters
        ----------
        episode_path : str or Path
            Path to the VRS file containing the episode data.
        arm : str
            String for which arm to add data for
        prestack : bool
            prestack the future actions or not
        Returns
        -------
        episode_feats : dict
            dictionary mapping keys in the episode to episode features
            {
                {action_key} :
                observations :
                    images.{camera_key} :
                    state.{state_key} :
            }

            #TODO: Add metadata to be a nested dict

        """
        episode_feats = dict()

        # file setup and opening
        filename = episode_path.name
        root_dir = episode_path.parent

        mps_sample_path = os.path.join(root_dir, ("mps_" + episode_path.stem + "_vrs"))

        hand_tracking_results_path = os.path.join(
            mps_sample_path, "hand_tracking", "hand_tracking_results.csv"
        )

        vrs_reader = data_provider.create_vrs_data_provider(str(episode_path))

        hand_tracking_results = mps.hand_tracking.read_hand_tracking_results(
            hand_tracking_results_path
        )

        device_calibration = vrs_reader.get_device_calibration()

        time_domain: TimeDomain = TimeDomain.DEVICE_TIME
        time_query_closest: TimeQueryOptions = TimeQueryOptions.CLOSEST

        stream_ids: Dict[str, StreamId] = {
            "rgb": StreamId("214-1"),
            "slam-left": StreamId("1201-1"),
            "slam-right": StreamId("1201-2"),
        }
        stream_labels: Dict[str, str] = {
            key: vrs_reader.get_label_from_stream_id(stream_id)
            for key, stream_id in stream_ids.items()
        }
        stream_timestamps_ns: Dict[str, List[int]] = {
            key: vrs_reader.get_timestamps_ns(stream_id, time_domain)
            for key, stream_id in stream_ids.items()
        }

        mps_data_paths_provider = mps.MpsDataPathsProvider(mps_sample_path)
        mps_data_paths = mps_data_paths_provider.get_data_paths()
        mps_reader = mps.MpsDataProvider(mps_data_paths)

        transform = slam_to_rgb(vrs_reader).to_matrix()
        episode_feats["observations"] = dict()

        # ee_pose
        # TODO: this will be useful for the future - when we add rotation and other state keys
        state_key = AriaVRSExtractor.get_state("ee_pose")[0]

        pose = AriaVRSExtractor.get_ee_pose(
            mps_reader=mps_reader,
            transform=transform,
            arm=arm,
            stream_timestamps_ns=stream_timestamps_ns,
            hand_tracking_results=hand_tracking_results,
        )

        # rgb_camera
        # TODO: this will be useful for the future - when we add other camera modalities
        camera_key = AriaVRSExtractor.get_cameras("front_img_1")[0]

        images = AriaVRSExtractor.get_images(
            vrs_reader=vrs_reader,
            stream_ids=stream_ids,
            stream_timestamps_ns=stream_timestamps_ns,
        )

        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()

        if low_res:
            images = F.interpolate(
                images, size=(240, 320), mode="bilinear", align_corners=False
            )

        images = images.byte().numpy()

        # actions
        actions = AriaVRSExtractor.get_action(
            pose=pose,
            mps_reader=mps_reader,
            vrs_reader=vrs_reader,
            stream_timestamps_ns=stream_timestamps_ns,
            transform=transform,
            arm=arm,
            prestack=prestack,
            hand_tracking_results=hand_tracking_results,
        )

        print(f"[DEBUG] LENGTH BEFORE CLEANING: {len(actions)}")
        actions, pose, images = AriaVRSExtractor.clean_data(
            actions=actions, pose=pose, images=images, arm=arm
        )
        # actions, pose, images = AriaVRSExtractor.clean_data_projection(actions=actions, pose=pose, images=images, arm=arm)
        print(f"[DEBUG] LENGTH AFTER CLEANING: {len(actions)}")

        episode_feats["observations"][f"state.{state_key}"] = pose
        episode_feats["observations"][f"images.{camera_key}"] = images
        episode_feats["actions_cartesian"] = actions

        num_timesteps = episode_feats["observations"][f"state.ee_pose"].shape[0]
        if arm == "right":
            value = EMBODIMENT.ARIA_RIGHT_ARM.value
        elif arm == "left":
            value = EMBODIMENT.ARIA_LEFT_ARM.value
        else:
            value = EMBODIMENT.ARIA_BIMANUAL.value

        episode_feats["metadata.embodiment"] = np.full(
            (num_timesteps, 1), value, dtype=np.int32
        )

        return episode_feats

    @staticmethod
    def get_action(
        pose: np.array,
        mps_reader,
        vrs_reader,
        stream_timestamps_ns: dict,
        transform: np.array,
        arm: str,
        hand_tracking_results,
        HORIZON=HORIZON_DEFAULT,
        STEP=STEP_DEFAULT,
        prestack=False,
        no_rot=False,
    ):
        """
        Calculates actions using stable reference frames
        Parameters
        ----------
        pose : np.array
            hand poses
        mps_reader : MPS Data Provider
            Object that reads and obtains data from VRS
        vrs_reader : VRS Data Provider
            Object that reads and obtains data from VRS
        stream_timestamps_ns : dict
            dict that maps sensor keys to a list of nanosecond timestamps in device time
        HORIZON : int
            number of timesteps in future
        STEP : int
            number of steps to interpolate
        prestack : bool
            precompute action chunks
        no_rot : bool
            Not Implemented yet
        Returns
        -------
        actions : np.array
        """
        frame_length = len(stream_timestamps_ns["rgb"])
        actions = []

        time_domain: TimeDomain = TimeDomain.DEVICE_TIME
        time_query_closest: TimeQueryOptions = TimeQueryOptions.CLOSEST

        for t in range(frame_length - int(HORIZON * STEP)):
            query_timestamp = stream_timestamps_ns["rgb"][t]
            head_pose_t = mps_reader.get_closed_loop_pose(
                query_timestamp, time_query_closest
            )
            camera_matrix = build_camera_matrix(vrs_reader, head_pose_t)
            camera_t_inv = np.linalg.inv(camera_matrix)

            actions_t = []

            for offset in range(HORIZON):
                sample_timestamp_ns = stream_timestamps_ns["rgb"][
                    int(t + offset * STEP)
                ]
                hand_tracking_result_offset = get_nearest_hand_tracking_result(
                    hand_tracking_results, sample_timestamp_ns
                )

                if hand_tracking_result_offset is None:
                    left_hand, right_hand = None, None
                else:
                    left_hand = hand_tracking_result_offset.left_hand
                    right_hand = hand_tracking_result_offset.right_hand

                head_pose_offset = mps_reader.get_closed_loop_pose(
                    sample_timestamp_ns, time_query_closest
                )
                camera_matrix_offset = build_camera_matrix(vrs_reader, head_pose_offset)
                if arm == "right":
                    right_pose_in_camera_t = get_hand_pose_in_camera_frame(
                        right_hand,
                        cam_t_inv=camera_t_inv,
                        cam_offset=camera_matrix_offset,
                        transform=transform,
                    )
                    actions_t.append(right_pose_in_camera_t)
                elif arm == "left":
                    left_pose_in_camera_t = get_hand_pose_in_camera_frame(
                        left_hand,
                        cam_t_inv=camera_t_inv,
                        cam_offset=camera_matrix_offset,
                        transform=transform,
                    )
                    actions_t.append(left_pose_in_camera_t)
                elif arm == "bimanual":
                    # Process left hand.
                    left_pose_in_camera_t = get_hand_pose_in_camera_frame(
                        left_hand,
                        cam_t_inv=camera_t_inv,
                        cam_offset=camera_matrix_offset,
                        transform=transform,
                    )

                    # Process right hand.
                    right_pose_in_camera_t = get_hand_pose_in_camera_frame(
                        right_hand,
                        cam_t_inv=camera_t_inv,
                        cam_offset=camera_matrix_offset,
                        transform=transform,
                    )
                    combined_pose = np.concatenate(
                        [left_pose_in_camera_t, right_pose_in_camera_t], axis=None
                    )
                    actions_t.append(combined_pose)

            actions_t = np.array(actions_t)
            actions.append(actions_t)

        actions = np.array(actions)

        if arm == "bimanual":
            actions_left = actions[..., :6]
            actions_right = actions[..., 6:]
            actions_left = interpolate_arr_euler(actions_left, CHUNK_LENGTH_ACT)
            actions_right = interpolate_arr_euler(actions_right, CHUNK_LENGTH_ACT)
            actions = np.concatenate((actions_left, actions_right), axis=-1)
        else:
            actions = interpolate_arr_euler(actions, CHUNK_LENGTH_ACT)

        if not prestack:
            actions = actions[:, 1, :]

        if no_rot:
            if arm == "bimanual":
                actions_left = actions[..., :3]
                actions_right = actions[..., 6:9]
                actions = np.concatenate((actions_left, actions_right), axis=-1)
            else:
                actions = actions[..., :3]
        return actions

    @staticmethod
    def clean_data(actions, pose, images, arm):
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
        bad_data_mask = np.any(pose >= 1e8, axis=1)

        actions = actions[~bad_data_mask]
        pose = pose[~bad_data_mask]
        images = images[~bad_data_mask]

        bad_data_mask = np.any(actions >= 1e8, axis=(1, 2))

        actions = actions[~bad_data_mask]
        pose = pose[~bad_data_mask]
        images = images[~bad_data_mask]

        return actions, pose, images

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
        HORIZON=HORIZON_DEFAULT,
        STEP=STEP_DEFAULT,
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

        for t in range(frame_length - int(HORIZON * STEP)):
            query_timestamp = stream_timestamps_ns["rgb"][t]

            sample_frame = vrs_reader.get_image_data_by_time_ns(
                stream_ids["rgb"],
                query_timestamp,
                time_domain,
                time_query_closest,
            )

            image_t = undistort_to_linear(
                vrs_reader, stream_ids, raw_image=sample_frame[0].to_numpy_array()
            )

            images.append(image_t)

        images = np.array(images)
        return images

    @staticmethod
    def get_ee_pose(
        mps_reader,
        transform: np.array,
        arm: str,
        stream_timestamps_ns: dict,
        hand_tracking_results,
        no_rot=False,
        HORIZON=HORIZON_DEFAULT,
        STEP=STEP_DEFAULT,
    ):
        """
        Get EE Pose from VRS
        Parameters
        ----------
        mps_reader : MPS Data Provider
            Object that reads and obtains data from MPS
        transform : np.array
            Transform from world coordinates to ARIA camera frame
        arm : str
            Which arm for data
        stream_timestamps_ns : dict
            dict that maps sensor keys to a list of nanosecond timestamps in device time
        no_rot :
            Whether or not to compute rotation (#TODO: Incomplete for now)

        Returns
        -------
        ee_pose : np.array
            ee_pose SE{3}
        """
        ee_pose = []
        frame_length = len(stream_timestamps_ns["rgb"])

        time_domain = TimeDomain.DEVICE_TIME
        time_query_closest = TimeQueryOptions.CLOSEST

        for t in range(frame_length - int(HORIZON * STEP)):
            query_timestamp = stream_timestamps_ns["rgb"][t]
            hand_tracking_result_t = get_nearest_hand_tracking_result(
                hand_tracking_results, query_timestamp
            )
            right_confidence = getattr(
                getattr(hand_tracking_result_t, "right_hand", None), "confidence", -1
            )
            left_confidence = getattr(
                getattr(hand_tracking_result_t, "left_hand", None), "confidence", -1
            )
            if arm == "right":
                ee_pose_obs_t = np.full(6, 1e9)
                if not right_confidence < 0:
                    right_palm_pose = (
                        hand_tracking_result_t.right_hand.get_palm_position_device()
                    )
                    right_wrist_pose = (
                        hand_tracking_result_t.right_hand.get_wrist_position_device()
                    )
                    right_palm_normal = hand_tracking_result_t.right_hand.wrist_and_palm_normal_device.palm_normal_device

                    right_coordinates = compute_coordinate_frame(
                        palm_pose=right_palm_pose,
                        wrist_pose=right_wrist_pose,
                        palm_normal=right_palm_normal,
                    )
                    right_x_axis, right_y_axis, right_z_axis = right_coordinates
                    right_palm_pose, right_x, right_y, right_z = transform_coordinates(
                        palm_pose=right_palm_pose,
                        x_axis=right_x_axis,
                        y_axis=right_y_axis,
                        z_axis=right_z_axis,
                        transform=transform,
                    )
                    right_palm_euler = coordinate_frame_to_ypr(
                        right_x, right_y, right_z
                    )
                    ee_pose_obs_t = np.concatenate(
                        (right_palm_pose, right_palm_euler), axis=None
                    )
            elif arm == "left":
                ee_pose_obs_t = np.full(6, 1e9)
                if not left_confidence < 0:
                    left_palm_pose = (
                        hand_tracking_result_t.left_hand.get_palm_position_device()
                    )
                    left_wrist_pose = (
                        hand_tracking_result_t.left_hand.get_wrist_position_device()
                    )
                    left_palm_normal = hand_tracking_result_t.left_hand.wrist_and_palm_normal_device.palm_normal_device

                    left_coordinates = compute_coordinate_frame(
                        palm_pose=left_palm_pose,
                        wrist_pose=left_wrist_pose,
                        palm_normal=left_palm_normal,
                    )
                    left_x_axis, left_y_axis, left_z_axis = left_coordinates
                    left_palm_pose, left_x, left_y, left_z = transform_coordinates(
                        palm_pose=left_palm_pose,
                        x_axis=left_x_axis,
                        y_axis=left_y_axis,
                        z_axis=left_z_axis,
                        transform=transform,
                    )

                    left_palm_euler = coordinate_frame_to_ypr(left_x, left_y, left_z)
                    ee_pose_obs_t = np.concatenate(
                        (left_palm_pose, left_palm_euler), axis=None
                    )
            elif arm == "bimanual":
                left_obs_t = np.full(6, 1e9)
                if not left_confidence < 0:
                    left_palm_pose = (
                        hand_tracking_result_t.left_hand.get_palm_position_device()
                    )
                    left_wrist_pose = (
                        hand_tracking_result_t.left_hand.get_wrist_position_device()
                    )
                    left_palm_normal = hand_tracking_result_t.left_hand.wrist_and_palm_normal_device.palm_normal_device

                    left_coordinates = compute_coordinate_frame(
                        palm_pose=left_palm_pose,
                        wrist_pose=left_wrist_pose,
                        palm_normal=left_palm_normal,
                    )
                    left_x_axis, left_y_axis, left_z_axis = left_coordinates
                    left_palm_pose, left_x, left_y, left_z = transform_coordinates(
                        palm_pose=left_palm_pose,
                        x_axis=left_x_axis,
                        y_axis=left_y_axis,
                        z_axis=left_z_axis,
                        transform=transform,
                    )

                    left_palm_euler = coordinate_frame_to_ypr(left_x, left_y, left_z)
                    left_obs_t = np.concatenate(
                        (left_palm_pose, left_palm_euler), axis=None
                    )

                right_obs_t = np.full(6, 1e9)
                if not right_confidence < 0:
                    right_palm_pose = (
                        hand_tracking_result_t.right_hand.get_palm_position_device()
                    )
                    right_wrist_pose = (
                        hand_tracking_result_t.right_hand.get_wrist_position_device()
                    )
                    right_palm_normal = hand_tracking_result_t.right_hand.wrist_and_palm_normal_device.palm_normal_device

                    right_coordinates = compute_coordinate_frame(
                        palm_pose=right_palm_pose,
                        wrist_pose=right_wrist_pose,
                        palm_normal=right_palm_normal,
                    )
                    right_x_axis, right_y_axis, right_z_axis = right_coordinates
                    right_palm_pose, right_x, right_y, right_z = transform_coordinates(
                        palm_pose=right_palm_pose,
                        x_axis=right_x_axis,
                        y_axis=right_y_axis,
                        z_axis=right_z_axis,
                        transform=transform,
                    )

                    right_palm_euler = coordinate_frame_to_ypr(
                        right_x, right_y, right_z
                    )
                    right_obs_t = np.concatenate(
                        (right_palm_pose, right_palm_euler), axis=None
                    )

                ee_pose_obs_t = np.concatenate((left_obs_t, right_obs_t), axis=None)
            else:
                print(f"[WARNING]: INCORRECT ARM PROVIDED : {arm}")
            ee_pose.append(np.ravel(ee_pose_obs_t))
        ee_pose = np.array(ee_pose)
        if no_rot:
            if arm == "bimanual":
                ee_pose_left = ee_pose[..., :3]
                ee_pose_right = ee_pose[..., 6:9]
                ee_pose = np.concatenate((ee_pose_left, ee_pose_right), axis=-1)
            else:
                ee_pose = ee_pose[..., :3]
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

    # TODO don't touch this and just modify previous methods
    @staticmethod
    def extract_episode_frames(
        episode_path: str | Path,
        features: dict[str, dict],
        image_compressed: bool,
        arm: str,
        prestack: bool = False,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Extract frames from an episode by processing it and using the feature dictionary.

        Parameters
        ----------
        episode_path : str or Path
            Path to the HDF5 file containing the episode data.
        features : dict of str to dict
            Dictionary where keys are feature identifiers and values are dictionaries with feature details.
        image_compressed : bool
            Flag indicating whether the images are stored in a compressed format.
        arm : str
            The arm to process (e.g., 'left', 'right', or 'bimanual').
        prestack : bool, optional
            Whether to precompute action chunks, by default False.

        Returns
        -------
        list[dict[str, torch.Tensor]]
            List of frames, where each frame is a dictionary mapping feature identifiers to tensors.
        """
        frames = []
        episode_feats = AriaVRSExtractor.process_episode(
            episode_path, arm=arm, prestack=prestack
        )
        num_frames = next(iter(episode_feats["observations"].values())).shape[0]
        for frame_idx in range(num_frames):
            frame = {}
            for feature_id, feature_info in features.items():
                if "observations" in feature_id:
                    value = episode_feats["observations"][feature_id.split(".", 1)[-1]]
                else:
                    value = episode_feats.get(feature_id, None)
                if value is None:
                    break
                if value is not None:
                    if isinstance(value, np.ndarray):
                        if "images" in feature_id and image_compressed:
                            decompressed_image = cv2.imdecode(value[frame_idx], 1)
                            frame[feature_id] = torch.from_numpy(
                                decompressed_image.transpose(2, 0, 1)
                            )
                        else:
                            frame[feature_id] = torch.from_numpy(value[frame_idx])
                    elif isinstance(value, torch.Tensor):
                        frame[feature_id] = value[frame_idx]
                    else:
                        logging.warning(
                            f"[AriaVRSExtractor] Could not add dataset key at {feature_id} due to unsupported type. Skipping ..."
                        )
                        continue

            frames.append(frame)
        return frames

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


class DatasetConverter:
    """
    A class to convert datasets to Lerobot format.
    Parameters
    ----------
    raw_path : Path or str
        The path to the raw dataset.
    dataset_repo_id : str
        The repository ID where the dataset will be stored.
    fps : int
        Frames per second for the dataset.
    arm : str, optional
        The arm to process (e.g., 'left', 'right', or 'bimanual'), by default "".
    encode_as_videos : bool, optional
        Whether to encode images as videos, by default True.
    image_compressed : bool, optional
        Whether the images are compressed, by default True.
    image_writer_processes : int, optional
        Number of processes for writing images, by default 0.
    image_writer_threads : int, optional
        Number of threads for writing images, by default 0.
    prestack : bool, optional
        Whether to precompute action chunks, by default False.
    Methods
    -------
    extract_episode(episode_path, task_description='')
        Extracts frames from a single episode and saves it with a description.
    extract_episodes(episode_description='')
        Extracts frames from all episodes and saves them with a description.
    push_dataset_to_hub(dataset_tags=None, private=False, push_videos=True, license="apache-2.0")
        Pushes the dataset to the Hugging Face Hub.
    init_lerobot_dataset()
        Initializes the Lerobot dataset.
    """

    def __init__(
        self,
        raw_path: Path | str,
        dataset_repo_id: str,
        fps: int,
        arm: str = "",
        encode_as_videos: bool = True,
        image_compressed: bool = True,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        prestack: bool = False,
        debug: bool = False,
    ):
        self.raw_path = raw_path if isinstance(raw_path, Path) else Path(raw_path)
        self.dataset_repo_id = dataset_repo_id
        self.fps = fps
        self.arm = arm
        self.image_compressed = image_compressed
        self.image_writer_threads = image_writer_threads
        self.image_writer_processes = image_writer_processes
        self.encode_as_videos = encode_as_videos
        self.prestack = prestack

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - [%(name)s] - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.info(f"{'-' * 10} Aria VRS -> Lerobot Converter {'-' * 10}")
        self.logger.info(f"Processing Aria VRS dataset from {self.raw_path}")
        self.logger.info(f"Dataset will be stored in {self.dataset_repo_id}")
        self.logger.info(f"FPS: {self.fps}")
        self.logger.info(f"Arm: {self.arm}")
        self.logger.info(f"Image compressed: {self.image_compressed}")
        self.logger.info(f"Encoding images as videos: {self.encode_as_videos}")
        self.logger.info(f"Prestack: {self.prestack}")
        self.logger.info(f"#writer processes: {self.image_writer_processes}")
        self.logger.info(f"#writer threads: {self.image_writer_threads}")

        self._mp4_path = None  # set from main() if --save-mp4
        self._mp4_writer = None  # lazy-initialized in extract_episode()
        self.episode_list = list(self.raw_path.glob("*.vrs"))
        self.buffer = []

        if debug:
            self.episode_list = self.episode_list[:2]

        processed_episode = AriaVRSExtractor.process_episode(
            episode_path=self.episode_list[0],
            arm=self.arm,
            prestack=self.prestack,
        )

        if self.arm == "bimanual":
            self.robot_type = "aria_bimanual"
        elif self.arm == "right":
            self.robot_type = "aria_right_arm"
        elif self.arm == "left":
            self.robot_type = "aria_left_arm"

        self.features, metadata = AriaVRSExtractor.define_features(
            processed_episode,
            image_compressed=self.image_compressed,
            encode_as_video=self.encode_as_videos,
        )

        self.logger.info(f"Dataset Features: {self.features}")

    def save_preview_mp4(
        self, frames: list[dict], output_path: Path, fps: int, image_compressed: bool
    ):
        """
        Save a single half-resolution MP4 from a list of frame dicts.
        Each frame dict must contain 'observations.images.front_img_1' -> torch.Tensor (C,H,W) uint8.
        """
        img_key = "observations.images.front_img_1"
        imgs = [f[img_key] for f in frames if img_key in f]
        if not imgs:
            print(f"[MP4] No frames with key '{img_key}' found — skipping video save.")
            return

        C, H, W = imgs[0].shape
        size = (W // 2, H // 2)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)

        for chw in imgs:
            np_chw = chw.detach().cpu().numpy()  # (C,H,W)
            if np_chw.shape[0] == 1:
                np_chw = np.repeat(np_chw, 3, axis=0)  # grayscale → 3-ch
            np_hwc = np.transpose(np_chw, (1, 2, 0))  # CHW → HWC
            # If images were decoded with cv2 (compressed), they're already BGR;
            # otherwise convert RGB→BGR for OpenCV.
            if not image_compressed:
                np_hwc = cv2.cvtColor(np_hwc, cv2.COLOR_RGB2BGR)
            np_hwc = cv2.resize(np_hwc, size)
            writer.write(np_hwc)

        writer.release()
        print(f"[MP4] Saved half-res preview to {output_path}")

    def extract_episode(self, episode_path, task_description: str = ""):
        """
        Extracts frames from an episode and saves them to the dataset.
        Parameters
        ----------
        episode_path : str
            The path to the episode file.
        task_description : str, optional
            A description of the task associated with the episode (default is an empty string).
        Returns
        -------
        None
        """

        frames = AriaVRSExtractor.extract_episode_frames(
            episode_path,
            features=self.features,
            image_compressed=self.image_compressed,
            arm=self.arm,
            prestack=self.prestack,
        )

        if self._mp4_path is not None:
            ep_stem = Path(episode_path).stem
            mp4_path = self._mp4_path / f"{ep_stem}_video.mp4"
            self.save_preview_mp4(frames, mp4_path, self.fps, self.image_compressed)

        for i, frame in enumerate(frames):
            self.buffer.append(frame)

            if len(self.buffer) == EPISODE_LENGTH:
                for f in self.buffer:
                    self.dataset.add_frame(f)

                self.logger.info(f"Saving Episode after {i + 1} frames...")
                self.dataset.save_episode(task=task_description)
                self.buffer.clear()

    def extract_episodes(self, episode_description: str = ""):
        """
        Extracts episodes from the episode list and processes them.
        Parameters
        ----------
        episode_description : str, optional
            A description of the task to be passed to the extract_episode method (default is '').
        Raises
        ------
        Exception
            If an error occurs during the processing of an episode, it will be caught and printed.
        Notes
        -----
        After processing all episodes, the dataset is consolidated.
        """

        for episode_path in self.episode_list:
            try:
                self.extract_episode(episode_path, task_description=episode_description)
            except Exception as e:
                self.logger.error(f"Error processing episode {episode_path}: {e}")
                traceback.print_exc()
                continue

        self.buffer.clear()
        t0 = time.time()
        self.dataset.consolidate()
        elapsed_time = time.time() - t0
        self.logger.info(f"Episode consolidation time: {elapsed_time:.2f}")

    def push_dataset_to_hub(
        self,
        dataset_tags: list[str] | None = None,
        private: bool = False,
        push_videos: bool = True,
        license: str | None = "apache-2.0",
    ):
        """
        Pushes the dataset to the Hugging Face Hub.
        Parameters
        ----------
        dataset_tags : list of str, optional
            A list of tags to associate with the dataset on the Hub. Default is None.
        private : bool, optional
            If True, the dataset will be private. Default is False.
        push_videos : bool, optional
            If True, videos will be pushed along with the dataset. Default is True.
        license : str, optional
            The license under which the dataset is released. Default is "apache-2.0".
        Returns
        -------
        None
        """
        self.logger.info(
            f"Pushing dataset to Hugging Face Hub. ID: {self.dataset_repo_id} ..."
        )
        self.dataset.push_to_hub(
            tags=dataset_tags,
            license=license,
            push_videos=push_videos,
            private=private,
        )

    def init_lerobot_dataset(self, output_dir, name=Path("Test")):
        """
        Initializes the LeRobot dataset.
        This method cleans the cache if the dataset already exists and then creates a new LeRobot dataset.
        Parameters
        ----------
        output_dir : Path
            Path to root directory to store dataset
        name : Path
            Name of dataset as a Path object
        Returns
        -------
        LeRobotDataset
            The initialized LeRobot dataset.
        """
        # Clean the cache if the dataset already exists
        if os.path.exists(output_dir / name):
            shutil.rmtree(output_dir / name)

        self._out_base = Path(output_dir)

        output_dir = output_dir / name

        self.dataset = LeRobotDataset.create(
            repo_id=self.dataset_repo_id,
            fps=self.fps,
            robot_type=self.robot_type,
            features=self.features,
            image_writer_threads=self.image_writer_threads,
            image_writer_processes=self.image_writer_processes,
            root=output_dir,
        )

        return self.dataset


def argument_parse():
    parser = argparse.ArgumentParser(
        description="Convert Aria VRS dataset to LeRobot-Robomimic hybrid and push to Hugging Face hub."
    )

    # Required arguments
    parser.add_argument("--name", type=str, required=True, help="Name for dataset")
    parser.add_argument(
        "--raw-path",
        type=Path,
        required=True,
        help="Directory containing the raw HDF5 files.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="Repository ID where the dataset will be stored.",
    )
    parser.add_argument(
        "--fps", type=int, required=True, help="Frames per second for the dataset."
    )

    # Optional arguments
    parser.add_argument(
        "--description",
        type=str,
        default="Aria recorded dataset.",
        help="Description of the dataset.",
    )
    parser.add_argument(
        "--arm",
        type=str,
        choices=["left", "right", "bimanual"],
        default="bimanual",
        help="Specify the arm for processing.",
    )
    parser.add_argument(
        "--private",
        type=str2bool,
        default=False,
        help="Set to True to make the dataset private.",
    )
    parser.add_argument(
        "--push",
        type=str2bool,
        default=True,
        help="Set to True to push videos to the hub.",
    )
    parser.add_argument(
        "--license", type=str, default="apache-2.0", help="License for the dataset."
    )
    parser.add_argument(
        "--image-compressed",
        type=str2bool,
        default=True,
        help="Set to True if the images are compressed.",
    )
    parser.add_argument(
        "--video-encoding",
        type=str2bool,
        default=True,
        help="Set to True to encode images as videos.",
    )
    parser.add_argument(
        "--prestack",
        type=str2bool,
        default=True,
        help="Set to True to precompute action chunks.",
    )

    # Performance tuning arguments
    parser.add_argument(
        "--nproc", type=int, default=8, help="Number of image writer processes."
    )
    parser.add_argument(
        "--nthreads", type=int, default=2, help="Number of image writer threads."
    )

    # Debugging and output configuration
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(LEROBOT_HOME),
        help="Directory where the processed dataset will be stored. Defaults to LEROBOT_HOME.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Store only 2 episodes for debug purposes."
    )

    parser.add_argument(
        "--save-mp4",
        type=str2bool,
        default=True,
        help="If True, save a single half-resolution MP4 with all frames across episodes.",
    )

    args = parser.parse_args()

    return args


def main(args):
    """
    Convert ARIA VRS files and push to Hugging Face hub.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    print(
        args.video_encoding,
        "-------------------------------------------------------------------------------------------------------",
    )

    # Initialize the dataset converter
    converter = DatasetConverter(
        raw_path=args.raw_path,
        dataset_repo_id=args.dataset_repo_id,
        fps=args.fps,
        arm=args.arm,
        image_compressed=args.image_compressed,
        encode_as_videos=args.video_encoding,
        image_writer_processes=args.nproc,
        image_writer_threads=args.nthreads,
        prestack=args.prestack,
        debug=args.debug,
    )

    # Initialize the dataset
    converter.init_lerobot_dataset(output_dir=args.output_dir, name=Path(args.name))
    if args.save_mp4:
        converter._mp4_path = converter._out_base
    # Extract episodes
    converter.extract_episodes(episode_description=args.description)

    # Push the dataset to the Hugging Face Hub, if specified
    if args.push:
        converter.push_dataset_to_hub(
            dataset_tags=AriaVRSExtractor.TAGS,
            private=args.private,
            push_videos=args.video_encoding,
            license=args.license,
        )


if __name__ == "__main__":
    args = argument_parse()
    main(args)
