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
from egomimic.utils.egomimicUtils import (
    nds,
    ee_pose_to_cam_frame,
    EXTRINSICS,
    str2bool,
    ee_orientation_to_cam_frame,
)

from egomimic.robot.eva.eva_kinematics import (
    EvaMinkKinematicsSolver as EvaKinematicsSolver,
)
from egomimic.rldb.utils import EMBODIMENT

from typing import Union
import egomimic

import time

import numpy as np

import torch
import subprocess
import shutil
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R

from enum import Enum

## CHANGE THIS TO YOUR DESIRED CACHE FOR HF
os.environ["HF_HOME"] = "~/.cache/huggingface"

DATASET_KEY_MAPPINGS = {
    "joint_positions": "joint_positions",
    "front_img_1": "front_img_1",
    "right_wrist_img": "right_wrist_img",
    "left_wrist_img": "left_wrist_img",
}

EVA_XML_PATH = os.path.join(
    os.path.dirname(egomimic.__file__), "resources/model_x5.xml"
)

POINT_GAP_ACT = 2
CHUNK_LENGTH_ACT = 100

Array2D = Union[np.ndarray, torch.Tensor]


def _as_rotation(x):
    """Return a scipy Rotation from either Rotation or 3x3 ndarray."""
    return x if isinstance(x, R) else R.from_matrix(np.asarray(x, dtype=float))


def _as_matrix(x):
    """Return a 3x3 ndarray from either Rotation or 3x3 ndarray."""
    return x.as_matrix() if isinstance(x, R) else np.asarray(x, dtype=float)


def _row_to_numpy(x: Array2D, i: int) -> np.ndarray:
    """Get i-th row as numpy array (handles torch or numpy input)."""
    if isinstance(x, np.ndarray):
        return x[i]
    return x[i].detach().cpu().numpy()


def fk_xyz(
    joints_2d: Array2D, eva_fk: EvaKinematicsSolver, *, dtype=torch.float32, device=None
) -> torch.Tensor:
    """
    Eva FK positions for a sequence of joint vectors.
    Returns torch.Tensor of shape (T, 3).
    """
    T = joints_2d.shape[0]
    out = torch.empty((T, 3), dtype=dtype, device=device)
    for i in range(T):
        q = _row_to_numpy(joints_2d, i)  # (DoF,)
        pos, _R = eva_fk.fk(q)  # pos: (3,), _R: (3,3) numpy
        out[i] = torch.as_tensor(pos, dtype=dtype, device=device)
    return out


def fk_SE3(
    joints_2d: Array2D, eva_fk: EvaKinematicsSolver, *, dtype=torch.float32, device=None
) -> torch.Tensor:
    """
    Eva FK full SE(3) for a sequence of joint vectors.
    Returns torch.Tensor of shape (T, 4, 4) with bottom-right set to 1.
    Downstream can keep using fk[:, :3, 3] and fk[:, :3, :3].
    """
    T = joints_2d.shape[0]
    out = torch.zeros((T, 4, 4), dtype=dtype, device=device)
    out[:, 3, 3] = 1.0
    for i in range(T):
        q = _row_to_numpy(joints_2d, i)
        pos, R_obj = eva_fk.fk(q)  # numpy: (3,), (3,3)
        Rm = _as_matrix(R_obj)
        out[i, :3, :3] = torch.as_tensor(Rm, dtype=dtype, device=device)
        out[i, :3, 3] = torch.as_tensor(pos, dtype=dtype, device=device)
    return out


def get_future_points(arr, POINT_GAP=POINT_GAP_ACT, CHUNK_LENGTH=CHUNK_LENGTH_ACT):
    """
    arr: (T, ACTION_DIM)
    POINT_GAP: how many timesteps to skip
    CHUNK_LENGTH: how many future points to collect
    given an array arr, prepack the future points into each timestep.  return an array of size (T, CHUNK_LENGTH, ACTION_DIM).  If there are not enough future points, pad with the last point.
    do it purely vectorized
    """
    T, ACTION_DIM = arr.shape
    result = np.zeros((T, CHUNK_LENGTH, ACTION_DIM))

    for t in range(T):
        future_indices = np.arange(t, t + POINT_GAP * (CHUNK_LENGTH), POINT_GAP)
        future_indices = np.clip(future_indices, 0, T - 1)
        result[t] = arr[future_indices]
    return result


def sample_interval_points(arr, POINT_GAP=POINT_GAP_ACT, CHUNK_LENGTH=CHUNK_LENGTH_ACT):
    """
    arr: (T, ACTION_DIM)
    Returns an array of points sampled at intervals of POINT_GAP * CHUNK_LENGTH.
    """
    num_samples, T, ACTION_DIM = arr.shape
    interval = T / 10
    indices = np.arange(0, T, interval).astype(int)
    sampled_points = arr[:, indices, :]
    return sampled_points


def joint_to_pose(pose, arm, left_extrinsics=None, right_extrinsics=None, no_rot=False):
    """
    pose: (T, ACTION_DIM)
    arm: left, right, both arms of the robot
    left_extrinsics: the left camera extrinsics
    right_extrinsics: the right camera extrinsics
    """
    eva_fk = EvaKinematicsSolver(model_path=str(EVA_XML_PATH))

    if arm == "both":
        joint_start = 0
        joint_end = 14
        joint_left_start = 0
        joint_left_end = 7
        joint_right_start = 7
        joint_right_end = 14
    else:
        if arm == "left":
            joint_start = 0
            joint_end = 7
        elif arm == "right":
            joint_start = 7
            joint_end = 14

    if no_rot:
        if arm == "both":
            fk_left_positions = fk_xyz(
                pose[:, joint_left_start : joint_left_end - 1], eva_fk
            )
            fk_right_positions = fk_xyz(
                pose[:, joint_right_start : joint_right_end - 1], eva_fk
            )
            fk_left_positions = ee_pose_to_cam_frame(fk_left_positions, left_extrinsics)
            fk_right_positions = ee_pose_to_cam_frame(
                fk_right_positions, right_extrinsics
            )
            fk_positions = np.concatenate(
                [fk_left_positions, fk_right_positions], axis=1
            )
        else:
            fk_positions = fk_xyz(pose[:, joint_start : joint_end - 1], eva_fk)
            extrinsics = left_extrinsics if arm == "left" else right_extrinsics
            fk_positions = ee_pose_to_cam_frame(fk_positions, extrinsics)

    else:
        if arm == "both":
            fk_left = fk_SE3(pose[:, joint_left_start : joint_left_end - 1], eva_fk)
            fk_right = fk_SE3(pose[:, joint_right_start : joint_right_end - 1], eva_fk)

            fk_left_positions = fk_left[:, :3, 3]
            fk_left_orientations = fk_left[:, :3, :3]
            fk_right_positions = fk_right[:, :3, 3]
            fk_right_orientations = fk_right[:, :3, :3]

            left_gripper = pose[:, joint_left_end - 1].reshape(-1, 1)
            right_gripper = pose[:, joint_right_end - 1].reshape(-1, 1)

            fk_left_positions = ee_pose_to_cam_frame(fk_left_positions, left_extrinsics)
            fk_right_positions = ee_pose_to_cam_frame(
                fk_right_positions, right_extrinsics
            )

            fk_left_orientations, fk_left_ypr = ee_orientation_to_cam_frame(
                fk_left_orientations, left_extrinsics
            )
            fk_right_orientations, fk_right_ypr = ee_orientation_to_cam_frame(
                fk_right_orientations, right_extrinsics
            )

            fk_positions = np.concatenate(
                [
                    fk_left_positions,
                    fk_left_ypr,
                    left_gripper,
                    fk_right_positions,
                    fk_right_ypr,
                    right_gripper,
                ],
                axis=1,
            )

        else:
            fk = fk_SE3(pose[:, joint_start : joint_end - 1], eva_fk)

            fk_positions = fk[:, :3, 3]
            fk_orientations = fk[:, :3, :3]

            gripper = pose[:, joint_end - 1].reshape(-1, 1)

            extrinsics = left_extrinsics if arm == "left" else right_extrinsics
            fk_positions = ee_pose_to_cam_frame(fk_positions, extrinsics)
            fk_orientations, fk_ypr = ee_orientation_to_cam_frame(
                fk_orientations, extrinsics
            )

            fk_positions = np.concatenate([fk_positions, fk_ypr, gripper], axis=1)

    return fk_positions


class EvaHD5Extractor:
    TAGS = ["eva", "robotics", "hdf5"]

    @staticmethod
    def process_episode(
        episode_path, arm, extrinsics, prestack=False, low_res=False, no_rot=False
    ):
        """
        Extracts all feature keys from a given episode and returns as a dictionary
        Parameters
        ----------
        episode_path : str or Path
            Path to the HDF5 file containing the episode data.
        arm : str
            String for which arm to add data for
        extrinsics : np.array
            camera extrinsic, It is a tuple of (left_extrinsics, right_extrinsics) if arm is both
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
        left_extrinsics = None
        right_extrinsics = None

        if arm == "both":
            if not isinstance(extrinsics, dict):
                logging.info(
                    "Error: Both arms selected. Expected extrinsics for both arms."
                )
            left_extrinsics = extrinsics["left"]
            right_extrinsics = extrinsics["right"]
        elif arm == "left":
            extrinsics = extrinsics["left"]
            left_extrinsics = extrinsics
        elif arm == "right":
            extrinsics = extrinsics["right"]
            right_extrinsics = extrinsics

        if arm == "left":
            joint_start = 0
            joint_end = 7
        elif arm == "right":
            joint_start = 7
            joint_end = 14
        elif arm == "both":
            joint_start = 0
            joint_end = 14

        episode_feats = dict()

        # TODO: benchmarking only, remove for release
        t0 = time.time()

        with h5py.File(episode_path, "r") as episode:
            # rgb camera
            logging.info(f"[EvaHD5Extractor] Reading HDF5 file: {episode_path}")

            episode_feats["observations"] = dict()

            for camera in EvaHD5Extractor.get_cameras(episode):
                images = (
                    torch.from_numpy(episode["observations"]["images"][camera][:])
                    .permute(0, 3, 1, 2)
                    .float()
                )

                if low_res:
                    images = F.interpolate(
                        images, size=(240, 320), mode="bilinear", align_corners=False
                    )

                images = images.byte().numpy()

                mapped_key = DATASET_KEY_MAPPINGS.get(camera, camera)
                episode_feats["observations"][f"images.{mapped_key}"] = images

            # state
            for state in EvaHD5Extractor.get_state(episode):
                mapped_key = DATASET_KEY_MAPPINGS.get(state, state)
                episode_feats["observations"][f"state.{mapped_key}"] = episode[
                    "observations"
                ][state][:]

            # ee_pose
            episode_feats["observations"][f"state.ee_pose"] = (
                EvaHD5Extractor.get_ee_pose(
                    episode_feats["observations"][f"state.joint_positions"],
                    arm,
                    left_extrinsics=left_extrinsics,
                    right_extrinsics=right_extrinsics,
                    no_rot=no_rot,
                )
            )

            # actions
            joint_actions, cartesian_actions = EvaHD5Extractor.get_action(
                episode["action"][:],
                arm=arm,
                prestack=prestack,
                POINT_GAP=POINT_GAP_ACT,
                CHUNK_LENGTH=CHUNK_LENGTH_ACT,
                left_extrinsics=left_extrinsics,
                right_extrinsics=right_extrinsics,
                no_rot=no_rot,
            )

            episode_feats["actions_joints"] = joint_actions
            episode_feats["actions_cartesian"] = cartesian_actions

            episode_feats["observations"][f"state.joint_positions"] = episode_feats[
                "observations"
            ][f"state.joint_positions"][:, joint_start:joint_end]

            num_timesteps = episode_feats["observations"][f"state.ee_pose"].shape[0]
            if arm == "right":
                value = EMBODIMENT.EVA_RIGHT_ARM.value
            elif arm == "left":
                value = EMBODIMENT.EVA_LEFT_ARM.value
            else:
                value = EMBODIMENT.EVA_BIMANUAL.value

            episode_feats["metadata.embodiment"] = np.full(
                (num_timesteps, 1), value, dtype=np.int32
            )

        # TODO: benchmarking only, remove for release
        elapsed_time = time.time() - t0
        logging.info(
            f"[EvaHD5Extractor] Finished processing episode at {episode_path} in {elapsed_time:.2f} sec"
        )

        return episode_feats

    @staticmethod
    def get_action(
        actions: np.array,
        arm: str,
        prestack=False,
        POINT_GAP=2,
        CHUNK_LENGTH=100,
        left_extrinsics=None,
        right_extrinsics=None,
        no_rot=False,
    ):
        """
        Uses FK to calculate ee pose from joints
        Parameters
        ----------
        pose : np.array
            array containing joint actions
        arm : str
            arm to convert data for
        prestack : bool
            whether or not to precompute action chunks
        POINT_GAP : int
            interpolation for timesteps
        CHUNK_LENGTH : int
            action chunk length
        left_extrinsics :
            camera extrinsics
        right_extrinsics :
            camera_extrinsics
        no_rot: bool
            calculate full 6dof trajectory or not
        Returns
        -------
        actions : tuple of np.array
            (joint actions, cartesian actions)
        """

        joint_actions = actions

        if arm == "left":
            joint_start = 0
            joint_end = 7
        elif arm == "right":
            joint_start = 7
            joint_end = 14
        elif arm == "both":
            joint_start = 0
            joint_end = 14

        cartesian_actions = joint_to_pose(
            pose=joint_actions,
            arm=arm,
            left_extrinsics=left_extrinsics,
            right_extrinsics=right_extrinsics,
            no_rot=no_rot,
        )

        joint_actions = joint_actions[:, joint_start:joint_end]

        if prestack:
            joint_actions = get_future_points(
                joint_actions, POINT_GAP=POINT_GAP, CHUNK_LENGTH=CHUNK_LENGTH
            )
            joint_actions_sampled = sample_interval_points(
                joint_actions, POINT_GAP=POINT_GAP, CHUNK_LENGTH=CHUNK_LENGTH
            )
            cartesian_actions = get_future_points(
                cartesian_actions, POINT_GAP=POINT_GAP, CHUNK_LENGTH=CHUNK_LENGTH
            )
            cartesian_actions_sampled = sample_interval_points(
                cartesian_actions, POINT_GAP=POINT_GAP, CHUNK_LENGTH=CHUNK_LENGTH
            )

        # TODO: fix saving the sampled
        return (joint_actions, cartesian_actions)

    @staticmethod
    def get_ee_pose(
        qpos: np.array,
        arm: str,
        left_extrinsics=None,
        right_extrinsics=None,
        no_rot=False,
    ):
        """
        Uses FK to calculate ee pose from joints
        Parameters
        ----------
        qpos : np.array
            array containing joint positions
        arm : str
            arm to convert data for
        left_extrinsics :
            camera extrinsics
        right_extrinsics :
            camera_extrinsics
        no_rot : bool
            calculate full 6dof pose or not
        Returns
        -------
        ee_pose : np.array
            ee_pose SE{3}
        """

        ee_pose = joint_to_pose(
            qpos, arm, left_extrinsics, right_extrinsics, no_rot=no_rot
        )

        return ee_pose

    @staticmethod
    def get_cameras(hdf5_data: h5py.File):
        """
        Extracts the list of RGB camera keys from the given HDF5 data.
        Parameters
        ----------
        hdf5_data : h5py.File
            The HDF5 file object containing the dataset.
        Returns
        -------
        list of str
            A list of keys corresponding to RGB cameras in the dataset.
        """

        rgb_cameras = [
            key for key in hdf5_data["/observations/images"] if "depth" not in key
        ]
        return rgb_cameras

    @staticmethod
    def get_state(hdf5_data: h5py.File):
        """
        Extracts the list of RGB camera keys from the given HDF5 data.
        Parameters
        ----------
        hdf5_data : h5py.File
            The HDF5 file object containing the dataset.
        Returns
        -------
        states : list of str
            A list of keys corresponding to states in the dataset.
        """

        states = [key for key in hdf5_data["/observations"] if "images" not in key]
        return states

    @staticmethod
    def check_format(
        episode_list: list[str] | list[Path], image_compressed: bool = True
    ):
        """
        Check the format of the given list of HDF5 files.
        Parameters
        ----------
        episode_list : list of str or list of Path
            List of paths to the HDF5 files to be checked.
        image_compressed : bool, optional
            Flag indicating whether the images are compressed (default is True).
        Raises
        ------
        ValueError
            If the episode_list is empty.
            If any HDF5 file is missing required keys '/action' or '/observations/joint_positions'.
            If the '/action' or '/observations/joint_positions' keys do not have 2 dimensions.
            If the number of frames in '/action' and '/observations/joint_positions' keys do not match.
            If the number of frames in '/observations/images/{camera}' does not match the number of frames in '/action' and '/observations/joint_positions'.
            If the dimensions of images do not match the expected dimensions based on the image_compressed flag.
            If uncompressed images do not have the expected (h, w, c) format.
        """

        if not episode_list:
            raise ValueError(
                "No hdf5 files found in the raw directory. Make sure they are named '*.hdf5'"
            )
        for episode_path in episode_list:
            with h5py.File(episode_path, "r") as data:
                if not all(
                    key in data for key in ["/action", "/observations/joint_positions"]
                ):
                    raise ValueError(
                        "Missing required keys in the hdf5 file. Make sure the keys '/action' and '/observations/joint_positions' are present."
                    )

                if (
                    not data["/action"].ndim
                    == data["/observations/joint_positions"].ndim
                    == 2
                ):
                    raise ValueError(
                        "The '/action' and '/observations/joint_positions' keys should have both 2 dimensions."
                    )

                if (num_frames := data["/action"].shape[0]) != data[
                    "/observations/joint_positions"
                ].shape[0]:
                    raise ValueError(
                        "The '/action' and '/observations/joint_positions' keys should have the same number of frames."
                    )

                for camera in EvaHD5Extractor.get_cameras(data):
                    if num_frames != data[f"/observations/images/{camera}"].shape[0]:
                        raise ValueError(
                            f"The number of frames in '/observations/images/{camera}' should be the same as in '/action' and '/observations/joint_positions' keys."
                        )

                    expected_dims = 2 if image_compressed else 4
                    if data[f"/observations/images/{camera}"].ndim != expected_dims:
                        raise ValueError(
                            f"Expect {expected_dims} dimensions for {'compressed' if image_compressed else 'uncompressed'} images but {data[f'/observations/images/{camera}'].ndim} provided."
                        )
                    if not image_compressed:
                        b, h, w, c = data[f"/observations/images/{camera}"].shape
                        if not c < h and c < w:
                            raise ValueError(
                                f"Expect (h,w,c) image format but ({h=},{w=},{c=}) provided."
                            )

    @staticmethod
    def extract_episode_frames(
        episode_path: str | Path,
        features: dict[str, dict],
        image_compressed: bool,
        arm: str,
        extrinsics: dict,
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
            The arm to process (e.g., 'left', 'right', or 'both').
        extrinsics : dict
            Camera extrinsics for the episode.
        prestack : bool, optional
            Whether to precompute action chunks, by default False.

        Returns
        -------
        list[dict[str, torch.Tensor]]
            List of frames, where each frame is a dictionary mapping feature identifiers to tensors.
        """
        frames = []
        episode_feats = EvaHD5Extractor.process_episode(
            episode_path, arm=arm, extrinsics=extrinsics, prestack=prestack
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
                            f"[EvaHD5Extractor] Could not add dataset key at {feature_id} due to unsupported type. Skipping ..."
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
                nested_features, nested_metadata = EvaHD5Extractor.define_features(
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
                shape = tuple(value[0].size())
                if "actions" in key and len(tuple(value[0].size())) > 1:
                    dim_names = ["chunk_length", "action_dim"]
                    dtype = f"prestacked_{str(value.dtype)}"
                else:
                    dim_names = [f"dim_{i}" for i in range(len(shape))]
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
        The arm to process (e.g., 'left', 'right', or 'both'), by default "".
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
        extrinsics_key: str = "",
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
        self.extrinsics_key = extrinsics_key
        self.image_compressed = image_compressed
        self.image_writer_threads = image_writer_threads
        self.image_writer_processes = image_writer_processes
        self.encode_as_videos = encode_as_videos
        self.prestack = prestack

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        self._mp4_path = None

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - [%(name)s] - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.info(f"{'-' * 10} Eva HD5 -> Lerobot Converter {'-' * 10}")
        self.logger.info(f"Processing Eva HD5 dataset from {self.raw_path}")
        self.logger.info(f"Dataset will be stored in {self.dataset_repo_id}")
        self.logger.info(f"FPS: {self.fps}")
        self.logger.info(f"Arm: {self.arm}")
        self.logger.info(f"Extrinsics key: {self.extrinsics_key}")
        self.logger.info(f"Image compressed: {self.image_compressed}")
        self.logger.info(f"Encoding images as videos: {self.encode_as_videos}")
        self.logger.info(f"Prestack: {self.prestack}")
        self.logger.info(f"#writer processes: {self.image_writer_processes}")
        self.logger.info(f"#writer threads: {self.image_writer_threads}")

        self.episode_list = list(self.raw_path.glob("*.hdf5"))

        if debug:
            self.episode_list = self.episode_list[:2]

        EvaHD5Extractor.check_format(
            self.episode_list, image_compressed=self.image_compressed
        )

        extrinsics = EXTRINSICS[self.extrinsics_key]
        processed_episode = EvaHD5Extractor.process_episode(
            episode_path=self.episode_list[0],
            arm=self.arm,
            extrinsics=extrinsics,
            prestack=self.prestack,
        )

        if self.arm == "both":
            self.robot_type = "eva_bimanual"
        elif self.arm == "right":
            self.robot_type = "eva_right_arm"
        elif self.arm == "left":
            self.robot_type = "eva_left_arm"

        self.features, metadata = EvaHD5Extractor.define_features(
            processed_episode,
            image_compressed=self.image_compressed,
            encode_as_video=self.encode_as_videos,
        )

        self.logger.info(f"Dataset Features: {self.features}")

    def save_preview_mp4(
        self, frames: list[dict], output_path: Path, fps: int, image_compressed: bool
    ):
        """
        Save a half-resolution, web-compatible MP4 (H.264, yuv420p).

        Strategy:
        1. Try torchvision.io.write_video (H.264 via FFmpeg libs, no CLI).
        2. If that fails, fall back to ffmpeg CLI via subprocess.
        3. If both fail, raise a RuntimeError.

        Expects each frame dict to contain:
            'observations.images.front_img_1' -> torch.Tensor (C,H,W), uint8, BGR.
        """
        img_key = "observations.images.front_img_1"
        imgs = [f[img_key] for f in frames if img_key in f]
        if not imgs:
            print(f"[MP4] No frames with key '{img_key}' found — skipping video save.")
            return

        # Assume imgs[0] is (C,H,W)
        C, H, W = imgs[0].shape

        # Compute half-res (force even dims for yuv420p)
        outW, outH = W // 2, H // 2
        if outW % 2:
            outW -= 1
        if outH % 2:
            outH -= 1
        if outW <= 0 or outH <= 0:
            raise ValueError(f"[MP4] Invalid output size: {outW}x{outH}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Build resized RGB frames once
        # -----------------------------
        rgb_frames = []
        for chw in imgs:
            # chw: (C,H,W) uint8, BGR from cv2.imdecode earlier
            t = chw.detach().cpu()
            if t.dtype != torch.uint8:
                t = t.to(torch.uint8)

            # If grayscale, repeat to 3 channels
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)

            # Resize to (outH, outW)
            t_resized = F.interpolate(
                t.unsqueeze(0),  # (1,C,H,W)
                size=(outH, outW),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # (C,outH,outW)

            # BGR -> RGB, then (H,W,C)
            t_resized = t_resized[[2, 1, 0], ...]  # (3,H,W) RGB
            hwc = t_resized.permute(1, 2, 0).contiguous()  # (H,W,3), uint8
            rgb_frames.append(hwc)

        video_tensor = torch.stack(rgb_frames, dim=0)  # (T, H, W, 3) uint8

        # -----------------------------
        # 1) Try torchvision.write_video
        # -----------------------------
        try:
            from torchvision.io import write_video

            write_video(
                filename=str(output_path),
                video_array=video_tensor,
                fps=float(fps),
                video_codec="libx264",  # H.264, web-compatible
                options={"crf": "23", "preset": "veryfast"},
            )
            print(
                f"[MP4] Saved web-compatible H.264 preview via torchvision to {output_path}"
            )
            return
        except Exception as e:
            print(
                f"[MP4] torchvision.io.write_video failed ({e}); trying ffmpeg CLI fallback..."
            )

        # -----------------------------
        # 2) Fallback: ffmpeg CLI (libx264)
        # -----------------------------
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "[MP4] Could not write web-compatible MP4:\n"
                "  - torchvision.io.write_video is unavailable or failed\n"
                "  - `ffmpeg` CLI not found on PATH\n"
                "Install either torchvision with video support or ffmpeg+libx264."
            )

        # For ffmpeg rawvideo, we need BGR24 frames of shape (outH, outW, 3)
        # We can convert our RGB hwc tensors back to BGR numpy.
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{outW}x{outH}",
            "-r",
            str(fps),
            "-i",
            "-",  # stdin
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "baseline",
            "-level",
            "3.0",
            "-movflags",
            "+faststart",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            str(output_path),
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            for hwc_rgb in rgb_frames:
                # hwc_rgb: (H,W,3), RGB uint8
                np_rgb = hwc_rgb.numpy()
                # RGB -> BGR
                np_bgr = np_rgb[..., ::-1]
                proc.stdin.write(np_bgr.tobytes())
        finally:
            if proc.stdin:
                proc.stdin.flush()
                proc.stdin.close()

        ret = proc.wait()
        if ret != 0:
            stderr = proc.stderr.read().decode(errors="ignore") if proc.stderr else ""
            raise RuntimeError(
                f"[MP4] ffmpeg/libx264 encoding failed (exit {ret}).\n{stderr}"
            )

        print(
            f"[MP4] Saved web-compatible H.264 preview via ffmpeg CLI to {output_path}"
        )

    def extract_episode(self, episode_path, task_description: str = ""):
        extrinsics = EXTRINSICS[self.extrinsics_key]

        frames = EvaHD5Extractor.extract_episode_frames(
            episode_path,
            features=self.features,
            image_compressed=self.image_compressed,
            arm=self.arm,
            extrinsics=extrinsics,
            prestack=self.prestack,
        )

        if self._mp4_path is not None:
            ep_stem = Path(episode_path).stem
            mp4_path = self._mp4_path / f"{ep_stem}_video.mp4"
            self.save_preview_mp4(frames, mp4_path, self.fps, self.image_compressed)

        for frame in frames:
            self.dataset.add_frame(frame)

        self.logger.info(f"Saving Episode with Description: {task_description} ...")
        self.dataset.save_episode(task=task_description)

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
        description="Convert Eva HD5 dataset to LeRobot-Robomimic hybrid and push to Hugging Face hub."
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
        default="Eva recorded dataset.",
        help="Description of the dataset.",
    )
    parser.add_argument(
        "--arm",
        type=str,
        choices=["left", "right", "both"],
        default="both",
        help="Specify the arm for processing.",
    )
    parser.add_argument(
        "--extrinsics-key",
        type=str,
        default="ariaOct18_arx",
        help="Key to look up camera extrinsics.",
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
        "--nproc", type=int, default=12, help="Number of image writer processes."
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
        help="If True, save one web-compatible MP4 per episode using front_img_1.",
    )

    args = parser.parse_args()

    return args


def main(args):
    """
    Convert Eva HD5 dataset and push to Hugging Face hub.

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
        extrinsics_key=args.extrinsics_key,
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
            dataset_tags=EvaHD5Extractor.TAGS,
            private=args.private,
            push_videos=args.video_encoding,
            license=args.license,
        )


if __name__ == "__main__":
    args = argument_parse()
    main(args)
