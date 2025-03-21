import argparse
import os
from pathlib import Path
import shutil
import traceback
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torch
import logging
from enum import Enum

from egomimic.utils.egomimicUtils import str2bool

from rldb.utils import EMBODIMENT

import time

import numpy as np

from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

# TODO(roger): these constants should be better managed
H1_HUMAN_SLOW_DOWN_FACTOR = 4
EPISODE_LENGTH = 1000
HORIZON_DEFAULT = 20
# UCSD H1 runs at 30 Hz
FPS = 30
H1_ARM_TYPE = "bimanual"
CHUNK_LENGTH_ACT = 100

RETARGETTING_INDICES = [0, 4, 9, 14, 19, 24]

OUTPUT_LEFT_EEF = np.arange(80, 89)
OUTPUT_RIGHT_EEF = np.arange(30, 39)
OUTPUT_HEAD_EEF = np.arange(0, 9)
OUTPUT_LEFT_KEYPOINTS = np.arange(10, 10 + 3 * len(RETARGETTING_INDICES))
assert OUTPUT_LEFT_KEYPOINTS[-1] < OUTPUT_RIGHT_EEF[0]
OUTPUT_RIGHT_KEYPOINTS = np.arange(40, 40 + 3 * len(RETARGETTING_INDICES))
assert OUTPUT_RIGHT_KEYPOINTS[-1] < OUTPUT_LEFT_EEF[0]

def cmd_dict_from_128dim(action):
    left_wrist_mat = np.eye(4)
    left_wrist_mat[0:3, 3] = action[OUTPUT_LEFT_EEF[0:3]]
    left_wrist_mat[0:3, 0:3] = rotation_6d_to_matrix(torch.tensor(action[OUTPUT_LEFT_EEF[3:]]).unsqueeze(0)).numpy()

    left_hand_keypoints = np.zeros((25,3))
    left_hand_keypoints[RETARGETTING_INDICES] = action[OUTPUT_LEFT_KEYPOINTS].reshape((6,3))

    right_wrist_mat = np.eye(4)
    right_wrist_mat[0:3, 3] = action[OUTPUT_RIGHT_EEF[0:3]]
    right_wrist_mat[0:3, 0:3] = rotation_6d_to_matrix(torch.tensor(action[OUTPUT_RIGHT_EEF[3:]]).unsqueeze(0)).numpy()

    right_hand_keypoints = np.zeros((25,3))
    right_hand_keypoints[RETARGETTING_INDICES] = action[OUTPUT_RIGHT_KEYPOINTS].reshape((6,3))

    head_mat = np.eye(4)
    head_mat[0:3, 3] = action[OUTPUT_HEAD_EEF[0:3]]
    head_rmat = rotation_6d_to_matrix(torch.tensor(action[OUTPUT_HEAD_EEF[3:]]).unsqueeze(0)).squeeze().numpy()
    head_mat[0:3, 0:3] = head_rmat

    return {
        'left_wrist_mat': left_wrist_mat,
        'left_hand_kpts': left_hand_keypoints,
        'right_wrist_mat': right_wrist_mat,
        'right_hand_kpts': right_hand_keypoints,
        'head_mat': head_mat
    }

class UCSDHDFExtractor:
    TAGS = ["ucsd", "robotics", "hdf5"]

    @staticmethod
    def process_episode(episode_path, arm: str, prestack=False, low_res=False):
        """
        Extracts all feature keys from a given episode and returns as a dictionary
        Parameters
        ----------
        episode_path : str or Path
            Path to the VRS file containing the episode data.
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
        episode_feats["observations"] = dict()

        assert arm in ["bimanual", "both"]

        # numpy images (B, H, W, C)
        with h5py.File(episode_path, "r") as f:
            left_images = []
            for img in f["observation.image.left"]:
                decompressed_img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                if low_res:
                    decompressed_img = cv2.resize(decompressed_img, (320, 240), interpolation=cv2.INTER_LINEAR)
                left_images.append(decompressed_img)
            left_images = np.array(left_images)

            right_images = []
            for img in f["observation.image.right"]:
                decompressed_img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                if low_res:
                    decompressed_img = cv2.resize(decompressed_img, (320, 240), interpolation=cv2.INTER_LINEAR)
                right_images.append(decompressed_img)
            right_images = np.array(right_images)

            # TODO(roger): there are constant H1 state and action offsets. Do we need a way to compensate?
            # TODO(roger): currently only save wrist eef poses
            state_list = []
            for state in f["observation.state"]:
                cur_cmd_dict = cmd_dict_from_128dim(state)
                cur_state = np.array([
                    cur_cmd_dict["left_wrist_mat"][0:3, 3],
                    cur_cmd_dict["right_wrist_mat"][0:3, 3]
                ]).flatten()
                state_list.append(cur_state)
            state_arr = np.array(state_list)

            action_list = []
            for action in f["action"]:
                cur_cmd_dict = cmd_dict_from_128dim(action)
                cur_action = np.array([
                    cur_cmd_dict["left_wrist_mat"][0:3, 3],
                    cur_cmd_dict["right_wrist_mat"][0:3, 3]
                ]).flatten()
                action_list.append(cur_action)
            action_arr = np.array(action_list)
        
        if True:
            # TODO: temporary solution to match axes direction.
            # Better way is to fix rotations + reference frames to be consistent with Aria.
            state_arr = -1 * state_arr
            action_arr = -1 * action_arr
        
        if prestack:
            actions = []
            sample_chunk_size = int(CHUNK_LENGTH_ACT / H1_HUMAN_SLOW_DOWN_FACTOR)
            for i in range(0, action_arr.shape[0]):
                if i + sample_chunk_size > action_arr.shape[0]:
                    # Copy the last action to fill the chunk
                    copy_size = i + sample_chunk_size - action_arr.shape[0]
                    first_action_chunk = action_arr[i:i+sample_chunk_size]
                    second_action_chunk = action_arr[-1] * np.ones((copy_size, action_arr.shape[1]))
                    action_chunk = np.concatenate((first_action_chunk, second_action_chunk), axis=0)
                    actions.append(action_chunk)
                else:
                    actions.append(action_arr[i:i+sample_chunk_size])
            actions = np.array(actions)
        else:
            actions = action_arr
        
        episode_feats["observations"][f"state.ee_pose"] = state_arr
        episode_feats["observations"][f"images.front_img_1"] = left_images.transpose((0, 3, 1, 2))
        episode_feats["observations"][f"images.front_img_2"] = right_images.transpose((0, 3, 1, 2))
        episode_feats["actions_cartesian"] = actions

        num_timesteps = episode_feats["observations"][f"state.ee_pose"].shape[0]

        import pdb; pdb.set_trace()

        value = EMBODIMENT.UCSD_H1_BIMANUAL.value

        episode_feats["metadata.embodiment"] = np.full((num_timesteps, 1), value, dtype=np.int32)

        return episode_feats

    # TODO: this is largely similar to AriaVRSExtractor.extract_episode_frames. Should use classmethod everywhere
    @classmethod
    def extract_episode_frames(
        cls, episode_path: str | Path, features: dict[str, dict], image_compressed: bool, arm: str, prestack: bool = False
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
        prestack : bool, optional
            Whether to precompute action chunks, by default False.

        Returns
        -------
        list[dict[str, torch.Tensor]]
            List of frames, where each frame is a dictionary mapping feature identifiers to tensors.
        """
        frames = []
        episode_feats = cls.process_episode(
            episode_path, arm=arm, prestack=prestack
        )
        num_frames = next(iter(episode_feats["observations"].values())).shape[0]
        for frame_idx in range(num_frames):
            frame = {}
            for feature_id, feature_info in features.items():
                if "observations" in feature_id:
                    value = episode_feats["observations"][feature_id.split('.', 1)[-1]]
                else:
                    value = episode_feats.get(feature_id, None)
                if value is None:
                    break
                if value is not None:
                    if isinstance(value, np.ndarray):
                        if "images" in feature_id and image_compressed:
                            decompressed_image = cv2.imdecode(value[frame_idx], 1)
                            frame[feature_id] = torch.from_numpy(decompressed_image.transpose(2, 0, 1))
                        else:
                            frame[feature_id] = torch.from_numpy(value[frame_idx])
                    elif isinstance(value, torch.Tensor):
                        frame[feature_id] = value[frame_idx]
                    else:
                        logging.warning(f"[AriaVRSExtractor] Could not add dataset key at {feature_id} due to unsupported type. Skipping ...")
                        continue

            frames.append(frame)
        return frames
    
    # TODO: this is largely similar to AriaVRSExtractor.define_features. Should use classmethod everywhere
    @staticmethod
    def define_features(episode_feats: dict) -> tuple:
        """
        Define features from episode_feats (output of process_episode), including a metadata section.

        Parameters
        ----------
        episode_feats : dict
            The output of the process_episode method, containing feature data.

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
                nested_features, nested_metadata = UCSDHDFExtractor.define_features(value)
                features.update({f"{key}.{nested_key}": nested_value for nested_key, nested_value in nested_features.items()})
                features.update({f"{key}.{nested_key}": nested_value for nested_key, nested_value in nested_metadata.items()})
            elif isinstance(value, np.ndarray):
                dtype = str(value.dtype)
                if "images" in key:
                    dtype = "image"
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
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        prestack: bool = False
    ):
        self.raw_path = raw_path if isinstance(raw_path, Path) else Path(raw_path)
        self.dataset_repo_id = dataset_repo_id
        self.image_writer_threads = image_writer_threads
        self.image_writer_processes = image_writer_processes
        self.prestack = prestack

        self.buffer = []

        assert os.path.isdir(self.raw_path), f"Raw path {self.raw_path} is not a directory"
        self.episode_list = list(self.raw_path.glob("*.hdf5"))

        processed_episode = UCSDHDFExtractor.process_episode(
            episode_path=self.episode_list[0],
            arm=H1_ARM_TYPE,
            prestack=self.prestack,
        )
        
        self.features, metadata = UCSDHDFExtractor.define_features(
            processed_episode,
        )

        self.robot_type = "ucsd_h1_bimanual"

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

        frames = UCSDHDFExtractor.extract_episode_frames(
            episode_path,
            features=self.features,
            image_compressed=False,
            arm=H1_ARM_TYPE,
            prestack=self.prestack
        )

        for i, frame in enumerate(frames):
            self.buffer.append(frame)

            if len(self.buffer) == EPISODE_LENGTH:
                for f in self.buffer:
                    self.dataset.add_frame(f)
                
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
                traceback.print_exc()
                continue
        
        self.buffer.clear()
        t0 = time.time()
        self.dataset.consolidate()
        elapsed_time = time.time() - t0

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

        output_dir = output_dir / name

        self.dataset = LeRobotDataset.create(
            repo_id=self.dataset_repo_id,
            fps=FPS,
            robot_type=self.robot_type,
            features=self.features,
            image_writer_threads=self.image_writer_threads,
            image_writer_processes=self.image_writer_processes,
            root=output_dir,
        )

        return self.dataset


def argument_parse():
    parser = argparse.ArgumentParser(description="Convert Aria VRS dataset to LeRobot-Robomimic hybrid and push to Hugging Face hub.")

    # Required arguments
    parser.add_argument("--name", type=str, required=True, help="Name for dataset")
    parser.add_argument("--raw-path", type=Path, required=True, help="Directory containing the raw HDF5 files.")
    parser.add_argument("--dataset-repo-id", type=str, required=True, help="Repository ID where the dataset will be stored.")

    # Optional arguments
    parser.add_argument("--description", type=str, default="Aria recorded dataset.", help="Description of the dataset.")
    parser.add_argument("--private", type=str2bool, default=False, help="Set to True to make the dataset private.")
    parser.add_argument("--push", type=str2bool, default=True, help="Set to True to push videos to the hub.")
    parser.add_argument("--license", type=str, default="apache-2.0", help="License for the dataset.")
    parser.add_argument("--image-compressed", type=str2bool, default=False, help="Set to True if the images are compressed.")
    parser.add_argument("--prestack", type=str2bool, default=True, help="Set to True to precompute action chunks.")

    # Performance tuning arguments
    parser.add_argument("--nproc", type=int, default=12, help="Number of image writer processes.")
    parser.add_argument("--nthreads", type=int, default=2, help="Number of image writer threads.")

    # Debugging and output configuration
    parser.add_argument("--output-dir", type=Path, default=Path(LEROBOT_HOME), help="Directory where the processed dataset will be stored. Defaults to LEROBOT_HOME.")

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

    # Initialize the dataset converter
    converter = DatasetConverter(
        raw_path=args.raw_path,
        dataset_repo_id=args.dataset_repo_id,
        image_writer_processes=args.nproc,
        image_writer_threads=args.nthreads,
        prestack=args.prestack
    )

    # Initialize the dataset
    converter.init_lerobot_dataset(output_dir=args.output_dir, name=Path(args.name))

    # Extract episodes
    converter.extract_episodes(episode_description=args.description)

    # Push the dataset to the Hugging Face Hub, if specified
    if args.push:
        converter.push_dataset_to_hub(
            dataset_tags=UCSDHDFExtractor.TAGS,
            private=args.private,
            license=args.license,
        )

if __name__ == "__main__":
    args = argument_parse()
    main(args)
