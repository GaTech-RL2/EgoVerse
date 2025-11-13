"""
Script to convert robomimic style datasets to LeRobot

Uses a config file to generically instantiate the parquet files.
"""

import argparse
import logging
import os
from pathlib import Path
import shutil
import traceback
import h5py
import json
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from utils import str2bool


class RobomimicHD5Extractor:
    TAGS = ["robomimic", "robotics", "hdf5"]

    @staticmethod
    def check_format(data: h5py.File, config: dict, ignore_episode_keys=True):
        """
        Check the format of the dataset based on the configuration file.
        Parameters
        ----------
        data : h5py.File
            HDF5 dataset object.
        config : dict
            Configuration dict describing the dataset structure.
        ignore_episode_keys : bool
            Ignore episode keys specified in config
        Raises
        ------
        ValueError
            If any key in the config is missing from the dataset.
        """
        episode_keys = []
        if not ignore_episode_keys:
            episode_keys = config["episode_keys"].copy()
        else:
            episode_keys = list(data["data"].keys())

        for episode_key in episode_keys:
            if episode_key not in data["data"]:
                raise ValueError(f"Missing episode key: {episode_key}")

            episode = data["data"][episode_key]
            for action_key in config["action_keys"]:
                if action_key not in episode:
                    raise ValueError(
                        f"Missing action key: {action_key} in {episode_key}"
                    )

            if "obs" not in episode:
                raise ValueError(f"Missing 'obs' key in {episode_key}")

            obs = episode["obs"]
            for obs_key in config["obs_keys"]:
                if obs_key not in obs:
                    raise ValueError(
                        f"Missing observation key: {obs_key} in {episode_key}"
                    )

    @staticmethod
    def extract_episode_frames(episode, config, image_compressed, encode_as_video):
        """
        Extract frames from a single episode.
        Parameters
        ----------
        episode : h5py.Group
            HDF5 group representing an episode.
        config : dict
            Configuration dict describing the dataset structure.
        image_compressed : bool
            Whether images are stored in a compressed format.
        encode_as_video : bool
            Whether to encode images as videos.
        Returns
        -------
        list of dict
            List of frames as dictionaries.
        """
        frames = []
        num_frames = episode["actions_joints"].shape[0]
        for frame_idx in range(num_frames):
            frame = {}
            for action_key in config["action_keys"]:
                frame[action_key] = torch.from_numpy(episode[action_key][frame_idx])

            # TODO: don't hard code so much stupid shit
            for obs_key in config["obs_keys"]:
                if obs_key in episode["obs"]:
                    if "img" or "image" in obs_key and image_compressed:
                        image = episode["obs"][obs_key][frame_idx]
                        if encode_as_video:
                            frame[f"obs.{obs_key}"] = torch.from_numpy(
                                cv2.imdecode(image, 1).transpose(2, 0, 1)
                            )
                        else:
                            frame[f"obs.{obs_key}"] = torch.from_numpy(image)
                    else:
                        frame[f"obs.{obs_key}"] = torch.from_numpy(
                            episode["obs"][obs_key][frame_idx]
                        )
            frames.append(frame)
        return frames

    @staticmethod
    def define_features(hdf5_file_path, config, image_compressed, encode_as_video):
        """
        Define features from an HDF5 file.
        Parameters
        ----------
        hdf5_file_path : Path
            The path to the HDF5 file.
        config : dict
            Configuration dict containing keys like 'obs_keys' and 'action_keys'.
        image_compressed : bool, optional
            Whether the images are compressed, by default True.
        encode_as_video : bool, optional
            Whether to encode images as video or as images, by default True.
        Returns
        -------
        dict[str, dict]
            A dictionary where keys are topic names and values are dictionaries
            containing feature information such as dtype, shape, and names.
        """
        topics = []
        features = {}

        # TODO: This seems a little hacky
        allowed_keys = set(
            ["obs/" + key for key in config.get("obs_keys", [])]
            + config.get("action_keys", [])
        )

        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            data_group = hdf5_file["data"]
            # TODO: This seems a little hacky
            hdf5_file = data_group[list(data_group.keys())[0]]
            hdf5_file.visititems(
                lambda name, obj: topics.append(name)
                if isinstance(obj, h5py.Dataset)
                else None
            )

            for topic in topics:
                if topic not in allowed_keys:
                    continue
                if "images" or "img" in topic.split("/"):
                    sample = hdf5_file[topic][0]
                    features[topic.replace("/", ".")] = {
                        "dtype": "video" if encode_as_video else "image",
                        "shape": cv2.imdecode(hdf5_file[topic][0], 1)
                        .transpose(2, 0, 1)
                        .shape
                        if image_compressed
                        else sample.shape,
                        "names": ["channel", "height", "width"],
                    }
                elif "compress_len" in topic.split("/"):
                    continue
                else:
                    features[topic.replace("/", ".")] = {
                        "dtype": str(hdf5_file[topic][0].dtype),
                        "shape": (topic_shape := hdf5_file[topic][0].shape),
                        "names": [
                            f"{topic.split('/')[-1]}_{k}" for k in range(topic_shape[0])
                        ],
                    }

        return features


class DatasetConverter:
    """
    A class to convert datasets to Lerobot format.
    Parameters
    ----------
    raw_path : Path or str
        The path to the raw dataset.
    config_path : Path or str
        Path to dataset key configuration
    dataset_repo_id : str
        The repository ID where the dataset will be stored.
    fps : int
        Frames per second for the dataset.
    robot_type : str, optional
        The type of robot, by default "".
    encode_as_video : bool, optional
        Whether to encode images as videos, by default True.
    image_compressed : bool, optional
        Whether the images are compressed, by default True.
    image_writer_processes : int, optional
        Number of processes for writing images, by default 0.
    image_writer_threads : int, optional
        Number of threads for writing images, by default 0.
    Methods
    -------
    extract_episode(episode_key, task_description='')
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
        config_path: Path | str,
        fps: int,
        robot_type: str,
        encode_as_video: bool = True,
        image_compressed: bool = True,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        ignore_episode_keys: bool = True,
    ):
        self.raw_path = raw_path if isinstance(raw_path, Path) else Path(raw_path)
        self.config_path = (
            config_path if isinstance(config_path, Path) else Path(config_path)
        )
        self.dataset_repo_id = dataset_repo_id
        self.fps = fps
        self.image_compressed = image_compressed
        self.image_writer_threads = image_writer_threads
        self.image_writer_processes = image_writer_processes
        self.encode_as_video = encode_as_video
        self.robot_type = robot_type

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - [%(name)s] - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.info(f"{'-' * 10} Robomimic HD5 -> Lerobot Converter {'-' * 10}")
        self.logger.info(f"Processing Robomimic HD5 dataset from {self.raw_path}")
        self.logger.info(f"Dataset will be stored in {self.dataset_repo_id}")
        self.logger.info(f"FPS: {self.fps}")
        self.logger.info(f"Robot type: {self.robot_type}")
        self.logger.info(f"Image compressed: {self.image_compressed}")
        self.logger.info(f"Encoding images as videos: {self.encode_as_video}")
        self.logger.info(f"#writer processes: {self.image_writer_processes}")
        self.logger.info(f"#writer threads: {self.image_writer_threads}")

        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.logger.info(
            "Initialized DatasetConverter with the following configuration:"
        )
        self.logger.info(json.dumps(self.config, indent=2))

        with h5py.File(self.raw_path, "r") as data:
            RobomimicHD5Extractor.check_format(data, self.config, ignore_episode_keys)
            if ignore_episode_keys:
                self.episode_keys = list(data["data"].keys())
            else:
                self.episode_keys = self.config["episode_keys"].copy()

        self.features = RobomimicHD5Extractor.define_features(
            raw_path, self.config, self.image_compressed, self.encode_as_video
        )

    def extract_episode(self, episode_key, episode, task_description=""):
        """
        Extract and process a single episode.
        Parameters
        ----------
        episode_key : str
            The key for the episode.
        episode : h5py.Group
            The HDF5 group representing the episode.
        task_description : str, optional
            A description of the task associated with the episode (default is an empty string).
        """

        frames = RobomimicHD5Extractor.extract_episode_frames(
            episode, self.config, self.image_compressed, self.encode_as_video
        )

        for frame in frames:
            self.dataset.add_frame(frame)

        self.logger.info(
            f"Saving Episode {episode_key} with Description: {task_description} ..."
        )
        self.dataset.save_episode(task=task_description)

    def extract_episodes(self, episode_description={}):
        """
        Extracts episodes from a dataset and processes them.
        Parameters
        ----------
        episode_description : dict, optional
            A dictionary of descriptions of the task to be passed to the extract_episode method (default is '').
        """

        with h5py.File(self.raw_path, "r") as data:
            for episode_key in self.episode_keys:
                episode = data["data"][episode_key]
                self.extract_episode(
                    episode_key,
                    episode,
                    task_description=episode_description.get(episode_key, ""),
                )

        self.dataset.consolidate()

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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert robomimic dataset to Lerobot format."
    )
    parser.add_argument(
        "--name", type=str, required=True, help="Name to store dataset as"
    )
    parser.add_argument(
        "--raw-path", type=Path, required=True, help="Path to the raw HDF5 dataset."
    )
    parser.add_argument(
        "--dataset-repo-id", type=str, required=True, help="Hugging Face repository ID."
    )
    parser.add_argument(
        "--config-path", type=Path, required=True, help="Path to the JSON config file."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the processed dataset.",
    )
    parser.add_argument(
        "--fps", type=int, required=True, help="Frames per second for the dataset."
    )
    parser.add_argument(
        "--image-compressed",
        action="store_true",
        help="Flag to indicate if images are compressed.",
    )
    parser.add_argument(
        "--encode-as-video", action="store_true", help="Flag to encode images as video."
    )
    parser.add_argument(
        "--nproc", type=int, default=10, help="Number of image writer processes."
    )
    parser.add_argument(
        "--nthreads", type=int, default=5, help="Number of image writer threads."
    )
    parser.add_argument(
        "--push", action="store_true", help="Push dataset to Hugging Face Hub."
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--license", type=str, default="apache-2.0", help="License for the dataset."
    )
    parser.add_argument(
        "--overcap",
        action="store_true",
        help="Flag for using 'overcap' partition in SLURM.",
    )
    parser.add_argument(
        "--gpus-per-node", type=int, default=1, help="Number of GPUs per SLURM node."
    )
    parser.add_argument(
        "--num-nodes", type=int, default=1, help="Number of SLURM nodes."
    )
    parser.add_argument(
        "--partition", type=str, default="hoffman-lab", help="SLURM partition name."
    )
    parser.add_argument(
        "--ignore_episode_keys",
        action="store_true",
        help="Use episode keys inside config",
    )
    parser.add_argument("--robot-type", type=str, default="eve", help="Robot type")
    return parser.parse_args()


def main(args):
    print(
        args.encode_as_video,
        "-------------------------------------------------------------------------------------------------------",
    )

    converter = DatasetConverter(
        raw_path=args.raw_path,
        dataset_repo_id=args.dataset_repo_id,
        config_path=args.config_path,
        fps=args.fps,
        robot_type=args.robot_type,
        image_compressed=args.image_compressed,
        encode_as_video=args.encode_as_video,
        image_writer_processes=args.nproc,
        image_writer_threads=args.nthreads,
        ignore_episode_keys=args.ignore_episode_keys,
    )
    converter.init_lerobot_dataset(Path(args.output_dir), Path(args.name))

    # TODO: add description in config
    converter.extract_episodes()

    if args.push:
        converter.push_dataset_to_hub(
            dataset_tags=RobomimicHD5Extractor.TAGS,
            private=args.private,
            push_videos=True,
            license=args.license,
        )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
