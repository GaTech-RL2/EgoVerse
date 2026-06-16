import argparse
import ctypes
import gc
import logging
import traceback
from pathlib import Path

import numpy as np

from egomimic.rldb.embodiment.human import ARIA_INTRINSICS
from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.scripts.aria_process.aria_utils import AriaVRSExtractor
from egomimic.utils.aws.aws_sql import timestamp_ms_to_episode_hash
from egomimic.utils.egomimicUtils import str2bool
from egomimic.utils.video_utils import save_preview_mp4

logger = logging.getLogger(__name__)


class DatasetConverter:
    """
    A class to convert Aria VRS dataset to Zarr episodes.
    Parameters
    ----------
    raw_path : Path or str
        The path to the raw dataset.
    fps : int
        Frames per second for the dataset.
    arm : str, optional
        The arm to process (e.g., 'left', 'right', or 'bimanual'), by default "".
    save_mp4 : bool, optional
        Whether to save a MP4 of the episode, by default False.
    image_compressed : bool, optional
        Whether the images are compressed, by default True.
    Methods
    -------
    extract_episode(episode_path, task_name='', output_dir='.', dataset_name='', chunk_timesteps=100)
        Extracts frames from a single episode and saves it with a description.
    main(args)
        Main function to convert the dataset.
    argument_parse()
        Parses the command-line arguments.
    """

    def __init__(
        self,
        raw_path: Path | str,
        fps: int,
        arm: str = "",
        save_mp4: bool = False,
        image_compressed: bool = True,
        debug: bool = False,
        height: int = 480,
        width: int = 640,
        convert_mano: bool = True,
        mano_model_dir: str | None = None,
        mano_device: str | None = None,
        mano_n_iters: int = 400,
        mano_lr: float = 0.02,
        mano_beta_reg: float = 0.01,
        mano_chunk_size: int = 512,
    ):
        self.raw_path = raw_path if isinstance(raw_path, Path) else Path(raw_path)
        self.fps = fps
        self.arm = arm
        self.image_compressed = image_compressed
        self.save_mp4 = save_mp4
        self.height = height
        self.width = width
        self.convert_mano = convert_mano
        self.mano_model_dir = mano_model_dir
        self.mano_device = mano_device
        self.mano_n_iters = mano_n_iters
        self.mano_lr = mano_lr
        self.mano_beta_reg = mano_beta_reg
        self.mano_chunk_size = mano_chunk_size

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
        self.logger.info(f"FPS: {self.fps}")
        self.logger.info(f"Arm: {self.arm}")
        self.logger.info(f"Image compressed: {self.image_compressed}")
        self.logger.info(f"Save MP4: {self.save_mp4}")
        self.logger.info(
            f"MANO conversion: {self.convert_mano}"
            + (
                f" (iters={self.mano_n_iters}, lr={self.mano_lr}, "
                f"beta_reg={self.mano_beta_reg}, chunk={self.mano_chunk_size}, "
                f"device={self.mano_device or 'auto'})"
                if self.convert_mano
                else ""
            )
        )

        self._mp4_path = None  # set from main() if --save-mp4
        self._mp4_writer = None  # lazy-initialized in extract_episode()
        self.episode_list = list(self.raw_path.glob("*.vrs"))

        self.feats_to_zarr_keys = {}

        if self.arm == "both":
            self.embodiment = "human_bimanual"
        elif self.arm == "right":
            self.embodiment = "human_right_arm"
        elif self.arm == "left":
            self.embodiment = "human_left_arm"
        else:
            raise ValueError(
                f"Unsupported arm: {self.arm!r} (expected 'both', 'right', or 'left')"
            )

    def extract_episode(
        self,
        episode_path,
        task_name: str = "",
        task_description: str = "",
        output_dir: Path = Path("."),
        dataset_name: str = "",
        chunk_timesteps: int = 100,
    ):
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
        episode_name = dataset_name

        episode_feats = AriaVRSExtractor.process_episode(
            episode_path=episode_path,
            arm=self.arm,
            height=self.height,
            width=self.width,
            convert_mano=self.convert_mano,
            mano_model_dir=self.mano_model_dir,
            mano_device=self.mano_device,
            mano_n_iters=self.mano_n_iters,
            mano_lr=self.mano_lr,
            mano_beta_reg=self.mano_beta_reg,
            mano_chunk_size=self.mano_chunk_size,
        )
        numeric_data = {}

        image_data = {}
        for key, value in episode_feats.items():
            if "images" in key:
                if key in self.feats_to_zarr_keys:
                    image_data[self.feats_to_zarr_keys[key]] = value
                else:
                    image_data[key] = value
            else:
                if key in self.feats_to_zarr_keys:
                    numeric_data[self.feats_to_zarr_keys[key]] = value
                else:
                    numeric_data[key] = value

        zarr_path = ZarrWriter.create_and_write(
            episode_path=output_dir / f"{episode_name}.zarr",
            numeric_data=numeric_data if numeric_data else None,
            image_data=image_data if image_data else None,
            fps=self.fps,
            embodiment=self.embodiment,
            task_name=task_name,
            task_description=task_description,
            chunk_timesteps=chunk_timesteps,
            intrinsics={"front_1": ARIA_INTRINSICS},
        )
        if self.save_mp4:
            mp4_path = output_dir / f"{episode_name}.mp4"
            images_tchw = np.asarray(image_data["images.front_1"]).transpose(0, 3, 1, 2)
            save_preview_mp4(images_tchw, mp4_path, self.fps, half_res=False)
        else:
            mp4_path = None
        return zarr_path, mp4_path


def main(args) -> None:
    """Convert Eva HDF5 dataset to Zarr episodes.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (same shape as eva_to_lerobot).
    """

    try:
        episode_hash = timestamp_ms_to_episode_hash(Path(args.raw_path).stem)

        converter = DatasetConverter(
            raw_path=Path(args.raw_path),
            fps=args.fps,
            arm=args.arm,
            image_compressed=args.image_compressed,
            save_mp4=args.save_mp4,
            debug=args.debug,
            # getattr defaults: callers that build their own Namespace (e.g.
            # run_conversion-style orchestrators) keep working without the new fields.
            convert_mano=getattr(args, "convert_mano", True),
            mano_model_dir=getattr(args, "mano_model_dir", None),
            mano_device=getattr(args, "mano_device", None),
            mano_n_iters=getattr(args, "mano_iters", 400),
            mano_lr=getattr(args, "mano_lr", 0.02),
            mano_beta_reg=getattr(args, "mano_beta_reg", 0.01),
            mano_chunk_size=getattr(args, "mano_chunk_size", 512),
        )

        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        zarr_path, mp4_path = converter.extract_episode(
            episode_path=Path(args.raw_path),
            task_name=args.task_name,
            task_description=args.task_description,
            output_dir=Path(args.output_dir),
            dataset_name=episode_hash,
        )
        return zarr_path, mp4_path
    except Exception:
        logger.error(
            "Error converting %s:\n%s", Path(args.raw_path), traceback.format_exc()
        )
        return None


def argument_parse():
    parser = argparse.ArgumentParser(
        description="Convert Aria VRS dataset to LeRobot-Robomimic hybrid and push to Hugging Face hub."
    )

    # Required arguments
    parser.add_argument(
        "--raw-path",
        type=Path,
        required=True,
        help="Directory containing the vrs, vrs_json, and the processed mps folder.",
    )
    parser.add_argument(
        "--fps", type=int, required=True, help="Frames per second for the dataset."
    )
    # Optional arguments
    parser.add_argument(
        "--task-name",
        type=str,
        default="Aria recorded dataset.",
        help="Task name of the data.",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default="Aria recorded dataset.",
        help="Task description of the data.",
    )
    parser.add_argument(
        "--arm",
        type=str,
        choices=["left", "right", "both"],
        default="both",
        help="Specify the arm for processing.",
    )
    parser.add_argument(
        "--image-compressed",
        type=str2bool,
        default=False,
        help="Set to True if the images are compressed.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the processed dataset will be stored. Defaults to LEROBOT_HOME.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Store only 2 episodes for debug purposes."
    )

    parser.add_argument(
        "--save-mp4",
        action="store_true",
        help="If enabled, save a single half-resolution MP4 with all frames across episodes.",
    )

    parser.add_argument(
        "--convert-mano",
        type=str2bool,
        default=True,
        help="Fit MANO to the Aria keypoints: zarr gets MANO keypoints under "
        "<side>.obs_keypoints and raw Aria under <side>.obs_aria_keypoints. "
        "False = legacy schema (raw Aria under <side>.obs_keypoints).",
    )
    parser.add_argument(
        "--mano-model-dir",
        type=str,
        default=None,
        help="Directory with MANO_LEFT.pkl / MANO_RIGHT.pkl. Defaults to <repo>/external_ckpts/mano.",
    )
    parser.add_argument(
        "--mano-device",
        type=str,
        default=None,
        help="Device for the MANO fit (e.g. cuda, mps, cpu). Default: auto (cuda > mps > cpu).",
    )
    parser.add_argument(
        "--mano-iters", type=int, default=400, help="Adam iterations per MANO fit chunk."
    )
    parser.add_argument("--mano-lr", type=float, default=0.02, help="MANO fit learning rate.")
    parser.add_argument(
        "--mano-beta-reg", type=float, default=0.01, help="L2 regularization on MANO shape betas."
    )
    parser.add_argument(
        "--mano-chunk-size",
        type=int,
        default=512,
        help="Frames per batched MANO fit (bounds memory).",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = argument_parse()
    zarr_path, mp4_path = main(args)
    print(zarr_path, mp4_path)
