import argparse
import ctypes
import dataclasses
import gc
import logging
import traceback
from pathlib import Path

import numpy as np

from egomimic.rldb.embodiment.human import ARIA_INTRINSICS
from egomimic.rldb.zarr.required_metadata import (
    add_required_metadata_args,
    check_writer_metadata,
    required_metadata_from_args,
)
from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.scripts.aria_process.aria_utils import AriaVRSExtractor
from egomimic.scripts.aria_process.gen2_filter import Gen2FilterConfig
from egomimic.utils.aws.aws_sql import timestamp_ms_to_episode_hash
from egomimic.utils.type_utils import str2bool
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
    extract_episode(episode_path, required_metadata=..., output_dir='.', dataset_name='', chunk_timesteps=100)
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
        mano_n_iters: int = 2400,
        mano_lr: float = 0.05,
        mano_beta_reg: float = 0.01,
        mano_chunk_size: int = 512,
        gen2_source: str = "mps",
        split_demos: bool = True,
        intra_demo_drop: bool = True,
        gen2_filter: Gen2FilterConfig | None = None,
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
        self.gen2_source = gen2_source
        self.split_demos = split_demos
        self.intra_demo_drop = intra_demo_drop
        self.gen2_filter = gen2_filter or Gen2FilterConfig()

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
        *,
        required_metadata: dict,
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
        required_metadata : dict
            task_name/operator/lab/scene/objects kwargs for create_and_write (see
            egomimic.rldb.zarr.required_metadata.required_metadata_from_args).
        Returns
        -------
        (zarr_path, mp4_path), or with split_demos on a Gen 2 recording a
        list of dicts (episode_hash, zarr_path, mp4_path, demo info), one per
        demo.
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
            gen2_source=self.gen2_source,
            split_demos=self.split_demos,
            intra_demo_drop=self.intra_demo_drop,
            gen2_filter=self.gen2_filter,
        )
        demo_index = episode_feats.pop("_demo_index", None)
        demos = episode_feats.pop("_gen2_demos", None)
        attrs = episode_feats.pop("_attrs", None)
        write = dict(
            episode_path=episode_path,
            required_metadata=required_metadata,
            task_description=task_description,
            output_dir=output_dir,
            chunk_timesteps=chunk_timesteps,
        )
        if demos is None:
            return self._write_episode(
                episode_feats, episode_name, attrs=attrs, **write
            )

        demos = [info for info in demos if info["first_kept_utc_ns"] is not None]
        hashes = gen2_demo_hashes(
            [info["first_kept_utc_ns"] for info in demos], parent_hash=episode_name
        )
        results = []
        for info, name in zip(demos, hashes):
            sel = demo_index == info["demo_index"]
            zarr_path, mp4_path = self._write_episode(
                {key: value[sel] for key, value in episode_feats.items()},
                name,
                attrs={
                    **(attrs or {}),
                    "parent_episode_hash": episode_name,
                    "demo_index": info["demo_index"],
                    "demo_count": len(demos),
                    "demo_start_s": round(info["start_s"], 3),
                    "demo_end_s": round(info["end_s"], 3),
                    "demo_span_frames": info["span_frames"],
                    "demo_error_frames": info["error_frames"],
                    "intra_demo_drop": self.intra_demo_drop,
                },
                **write,
            )
            results.append(
                {
                    **info,
                    "episode_hash": name,
                    "zarr_path": zarr_path,
                    "mp4_path": mp4_path,
                }
            )
        return results

    def _write_episode(
        self,
        episode_feats,
        name,
        *,
        attrs,
        episode_path,
        required_metadata,
        task_description,
        output_dir,
        chunk_timesteps,
    ):
        """Write one zarr (and optional preview mp4); returns (zarr, mp4)."""
        # Gen 2 is its own embodiment (see EMBODIMENT.ARIA_GEN2_*).
        embodiment = self.embodiment
        if (attrs or {}).get("device") == "aria_gen2":
            embodiment = embodiment.replace("human_", "aria_gen2_", 1)
        numeric_data, image_data = {}, {}
        for key, value in episode_feats.items():
            target = image_data if "images" in key else numeric_data
            target[self.feats_to_zarr_keys.get(key, key)] = value

        zarr_path = ZarrWriter.create_and_write(
            episode_path=output_dir / f"{name}.zarr",
            numeric_data=numeric_data if numeric_data else None,
            image_data=image_data if image_data else None,
            fps=self.fps,
            embodiment=embodiment,
            task_description=task_description,
            chunk_timesteps=chunk_timesteps,
            intrinsics={"front_1": ARIA_INTRINSICS},
            vendor_recording_id=Path(episode_path).stem,
            metadata_override=attrs,
            **required_metadata,
        )
        if self.save_mp4:
            mp4_path = output_dir / f"{name}.mp4"
            images_tchw = np.asarray(image_data["images.front_1"]).transpose(0, 3, 1, 2)
            save_preview_mp4(images_tchw, mp4_path, self.fps, half_res=False)
        else:
            mp4_path = None
        return zarr_path, mp4_path


def gen2_demo_hashes(first_kept_utc_ns: list[int], parent_hash: str) -> list[str]:
    """Episode hash per demo: the UTC time of its first kept frame.

    Hashes are unique and never equal the parent recording's hash; a clash is
    resolved by moving the later demo forward one millisecond.
    """
    taken = {parent_hash}
    hashes = []
    for utc_ns in first_kept_utc_ns:
        ms = int(round(utc_ns / 1e6))
        while timestamp_ms_to_episode_hash(ms) in taken:
            ms += 1
        hashes.append(timestamp_ms_to_episode_hash(ms))
        taken.add(hashes[-1])
    return hashes


def _gen2_filter_from_args(args) -> Gen2FilterConfig:
    """Gen 2 filter thresholds: optional YAML or override dict (from the Ray
    path), then the --min-demo-s override."""
    path = getattr(args, "gen2_filter_config", None)
    cfg = Gen2FilterConfig.from_yaml(path) if path else Gen2FilterConfig()
    cfg = dataclasses.replace(cfg, **(getattr(args, "gen2_filter", None) or {}))
    min_demo_s = getattr(args, "min_demo_s", None)
    if min_demo_s is not None:
        cfg = dataclasses.replace(cfg, min_demo_s=min_demo_s)
    return cfg


def main(args) -> None:
    """Convert Eva HDF5 dataset to Zarr episodes.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (same shape as eva_to_lerobot).
    """

    try:
        episode_hash = timestamp_ms_to_episode_hash(Path(args.raw_path).stem)
        required_metadata = required_metadata_from_args(args)
        check_writer_metadata(required_metadata)

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
            mano_n_iters=getattr(args, "mano_iters", 2400),
            mano_lr=getattr(args, "mano_lr", 0.05),
            mano_beta_reg=getattr(args, "mano_beta_reg", 0.01),
            mano_chunk_size=getattr(args, "mano_chunk_size", 512),
            gen2_source=getattr(args, "gen2_source", "mps"),
            split_demos=getattr(args, "split_demos", True),
            intra_demo_drop=getattr(args, "intra_demo_drop", True),
            gen2_filter=_gen2_filter_from_args(args),
        )

        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        # With split_demos on a Gen 2 recording this is a list of per-demo dicts.
        return converter.extract_episode(
            episode_path=Path(args.raw_path),
            task_description=args.task_description,
            output_dir=Path(args.output_dir),
            dataset_name=episode_hash,
            required_metadata=required_metadata,
        )
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
        default=None,
        help="Task name of the data (must match the DB row's task).",
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
    add_required_metadata_args(parser)
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
        "--mano-iters",
        type=int,
        default=2400,
        help="Adam iterations per MANO fit chunk.",
    )
    parser.add_argument(
        "--mano-lr", type=float, default=0.05, help="MANO fit learning rate."
    )
    parser.add_argument(
        "--mano-beta-reg",
        type=float,
        default=0.01,
        help="L2 regularization on MANO shape betas.",
    )
    parser.add_argument(
        "--mano-chunk-size",
        type=int,
        default=512,
        help="Frames per batched MANO fit (bounds memory).",
    )
    parser.add_argument(
        "--gen2-source",
        choices=["auto", "mps", "on_device"],
        default="mps",
        help="Aria Gen 2 only: read MPS outputs (default), the VRS's on-device "
        "hand tracking / VIO / eye gaze, or MPS when present (auto).",
    )
    parser.add_argument(
        "--split-demos",
        type=str2bool,
        default=True,
        help="Aria Gen 2 only: write one episode per demo (split at resets and "
        "the opening calibration) instead of one per recording.",
    )
    parser.add_argument(
        "--intra-demo-drop",
        type=str2bool,
        default=True,
        help="Aria Gen 2 only: drop short in-demo errors (tracking gaps, head "
        "sweeps, hand at headset, untracked hands). False keeps them with NaN "
        "poses and an obs_frame_valid flag (needs --convert-mano False).",
    )
    parser.add_argument(
        "--gen2-filter-config",
        type=Path,
        default=None,
        help="Aria Gen 2 only: YAML of Gen2FilterConfig overrides (thresholds "
        "for the calibration/reset/demo filter).",
    )
    parser.add_argument(
        "--min-demo-s",
        type=float,
        default=None,
        help="Aria Gen 2 only: pieces between resets shorter than this are not "
        "demos (overrides the filter config; default 5 s).",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = argument_parse()
    print(main(args))
