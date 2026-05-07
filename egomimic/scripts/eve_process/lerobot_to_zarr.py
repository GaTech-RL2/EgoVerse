"""
Convert an EVE/faive LeRobot parquet dataset to per-episode Zarr v3 stores.

This is a re-packaging step: trajectories are written byte-for-byte from the
parquet, no math is applied (all coordinate-frame transforms have already
been done by faive2lerobot). Math identity vs. the source LeRobot dataset
is therefore guaranteed by construction.

Inputs:
    --root: LeRobot dataset root containing data/chunk-XXX/episode_XXX.parquet
            and meta/info.json (e.g. .../cup_lerobot_base_frame).
    --output-dir: directory to write per-episode .zarr/ stores.
    --max-episodes: optional cap (defaults to all episodes).
    --embodiment: embodiment name string (default: eve_bimanual).
    --task-name / --task-description: written to zarr metadata.

Output layout:
    <output-dir>/<episode_hash>.zarr/
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

logger = logging.getLogger(__name__)


# Schema mapping: parquet column name -> zarr key (kept identical so DataSchematic
# and Eve.get_keymap entries match the LeRobot column names verbatim).
_NUMERIC_COLS = (
    "observations.state.ee_pose",
    "observations.state.cartesian_arm",
    "observations.state.joints_hand",
    "actions_cartesian",
    "actions_joints",
)
_IMAGE_COLS = ("observations.images.aria_rgb_cam",)


def _decode_image_value(value) -> np.ndarray:
    """Decode a single LeRobot image cell to (H, W, 3) uint8."""
    # LeRobot's pyarrow "image" dtype materializes as either a dict with
    # {'bytes': b'...', 'path': None} or raw bytes depending on writer.
    if isinstance(value, dict):
        data = value.get("bytes")
        if data is None and value.get("path"):
            data = Path(value["path"]).read_bytes()
    elif isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
    else:
        raise TypeError(f"Unsupported image cell type: {type(value)}")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _stack_numeric(values) -> np.ndarray:
    """Stack a Series whose elements are scalars or array-like into one ndarray."""
    arr = np.asarray(list(values))
    if arr.dtype == object:
        # Heterogeneous (typically the prestacked action chunks)
        arr = np.stack([np.asarray(v) for v in values], axis=0)
    return arr


def _output_image_key(parquet_col: str) -> str:
    # Rename the cup dataset's wrist-cam-style key so it lines up with the
    # canonical front-view used by HPT trunk: front_img_1.
    if parquet_col == "observations.images.aria_rgb_cam":
        return "observations.images.front_img_1"
    return parquet_col


def convert_episode(
    parquet_path: Path,
    output_path: Path,
    embodiment: str,
    embodiment_id: int,
    fps: int,
    task_name: str,
    task_description: str,
    chunk_timesteps: int = 100,
) -> Path:
    df = pd.read_parquet(parquet_path)
    total_frames = len(df)
    if total_frames == 0:
        raise ValueError(f"Empty parquet: {parquet_path}")

    numeric_data: dict[str, np.ndarray] = {}
    for col in _NUMERIC_COLS:
        if col not in df.columns:
            logger.warning("Column %s missing in %s; skipping", col, parquet_path)
            continue
        numeric_data[col] = _stack_numeric(df[col].values)

    # Embodiment id stored per-frame (HPT batch carries it; ZarrEpisode reads
    # zarr key 'metadata.embodiment').
    numeric_data["metadata.embodiment"] = np.full(
        (total_frames, 1), embodiment_id, dtype=np.int32
    )

    image_data: dict[str, np.ndarray] = {}
    for col in _IMAGE_COLS:
        if col not in df.columns:
            logger.warning("Image column %s missing in %s; skipping", col, parquet_path)
            continue
        decoded = np.stack([_decode_image_value(v) for v in df[col].values], axis=0)
        image_data[_output_image_key(col)] = decoded

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ZarrWriter.create_and_write(
        episode_path=output_path,
        numeric_data=numeric_data,
        image_data=image_data,
        embodiment=embodiment,
        fps=fps,
        task_name=task_name,
        task_description=task_description,
        chunk_timesteps=chunk_timesteps,
    )
    logger.info("Wrote zarr episode: %s (%d frames)", output_path, total_frames)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embodiment", type=str, default="eve_bimanual")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None,
                        help="Override fps; defaults to value from meta/info.json.")
    parser.add_argument("--task-name", type=str, default="put_cup_on_saucer")
    parser.add_argument("--task-description", type=str, default="")
    parser.add_argument("--chunk-timesteps", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    info_path = args.root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    fps = args.fps or int(info.get("fps", 30))

    embodiment_id = get_embodiment_id(args.embodiment)

    data_dir = args.root / "data"
    parquet_paths = sorted(data_dir.rglob("episode_*.parquet"))
    if args.max_episodes is not None:
        parquet_paths = parquet_paths[: args.max_episodes]
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files under {data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for pq in parquet_paths:
        # Use the parquet stem as the episode hash so each episode has a stable
        # filesystem name (e.g. episode_000000 -> episode_000000.zarr).
        episode_hash = pq.stem
        out = args.output_dir / f"{episode_hash}.zarr"
        convert_episode(
            parquet_path=pq,
            output_path=out,
            embodiment=args.embodiment,
            embodiment_id=embodiment_id,
            fps=fps,
            task_name=args.task_name,
            task_description=args.task_description,
            chunk_timesteps=args.chunk_timesteps,
        )


if __name__ == "__main__":
    main()
