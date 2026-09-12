"""Tiny synthetic zarr episodes per vendor, written through the real ZarrWriter.

Vendor differences that matter to the data pipeline are reproduced (camera set,
stride, head pose); everything else is random but non-degenerate so norm stats
are finite.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.zarr.zarr_writer import ZarrWriter


@dataclass(frozen=True)
class Vendor:
    embodiment: str
    stride: int
    cameras: tuple[str, ...]  # zarr image keys without the "images." prefix
    has_head_pose: bool


VENDORS: dict[str, Vendor] = {
    "eva": Vendor("eva_bimanual", 1, ("front_1", "right_wrist", "left_wrist"), False),
    "aria": Vendor("human_bimanual", 3, ("front_1",), True),
    "mecka": Vendor("human_bimanual", 1, ("front_1",), True),
    "scale": Vendor("human_bimanual", 1, ("front_1",), True),
}


def _quat_wxyz(T: int, rng: np.random.Generator) -> np.ndarray:
    q = R.from_rotvec(rng.normal(0, 0.05, (T, 3))).as_quat()  # xyzw
    return np.concatenate([q[:, 3:4], q[:, :3]], axis=1)


def _pose(T: int, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate(
        [rng.normal(0, 0.1, (T, 3)), _quat_wxyz(T, rng)], axis=1
    ).astype(np.float64)


def write_episode(
    root: Path, vendor: str, *, T: int = 48, H: int = 64, W: int = 64, seed: int = 0
) -> Path:
    v = VENDORS[vendor]
    rng = np.random.default_rng(seed)
    K = np.array(
        [[100.0, 0.0, W / 2, 0.0], [0.0, 100.0, H / 2, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    images = {
        f"images.{c}": rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8)
        for c in v.cameras
    }
    numeric = {"left.obs_ee_pose": _pose(T, rng), "right.obs_ee_pose": _pose(T, rng)}
    extrinsics = None
    if v.embodiment == "eva_bimanual":
        numeric.update(
            {
                "left.cmd_ee_pose": _pose(T, rng),
                "right.cmd_ee_pose": _pose(T, rng),
                "left.obs_gripper": rng.uniform(0, 1, (T, 1)),
                "right.obs_gripper": rng.uniform(0, 1, (T, 1)),
                "left.cmd_gripper": rng.uniform(0, 1, (T, 1)),
                "right.cmd_gripper": rng.uniform(0, 1, (T, 1)),
            }
        )
        extrinsics = {"left": np.eye(4), "right": np.eye(4)}
    if v.has_head_pose:
        numeric["obs_head_pose"] = _pose(T, rng)
    return ZarrWriter.create_and_write(
        root / f"{vendor}_{seed:02d}.zarr",
        numeric_data=numeric,
        image_data=images,
        embodiment=v.embodiment,
        fps=30,
        task_name="synthetic",
        annotations=[
            ("pick up the red cube", 0, T // 2),
            ("place it in the bin", T // 2, T - 1),
        ],
        intrinsics={"front_1": K},
        extrinsics=extrinsics,
    )
