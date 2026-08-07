"""Fit MANO to Aria's 21-keypoint hand data and render a side-by-side viz mp4.

Pipeline:
1. Pull first aria episode via SQL.
2. Build MultiDataset with Aria.get_keymap("keypoints") + transform_list
   "keypoints_headframe_ypr" -> "actions_keypoints" (left wrist+kp, right wrist+kp).
3. For the first N frames, stack aria keypoints into a batched tensor and fit
   MANO_RIGHT / MANO_LEFT to them in parallel (batched Adam).
4. Permute MANO's otaheri joint order (wrist, index*3, middle*3, pinky*3,
   ring*3, thumb*3, then 5 tips) to the canonical MANO ordering used by
   Human.FINGER_EDGES (wrist, thumb*4, index*4, middle*4, ring*4, pinky*4).
5. Render side-by-side: left half = Aria viz with Aria.FINGER_EDGES; right
   half = fitted MANO viz with Human (=Mecka) FINGER_EDGES.
6. Write mp4 to scratch/aria_to_mano.mp4.

Prerequisites (not checked in):
- MANO PyTorch loader: clone github.com/otaheri/MANO into <repo>/external/MANO/
  and `uv pip install -e external/MANO --no-build-isolation`.
- MANO model files MANO_RIGHT.pkl and MANO_LEFT.pkl: register and download
  from https://mano.is.tue.mpg.de/ (academic non-commercial license; not
  redistributable, hence not in this repo). Place them at:
      <repo>/external_ckpts/mano/MANO_RIGHT.pkl
      <repo>/external_ckpts/mano/MANO_LEFT.pkl
- Python 3.11 + numpy>=1.20 compat patches for chumpy 0.70:
  s/inspect.getargspec/inspect.getfullargspec/ in chumpy/ch.py, and replace
  the `from numpy import bool, int, ...` line in chumpy/__init__.py with
  `from numpy import nan, inf`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio_ffmpeg
import mediapy as mpy
import numpy as np
import torch

from egomimic.rldb.embodiment.human import (
    ARIA_FINGER_EDGE_RANGES,
    ARIA_FINGER_EDGES,
    Human,
)
from egomimic.scripts.aria_process.aria_utils import fit_mano_to_aria_batched
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, S3EpisodeResolver
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

mpy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

REPO_ROOT = Path(__file__).resolve().parents[2].parent
SCRATCH_DIR = REPO_ROOT / "scratch"
CACHE_DIR = SCRATCH_DIR / "aria_to_mano_cache"
OUT_MP4 = SCRATCH_DIR / "aria_to_mano.mp4"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NUM_FRAMES = 100
NUM_FIT_ITERS = 400
LR = 0.02
BETA_REG = 0.01

# Correspondence tables (ARIA_TO_OTAHERI, OTAHERI_TO_CANONICAL) and the
# batched fit live in egomimic/scripts/aria_process/aria_utils.py, shared
# with the aria_to_zarr pipeline.


def fetch_episode_loader():
    load_env()
    engine = create_default_engine()
    df = episode_table_to_df(engine)
    aria_df = df[df["embodiment"].str.startswith("aria", na=False)]
    if len(aria_df) == 0:
        raise RuntimeError("No aria episodes found")
    episode_hash = aria_df.iloc[0]["episode_hash"]
    print(f"Aria episode: {episode_hash} (of {len(aria_df)})")

    key_map = Human.get_keymap(keymap_mode="keypoints")
    transform_list = Human.get_transform_list(mode="keypoints_headframe_ypr", stride=3)
    resolver = S3EpisodeResolver(
        str(CACHE_DIR), key_map=key_map, transform_list=transform_list
    )
    filters = DatasetFilter(
        filter_lambdas=[f"lambda row: row['episode_hash'] == {episode_hash!r}"]
    )
    dataset = MultiDataset._from_resolver(
        resolver, filters=filters, sync_from_s3=True, mode="total"
    )
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


def gather_aria_keypoints(loader, n_frames):
    """Collect first n_frames worth of (image, left_kp, right_kp, batch) tuples.

    Aria's batched actions_keypoints layout (after keypoints_headframe_ypr):
      [left_wrist_xyz(3), left_wrist_ypr(3), left_kp(63), right_wrist_xyz(3),
       right_wrist_ypr(3), right_kp(63)] = 138 per timestep, shape (T_chunk, 138).
    We take chunk[0] (current timestep).
    """
    rows = []
    for i, batch in enumerate(loader):
        if i >= n_frames:
            break
        akp = batch["actions_keypoints"][0, 0]  # (138,)
        left_kp = akp[6:6 + 63].reshape(21, 3)
        right_kp = akp[75:75 + 63].reshape(21, 3)
        rows.append(
            {
                "batch": batch,
                "left_aria_kp": left_kp.float(),
                "right_aria_kp": right_kp.float(),
            }
        )
    print(f"Collected {len(rows)} frames")
    return rows


FINGER_LEGEND = [
    ("thumb",  (255, 100, 100)),
    ("index",  (100, 255, 100)),
    ("middle", (100, 100, 255)),
    ("ring",   (255, 255, 100)),
    ("pinky",  (255, 100, 255)),
]


def _add_title_bar(panel, lines, bar_h=70):
    """Prepend a black bar at the top of `panel` with `lines` of white text."""
    w = panel.shape[1]
    bar = np.zeros((bar_h, w, 3), dtype=panel.dtype)
    y = 22
    for line in lines:
        cv2.putText(bar, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 22
    return np.concatenate([bar, panel], axis=0)


def _draw_legend(panel):
    """Draw a finger color legend in the top-right corner of `panel`."""
    x0 = panel.shape[1] - 130
    y = 20
    cv2.putText(panel, "fingers:", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    y += 18
    for name, color in FINGER_LEGEND:
        cv2.rectangle(panel, (x0, y - 10), (x0 + 14, y + 2), color, -1)
        cv2.putText(panel, name, (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        y += 16
    return panel


def render_side_by_side(rows, mano_left_canonical, mano_right_canonical):
    """Render aria-viz | mano-viz side-by-side with on-screen labels."""
    frames = []
    n = len(rows)
    for i, row in enumerate(rows):
        batch = row["batch"]
        aria_vis = Human.viz_transformed_batch(
            batch,
            mode="keypoints",
            viz_batch_key="actions_keypoints",
            finger_edges=ARIA_FINGER_EDGES,
            finger_edge_ranges=ARIA_FINGER_EDGE_RANGES,
        )
        left_flat = mano_left_canonical[i].reshape(-1).numpy()
        right_flat = mano_right_canonical[i].reshape(-1).numpy()
        viz_data = np.concatenate([left_flat, right_flat])  # (126,)
        image_t = batch[Human.VIZ_IMAGE_KEY][0]
        if image_t.ndim == 3 and image_t.shape[0] in (1, 3):
            image_t = image_t.permute(1, 2, 0)
        image = image_t.numpy()
        if image.dtype != np.uint8:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        mano_vis = Human.viz(
            image=image, viz_data=viz_data, mode="keypoints", intrinsics_key="base"
        )

        # Equal-height padding
        h = max(aria_vis.shape[0], mano_vis.shape[0])
        def pad(im):
            if im.shape[0] != h:
                im = cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            return im
        aria_vis = pad(aria_vis); mano_vis = pad(mano_vis)
        aria_vis = _draw_legend(aria_vis.copy())
        mano_vis = _draw_legend(mano_vis.copy())

        aria_vis = _add_title_bar(
            aria_vis,
            ["LEFT: Aria (original 21 kp/hand)", "edges: Aria layout (palm=5, tips=0-4)"],
        )
        mano_vis = _add_title_bar(
            mano_vis,
            ["RIGHT: Aria->MANO fit (per-frame optimization)", "edges: canonical MANO (wrist=0, thumb=1-4...)"],
        )

        # Vertical divider strip
        sep = np.full((aria_vis.shape[0], 4, 3), 80, dtype=aria_vis.dtype)
        combined = np.concatenate([aria_vis, sep, mano_vis], axis=1)

        # Frame counter footer
        footer = np.zeros((28, combined.shape[1], 3), dtype=combined.dtype)
        cv2.putText(
            footer, f"frame {i+1}/{n}   |   fit: 400 Adam iters, MANO PCA-45 + beta-10 + global R/t per hand   |   correspondence is hand-coded (see script header)",
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA,
        )
        combined = np.concatenate([combined, footer], axis=0)
        frames.append(combined)
    return frames


def main() -> None:
    loader = fetch_episode_loader()
    rows = gather_aria_keypoints(loader, NUM_FRAMES)

    aria_left = torch.stack([r["left_aria_kp"] for r in rows])    # (N, 21, 3)
    aria_right = torch.stack([r["right_aria_kp"] for r in rows])  # (N, 21, 3)
    print(f"Left aria valid wrist count: {torch.isfinite(aria_left[:,5]).all(-1).sum().item()}/{len(rows)}")
    print(f"Right aria valid wrist count: {torch.isfinite(aria_right[:,5]).all(-1).sum().item()}/{len(rows)}")

    print("Fitting LEFT hand...")
    mano_left = fit_mano_to_aria_batched(aria_left, is_rhand=False, n_iters=NUM_FIT_ITERS, lr=LR, beta_reg=BETA_REG, verbose=True)
    print("Fitting RIGHT hand...")
    mano_right = fit_mano_to_aria_batched(aria_right, is_rhand=True, n_iters=NUM_FIT_ITERS, lr=LR, beta_reg=BETA_REG, verbose=True)

    print("Rendering...")
    frames = render_side_by_side(rows, mano_left, mano_right)
    mpy.write_video(str(OUT_MP4), frames, fps=10)
    print(f"Wrote {OUT_MP4} ({len(frames)} frames at 10 fps -> {len(frames)/10:.1f}s)")
    # Spot-frame PNGs for static inspection if player misbehaves
    import imageio
    for idx in [0, len(frames)//3, 2*len(frames)//3, len(frames)-1]:
        png_path = SCRATCH_DIR / f"aria_to_mano_frame_{idx:03d}.png"
        imageio.imwrite(png_path, frames[idx])
    print(f"Spot PNGs in {SCRATCH_DIR}/aria_to_mano_frame_*.png")


if __name__ == "__main__":
    main()
