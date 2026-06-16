"""Fetch first mecka episode and render full-episode MANO keypoint overlay mp4."""

from pathlib import Path

import imageio_ffmpeg
import mediapy as mpy
import torch

from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, S3EpisodeResolver
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

mpy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

REPO_ROOT = Path(__file__).resolve().parents[2].parent
SCRATCH_DIR = REPO_ROOT / "scratch"
CACHE_DIR = SCRATCH_DIR / "mecka_viz_cache"
OUT_MP4 = SCRATCH_DIR / "mecka_keypoints.mp4"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    load_env()

    engine = create_default_engine()
    df = episode_table_to_df(engine)
    mecka_df = df[df["embodiment"].str.startswith("mecka", na=False)]
    if len(mecka_df) == 0:
        raise RuntimeError("No mecka episodes found in the episode table")
    episode_hash = mecka_df.iloc[0]["episode_hash"]
    print(f"Using mecka episode: {episode_hash} (of {len(mecka_df)} candidates)")

    key_map = Human.get_keymap(keymap_mode="keypoints")
    transform_list = Human.get_transform_list(mode="keypoints_headframe_ypr", stride=1)

    resolver = S3EpisodeResolver(
        str(CACHE_DIR), key_map=key_map, transform_list=transform_list
    )
    filters = DatasetFilter(
        filter_lambdas=[f"lambda row: row['episode_hash'] == {episode_hash!r}"]
    )
    dataset = MultiDataset._from_resolver(
        resolver, filters=filters, sync_from_s3=True, mode="total"
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    frames = []
    for i, batch in enumerate(loader):
        vis = Human.viz_transformed_batch(
            batch, mode="keypoints", viz_batch_key="actions_keypoints"
        )
        frames.append(vis)
        if i % 50 == 0:
            print(f"  frame {i}")
    print(f"Rendered {len(frames)} frames")

    mpy.write_video(str(OUT_MP4), frames, fps=30)
    print(f"Wrote {OUT_MP4}")


if __name__ == "__main__":
    main()
