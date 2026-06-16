import imageio
import imageio_ffmpeg
import mediapy as mpy
import numpy as np
import torch

from egomimic.rldb.embodiment.human import Aria
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import (
    MultiDataset,
    S3EpisodeResolver,
)
from egomimic.utils.aws.aws_data_utils import load_env

# Ensure mediapy can find an ffmpeg executable in this environment
mpy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

TEMP_DIR = "/storage/home/hcoda1/4/paphiwetsa3/r-dxu345-0/datasets/temp_data"  # replace with your own temp directory for caching S3 data
load_env()

key_map = Aria.get_keymap(keymap_mode="cartesian")
transform_list = Aria.get_transform_list(mode="cartesian")

resolver = S3EpisodeResolver(TEMP_DIR, key_map=key_map, transform_list=transform_list)
filters = DatasetFilter(
    filter_lambdas=["lambda row: row['episode_hash'] in {'2026-03-17-01-32-38-000000'}"]
)
multi_ds = MultiDataset._from_resolver(
    resolver, filters=filters, sync_from_s3=True, mode="total"
)

loader = torch.utils.data.DataLoader(multi_ds, batch_size=1, shuffle=False)

images = []

for i, batch in enumerate(loader):
    vis = Aria.viz_transformed_batch(batch, mode="traj")
    images.append(vis)
    if i > 900:
        break

images = np.stack(images, axis=0)
writer = imageio.get_writer("sample_traj.mp4", fps=30)
for frame in images:
    writer.append_data(frame)
writer.close()
