import imageio_ffmpeg
import mediapy as mpy
import torch

from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import (
    Aria,
    _build_aria_keypoints_revert_eef_frame_transform_list,
    _build_eva_bimanual_revert_eef_frame_transform_list,
)
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, S3EpisodeResolver
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.viz_utils import save_image

# Ensure mediapy can find an ffmpeg executable in this environment
mpy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

TEMP_DIR = "/storage/project/r-dxu345-0/paphiwetsa3/datasets/temp_train"
load_env()

# Point this at a single episode directory, e.g. /path/to/episode_hash.zarr
# EPISODE_PATH = Path("/coc/flash7/scratch/egoverseDebugDatasets/1767495035712.zarr")

key_map = Eva.get_keymap()
transform_list = Eva.get_transform_list(mode="cartesian_wristframe_ypr")

# Build a MultiDataset with exactly one ZarrDataset inside
# single_ds = ZarrDataset(Episode_path=EPISODE_PATH, key_map=key_map, transform_list=transform_list)
# single_ds = ZarrDataset(Episode_path=EPISODE_PATH, key_map=key_map)

# multi_ds = MultiDataset(datasets={"single_episode": single_ds}, mode="total")
resolver = S3EpisodeResolver(TEMP_DIR, key_map=key_map, transform_list=transform_list)
filters = {"episode_hash": "2025-12-26-18-07-46-296000"}
multi_ds = MultiDataset._from_resolver(
    resolver, filters=filters, sync_from_s3=True, mode="total"
)

loader = torch.utils.data.DataLoader(multi_ds, batch_size=1, shuffle=False)


for batch in loader:
    vis_ypr = Eva.viz_transformed_batch(
        batch,
        mode="axes",
        transform_list=_build_eva_bimanual_revert_eef_frame_transform_list(
            is_quat=False
        ),
    )
    save_image(vis_ypr, "vis_ypr.png")
    break

temp_dir = "/storage/project/r-dxu345-0/paphiwetsa3/datasets/temp_train"

intrinsics_key = "base"

key_map = Aria.get_keymap(mode="keypoints")
transform_list = Aria.get_transform_list(mode="keypoints_wristframe_ypr")

resolver = S3EpisodeResolver(
    temp_dir,
    key_map=key_map,
    transform_list=transform_list,
)

filters = {"episode_hash": "2026-01-20-20-59-43-376000"}  # aria
# filters = {"episode_hash": "692ee048ef7557106e6c4b8d"} # mecka

cloudflare_ds = MultiDataset._from_resolver(
    resolver, filters=filters, sync_from_s3=True, mode="total"
)

loader = torch.utils.data.DataLoader(cloudflare_ds, batch_size=1, shuffle=False)

ims_keypoints = []
for i, batch in enumerate(loader):
    vis_keypoints = Aria.viz_transformed_batch(
        batch,
        mode="keypoints",
        color="Reds",
        action_key="actions_keypoints",
        transform_list=_build_aria_keypoints_revert_eef_frame_transform_list(
            is_quat=False
        ),
    )
    ims_keypoints.append(vis_keypoints)
    if i > 1:
        save_image(vis_keypoints, "keypoints.png")
        break
