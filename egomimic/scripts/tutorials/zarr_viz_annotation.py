import imageio
import torch

from egomimic.pl_utils.pl_data_utils import annotation_collate
from egomimic.rldb.embodiment.human import Scale
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, S3EpisodeResolver
from egomimic.utils.aws.aws_data_utils import load_env

# Ensure mediapy can find an ffmpeg executable in this environment
load_env()

intrinsics_key = "base"

key_map = Scale.get_keymap(keymap_mode="cartesian", annotation_key="annotations")
transform_list = None

resolver = S3EpisodeResolver(
    folder_path="/storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets/",
    key_map=key_map,
    transform_list=transform_list,
)

filters = DatasetFilter(
    filter_lambdas=["lambda row: row['episode_hash'] in {'2026-03-15-23-45-35-137000'}"]
)

cloudflare_ds = MultiDataset._from_resolver(
    resolver, filters=filters, sync_from_s3=True, mode="total"
)

loader = torch.utils.data.DataLoader(
    cloudflare_ds, batch_size=1, shuffle=False, collate_fn=annotation_collate
)

ims_annotations = []
for i, batch in enumerate(loader):
    vis = Scale.viz_transformed_batch(
        batch, mode="annotations", viz_batch_key="annotations"
    )
    ims_annotations.append(vis)
    if i > 900:
        break

# save mp4

writer = imageio.get_writer("sample_annotation.mp4", fps=30)
for frame in ims_annotations:
    writer.append_data(frame)
writer.close()
