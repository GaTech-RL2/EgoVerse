"""Compute exact projected pixel positions for the GT arc-tok trajectory and
check color at each in val-video mp4 frame 0."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torchvision.io as tvio

from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset
from egomimic.visualization.arc_tok_viz import detokenize_arc_actions

RUN = "/storage/project/r-dxu345-0/acheluva3/Egoverse-arc/logs/arc_tests_cotrain_arctok/h100_1gpu_bs128_constlr3e4_val30_2026-07-27_22-28-30"
VID = f"{RUN}/0/videos/epoch_179/HUMAN_BIMANUAL/validation_video_0.mp4"
NS = f"{RUN}/0/norm_stats/norm_stats.json"

r = LocalEpisodeResolver(
    Path("/storage/project/r-dxu345-0/shared/arc_tests"),
    key_map=Human.get_keymap(keymap_mode="arc_tokenizer_cartesian"),
    transform_list=Human.get_transform_list(
        mode="arc_tokenizer_cartesian",
        stride=3,
        min_distance_unit=0.20,
        resampled_vector_length=15,
    ),
)
ds = MultiDataset._from_resolver(
    r,
    filters=DatasetFilter(
        filter_lambdas=[
            "lambda row: row.get('embodiment') == 'human_bimanual' and row.get('task_name') == 'debug'"
        ]
    ),
    mode="valid",
    valid_ratio=0.2,
    bounds_check=False,
)
ds.populate_from_datasets()
payload = json.load(open(NS))
for emb_str, kd in payload["stats"].items():
    emb = int(emb_str)
    ds.norm_stats.setdefault(emb, {})
    for k, stats in kd.items():
        ds.norm_stats[emb][k] = {n: np.asarray(v) for n, v in stats.items()}
ds.norm_mode = "quantile"

sample = ds[0]
un = ds.unnormalize(dict(sample), sample["embodiment"])
arc = un["actions_cartesian"].cpu().numpy()
det = detokenize_arc_actions(
    arc, min_distance_unit=0.20, resampled_vector_length=15, action_horizon=100
)
K = un["intrinsics"].cpu().numpy()


def project(xyz, K):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    zsafe = np.where(np.abs(z) > 1e-8, z, 1e-8)
    u = K[0, 0] * x / zsafe + K[0, 2]
    v = K[1, 1] * y / zsafe + K[1, 2]
    return np.stack([u, v], axis=-1)


L_uv = project(det[:, 0:3], K)
R_uv = project(det[:, 7:10], K)
print(f"L_xyz first 3 rows:\n{det[:3, 0:3].round(4)}")
print(f"L pixel positions (first 5): {L_uv[:5].round(1)}")
print(f"R pixel positions (first 5): {R_uv[:5].round(1)}")
print(
    f"L pixel range: u={L_uv[:,0].min():.1f}..{L_uv[:,0].max():.1f}  v={L_uv[:,1].min():.1f}..{L_uv[:,1].max():.1f}"
)
print(
    f"R pixel range: u={R_uv[:,0].min():.1f}..{R_uv[:,0].max():.1f}  v={R_uv[:,1].min():.1f}..{R_uv[:,1].max():.1f}"
)

# Now sample the val-video at those pixel positions
vid = tvio.read_video(VID, pts_unit="sec")[0][0].cpu().numpy()
H, W = vid.shape[:2]


def sample_at(uv, im):
    u = np.clip(uv[:, 0].astype(int), 0, W - 1)
    v = np.clip(uv[:, 1].astype(int), 0, H - 1)
    return im[v, u]


L_colors = sample_at(L_uv, vid)
R_colors = sample_at(R_uv, vid)
print(
    f"\nColor sampled at L GT pixel positions in val-video (first 5):\n{L_colors[:5]}"
)
print(f"Color sampled at R GT pixel positions in val-video (first 5):\n{R_colors[:5]}")
print(f"Mean sampled color at L GT pixels: RGB = {L_colors.mean(0).round(1)}")
print(f"Mean sampled color at R GT pixels: RGB = {R_colors.mean(0).round(1)}")

# Draw crosshairs at expected L and R positions on val-video for visual confirm
vid_draw = vid.copy()
for uv in np.concatenate([L_uv, R_uv]):
    u, v = int(uv[0]), int(uv[1])
    if 0 <= u < W and 0 <= v < H:
        cv2.drawMarker(vid_draw, (u, v), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 8, 1)
cv2.imwrite(
    "/storage/project/r-dxu345-0/acheluva3/Egoverse-arc/out/valvid_with_expected_gt.png",
    vid_draw[:, :, ::-1],
)
print(
    "\nSaved: /storage/project/r-dxu345-0/acheluva3/Egoverse-arc/out/valvid_with_expected_gt.png"
)
print("Yellow X marks are where the GT should render if pipelines agree.")
print("If X marks land on the wrist in the val-video frame, everything is correct.")
