#!/usr/bin/env python3
"""Generate depth maps from Aria Gen2 stereo streams using Fast-FoundationStereo."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from projectaria_tools.core import data_provider
from projectaria_tools.core.image import InterpolationMethod

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import aria2_utils as a2u


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth from stereo for Aria Gen2 VRS (Fast-FoundationStereo).")
    parser.add_argument("--vrs", default="./data/video.vrs")
    parser.add_argument("--fs-home", default="/nethome/yzhu827/flash/Fast-FoundationStereo")
    parser.add_argument(
        "--ckpt",
        default="/nethome/yzhu827/flash/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth",
    )
    parser.add_argument("--output-dir", default="./outputs/depth_npy")
    parser.add_argument("--frame-step", type=int, default=3)
    parser.add_argument("--output-width", type=int, default=512)
    parser.add_argument("--output-height", type=int, default=512)
    parser.add_argument("--focal-scale", type=float, default=1.25)
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--min-disparity", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument(
        "--max-stereo-time-diff-ms",
        type=float,
        default=5.0,
        help="Maximum allowed timestamp difference when pairing left/right stereo frames",
    )
    return parser.parse_args()


def nearest_index(target_ts: int, sorted_ts: np.ndarray) -> tuple[int, int]:
    pos = int(np.searchsorted(sorted_ts, target_ts))
    if pos <= 0:
        return 0, int(sorted_ts[0])
    if pos >= len(sorted_ts):
        return len(sorted_ts) - 1, int(sorted_ts[-1])
    before = pos - 1
    after = pos
    if abs(int(sorted_ts[after]) - target_ts) < abs(int(sorted_ts[before]) - target_ts):
        return after, int(sorted_ts[after])
    return before, int(sorted_ts[before])


def load_fast_foundation_stereo(fs_home: str, ckpt_path: str, valid_iters: int, max_disp: int):
    fs_home_path = Path(fs_home)
    if not fs_home_path.exists():
        raise FileNotFoundError(f"Fast-FoundationStereo path not found: {fs_home}")
    if str(fs_home_path) not in sys.path:
        sys.path.insert(0, str(fs_home_path))
    from core.utils.utils import InputPadder

    with open(Path(ckpt_path).parent / "cfg.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["valid_iters"] = valid_iters
    cfg["max_disp"] = max_disp

    model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    return model.cuda().eval(), InputPadder, OmegaConf.create(cfg)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    provider = data_provider.create_vrs_data_provider(str(Path(args.vrs)))
    device_calib = provider.get_device_calibration()
    left_sid = provider.get_stream_id_from_label("slam-front-left")
    right_sid = provider.get_stream_id_from_label("slam-front-right")
    left_calib = device_calib.get_camera_calib("slam-front-left")
    right_calib = device_calib.get_camera_calib("slam-front-right")
    T_left = left_calib.get_transform_device_camera().inverse()
    T_right = right_calib.get_transform_device_camera().inverse()
    shared_linear = a2u.fisheye_to_linear_calib(
        left_calib, args.focal_scale, args.output_width, args.output_height
    )
    Rl_n, Rr_n = a2u.create_scanline_rectified_cameras(T_left, T_right)
    baseline = a2u.compute_stereo_baseline(T_left, T_right)
    focal = float(shared_linear.get_projection_params()[0])
    model, InputPadder, cfg = load_fast_foundation_stereo(
        args.fs_home, args.ckpt, args.valid_iters, args.max_disp
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    right_count = provider.get_num_data(right_sid)
    right_timestamps = np.zeros(right_count, dtype=np.int64)
    for ridx in range(right_count):
        _, right_rec = provider.get_image_data_by_index(right_sid, ridx)
        right_timestamps[ridx] = int(right_rec.capture_timestamp_ns)
    max_stereo_dt_ns = int(args.max_stereo_time_diff_ms * 1e6)

    t0 = time.time()
    for idx in range(
        0, provider.get_num_data(left_sid), args.frame_step
    ):
        left_data, left_record = provider.get_image_data_by_index(left_sid, idx)
        left_ts = int(left_record.capture_timestamp_ns)
        right_idx, right_ts = nearest_index(left_ts, right_timestamps)
        if abs(right_ts - left_ts) > max_stereo_dt_ns:
            continue
        right_data, _ = provider.get_image_data_by_index(right_sid, int(right_idx))
        left_rect, right_rect = a2u.rectify_stereo_pair(
            left_data.to_numpy_array(),
            right_data.to_numpy_array(),
            left_calib,
            right_calib,
            shared_linear,
            shared_linear,
            Rl_n,
            Rr_n,
            interpolation=InterpolationMethod.BILINEAR,
        )
        left_t = (
            torch.from_numpy(a2u.ensure_rgb3(left_rect))
            .float()
            .cuda()
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        right_t = (
            torch.from_numpy(a2u.ensure_rgb3(right_rect))
            .float()
            .cuda()
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
        left_t, right_t = padder.pad(left_t, right_t)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            with torch.no_grad():
                disparity = model.forward(
                    left_t, right_t, iters=cfg.valid_iters, test_mode=True,
                    optimize_build_volume="pytorch1",
                )
        disparity = padder.unpad(disparity.float()).cpu().numpy().squeeze()
        depth = a2u.disparity_to_depth(
            disparity, baseline, focal, args.min_disparity, args.max_depth
        )
        np.save(out_dir / f"{int(left_record.capture_timestamp_ns)}.npy", depth)
        if idx % max(1, 50 * args.frame_step) == 0:
            print(f"Frame {idx:06d} ({(time.time()-t0)*1000.0:.1f} ms)")
            t0 = time.time()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
