#!/usr/bin/env python3
import argparse, re
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torchvision.io import read_video, write_video
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def list_mp4s(dir_path: str) -> List[str]:
    files = sorted(str(x) for x in Path(dir_path).glob("*.mp4"))
    if not files:
        raise FileNotFoundError(f"No .mp4 files found in: {dir_path}")
    return files

def label_from_dir(dir_path: str) -> str:
    p = Path(dir_path)
    s = str(p)
    if "resnet" in s:
        return "resnet"
    # try to extract dino_v3/{model_type}
    # e.g., .../dino_v3_fold_clothes/vitbase16_frozen_2025-... -> dino_v3/vitbase16_frozen
    m = re.search(r"/dino_v3[^/]*/([^/]+)", s)
    if m:
        model = m.group(1)
        model = re.split(r"(_20\d{2}-\d{2}-\d{2}|\bepoch_\d+)", model)[0] or model
        return f"dino_v3/{model}"
    return p.name

def load_speedup_single(path: str, target_h: int, target_w: int, device: str, factor: int) -> Tuple[torch.Tensor, float]:
    vid, _, info = read_video(path, pts_unit="sec")  # [T,H,W,C] uint8
    if vid.numel() == 0:
        raise RuntimeError(f"Failed to read frames from {path}")
    fps = float(info.get("video_fps", 30.0))
    # 4×/N× speed via decimation
    vid = vid[::factor]
    if vid.shape[0] < 1:
        return vid[:1], fps
    H, W = vid.shape[1], vid.shape[2]
    if (H, W) != (target_h, target_w):
        x = vid.permute(0, 3, 1, 2).float().to(device)
        x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        vid = x.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu()
    return vid, fps

def load_concat_speedup(dir_path: str, target_h: int, target_w: int, device: str, factor: int) -> Tuple[torch.Tensor, int]:
    seqs, fps_vals = [], []
    for f in list_mp4s(dir_path):
        v, fps = load_speedup_single(f, target_h, target_w, device, factor)
        seqs.append(v)
        fps_vals.append(fps)
    video = torch.cat(seqs, dim=0)
    fps_out = int(round(min(fps_vals))) if fps_vals else 30
    return video, fps_out

def overlay_label(video: torch.Tensor, label: str, box_h: int = 28, pad: int = 8) -> torch.Tensor:
    T, H, W, C = video.shape
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    out = []
    for i in range(T):
        img = Image.fromarray(video[i].numpy())
        draw = ImageDraw.Draw(img, mode="RGBA")
        draw.rectangle([0, 0, W, box_h], fill=(0, 0, 0, 160))
        draw.text((pad, 4), label, fill=(255, 255, 255, 255), font=font)
        out.append(torch.from_numpy(np.array(img, dtype="uint8")))
    return torch.stack(out, dim=0)

def stack_2x2(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    tmin = min(a.shape[0], b.shape[0], c.shape[0], d.shape[0])
    a, b, c, d = a[:tmin], b[:tmin], c[:tmin], d[:tmin]
    top = torch.cat([a, b], dim=2)   # width
    bot = torch.cat([c, d], dim=2)
    return torch.cat([top, bot], dim=1)  # height

def main():
    ap = argparse.ArgumentParser(description="2x2 collage with torchvision; concat all mp4s per dir; N× speed via decimation.")
    ap.add_argument("--dirs", nargs=4, metavar=("D0","D1","D2","D3"), required=False, default=[
        "/nethome/rpunamiya6/flash/Projects/EgoVerse/logs/resnet_fold_clothes/18_finetune_2025-08-25_16-33-22/0/videos/epoch_1399/ARIA_BIMANUAL",
        "/nethome/rpunamiya6/flash/Projects/EgoVerse/logs/dino_v3_fold_clothes/vits16plus_frozen_2025-08-25_16-25-14/0/videos/epoch_1399/ARIA_BIMANUAL",
        "/nethome/rpunamiya6/flash/Projects/EgoVerse/logs/dino_v3_fold_clothes/vitbase16_frozen_2025-08-25_16-38-52/0/videos/epoch_1199/ARIA_BIMANUAL",
        "/nethome/rpunamiya6/flash/Projects/EgoVerse/logs/dino_v3_fold_clothes/convext_base_frozen_2025-08-25_17-07-37/0/videos/epoch_1199/ARIA_BIMANUAL",
    ])
    ap.add_argument("--out", type=str, default="collage_2x2_4x.mp4")
    ap.add_argument("--label", action="store_true", help="overlay labels on each tile")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--tile_h", type=int, default=480)
    ap.add_argument("--tile_w", type=int, default=640)
    ap.add_argument("--speed", type=int, default=4, help="speedup factor via frame decimation (e.g., 4 = 4x)")
    ap.add_argument("--crf", type=int, default=16, help="x264 quality (lower=better)")
    ap.add_argument("--bitrate", type=str, default="8M", help="target bitrate, e.g. 8M")
    ap.add_argument("--preset", type=str, default="slow", help="x264 preset")
    args = ap.parse_args()

    labels = [label_from_dir(d) for d in args.dirs]
    print("Dirs & labels:")
    for d,l in zip(args.dirs, labels):
        print(f"  {d} -> {l}")

    vids, fps_list = [], []
    for d in args.dirs:
        v, fps = load_concat_speedup(d, args.tile_h, args.tile_w, args.device, args.speed)
        vids.append(v)      # [T,480,640,3] uint8
        fps_list.append(fps)

    if args.label:
        vids = [overlay_label(v, lab) for v, lab in zip(vids, labels)]

    grid = stack_2x2(vids[0], vids[1], vids[2], vids[3])  # [T,960,1280,3]
    T, H2, W2, _ = grid.shape
    fps_out = min(fps_list) if fps_list else 30
    print(f"Output frames: {T} @ {H2}x{W2} | fps_out={fps_out}")

    # Higher quality encode: combine CRF + bitrate cap to avoid mushy blocks
    # Torchvision passes these as '-crf', '-preset', '-b:v', etc.
    write_video(
        args.out,
        grid,
        fps_out,
        video_codec="h264",
        options={"crf": str(args.crf), "preset": args.preset, "b:v": args.bitrate, "profile:v": "high"},
        audio_array=None, audio_fps=None, audio_codec=None,
    )
    print(f"✅ Saved {args.out}")

if __name__ == "__main__":
    main()
