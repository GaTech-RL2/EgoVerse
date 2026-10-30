"""Compare a zarr episode with its zarr_to_lance copy: steps rows, decoded video, loader samples.

Steps: every mapped zarr key equals its Lance column (as float32) on valid frames, and the
timestamps match (zarr's obs_rgb_timestamps_ns, else the hash's start time + i / fps). Video: PSNR/SSIM
of decoded MP4 frames against the zarr JPEGs (the 360p stream against the JPEG resized as
the converter does). Samples: LocalEpisodeResolver on a folder holding just the episode
against LanceEpisodeResolver on the root, at the same indices, with the keymap and
transforms of the training config for the zarr's format (CONFIGS); Lance images at a lower
resolution are compared against the zarr image resized to match.
"""

import argparse
import json
from pathlib import Path

import av
import cv2
import lance
import numpy as np
import simplejpeg
import torch
import zarr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.lance.dataset import LanceEpisodeResolver, zarr_to_lance
from egomimic.rldb.lance.schema import DEXMATE_STEPS, only_steps_path, zarr_columns
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver
from egomimic.scripts.zarr_to_lance import (
    FORMATS,
    detect_format,
    episode_hash_of,
    hash_start_ns,
)


def _mecka():
    # data/mecka_fold_flagship_opsplit_hpt_6d.yaml without its annotation key.
    km = Human.get_keymap(keymap_mode="cartesian", proprio_history=3, history_stride=5)
    tl = Human.get_transform_list(
        mode="cartesian_wristframe_6d",
        stride=1,
        fix_left_wrist_convention=True,
        pad_proprio_gripper=True,
    )
    return km, tl


def _yam():
    # data/eva_pilot_lance.yaml: YAM has no training config yet and stores EVA's arrays.
    return Eva.get_keymap(keymap_mode="cartesian"), Eva.get_transform_list(
        mode="cartesian_wristframe_6d"
    )


def _dexmate():
    # No Dexmate embodiment class yet: every numeric column, untransformed.
    km = {
        k: {"key_type": "proprio_keys", "zarr_key": k}
        for k in zarr_columns(DEXMATE_STEPS)
        if k != "sharpa.subtask_index"
    }
    km["images.front_1"] = {"key_type": "camera_keys", "zarr_key": "images.front_1"}
    return km, None


CONFIGS = {
    "mecka": _mecka,
    "scale": _mecka,
    "microagi_v2": _mecka,
    "yam": _yam,
    "dexmate": _dexmate,
}


def _column(tbl, dotted: str) -> np.ndarray:
    col = tbl.column(dotted.split(".")[0])
    for f in dotted.split(".")[1:]:
        col = col.combine_chunks().field(f)
    return np.asarray(col.to_pylist(), dtype=np.float32)


def check_steps(z, root: Path, h: str, rec: dict) -> dict:
    n = int(rec["num_frames"])
    steps = lance.dataset(str(root / only_steps_path(_one(rec)))).to_table(
        filter=f"recording_id = '{rec['recording_id']}'"
    )
    out = {"rows": steps.num_rows, "zarr_frames": n}
    ts = steps.column("timestamp_ns").to_numpy()
    if "obs_rgb_timestamps_ns" in z:
        ref = np.asarray(z["obs_rgb_timestamps_ns"][:n])
    else:
        start = hash_start_ns(h) or 0
        ref = start + np.round(np.arange(n) * 1e9 / rec["fps"]).astype(np.int64)
        out["timestamps_from"] = "hash start + i / fps"
    out["timestamps_equal"] = bool(np.array_equal(ts, ref))
    valid = {
        s: steps.column(f"{s}_valid").to_numpy(zero_copy_only=False)
        for s in ("left", "right")
    }
    if "head_pose_valid" in steps.column_names:
        valid["head"] = steps.column("head_pose_valid").to_numpy(zero_copy_only=False)
    out["valid_frac"] = {k: float(v.mean()) for k, v in valid.items()}
    diffs = {}
    for zk, lk in zarr_to_lance(rec["end_effector"]).items():
        if zk not in z:
            continue
        a = np.asarray(z[zk][:n], dtype=np.float32).reshape(n, -1)
        b = _column(steps, lk).reshape(n, -1)
        side = "head" if zk == "obs_head_pose" else zk.split(".")[0]
        v = valid.get(side, np.ones(n, bool))
        diffs[zk] = float(np.abs(a[v] - b[v]).max()) if v.any() else None
    out["max_abs_diff_on_valid"] = diffs
    return out


def _one(rec: dict):
    return {k: [rec[k]] for k in ("lab", "embodiment", "end_effector")}


def video_keys(fmt: str, videos, keys: set[str]) -> dict[str, str]:
    """{video name: zarr JPEG key}: images.<cam> for <cam> and <cam>_<short side>."""
    extra = {v: k for k, v in FORMATS[fmt].get("extra_videos", {}).items()}
    out = {}
    for v in videos:
        key = extra.get(v, f"images.{v}")
        if key not in keys:
            key = f"images.{v.rsplit('_', 1)[0]}"
        out[v] = key
    return out


def check_video(z, root: Path, rec: dict, every: int, fmt: str) -> dict:
    n = int(rec["num_frames"])
    out = {}
    paths = dict(rec["video_path"])
    for cam, key in video_keys(fmt, paths, set(z.keys())).items():
        jpegs, path = z[key], paths[cam]
        psnr, ssim, count = [], [], 0
        with av.open(str(root / path)) as c:
            for i, fr in enumerate(c.decode(video=0)):
                count += 1
                if i % every:
                    continue
                got = fr.to_ndarray(format="rgb24")
                ref = simplejpeg.decode_jpeg(jpegs[i : i + 1][0], colorspace="RGB")
                if ref.shape != got.shape:
                    ref = cv2.resize(
                        ref, got.shape[1::-1], interpolation=cv2.INTER_AREA
                    )
                psnr.append(peak_signal_noise_ratio(ref, got, data_range=255))
                ssim.append(
                    structural_similarity(ref, got, channel_axis=2, data_range=255)
                )
        out[cam] = {
            "frames": count,
            "frames_match": count == n,
            "compared": len(psnr),
            "psnr_mean": float(np.mean(psnr)),
            "psnr_min": float(np.min(psnr)),
            "ssim_mean": float(np.mean(ssim)),
            "ssim_min": float(np.min(ssim)),
        }
    return out


def check_samples(
    zarr_dir: Path, root: Path, h: str, rec: dict, fmt: str, k: int
) -> dict:
    key_map, tl = CONFIGS[fmt]()
    videos = set(dict(rec["video_path"]))
    cameras = {}
    for spec in key_map.values():
        if spec["key_type"] != "camera_keys":
            continue
        zk = spec["zarr_key"]
        cam = zk[len("images.") :]
        full = [
            v for v in videos if v.startswith(f"{cam}_") and v[len(cam) + 1 :].isdigit()
        ]
        cameras[zk] = full[0] if full else cam
    Z = LocalEpisodeResolver(zarr_dir, key_map=key_map, transform_list=tl).resolve()[h]
    L = LanceEpisodeResolver(
        root,
        key_map=key_map,
        transform_list=tl,
        cameras=cameras,
        require_hands_in_frame="none",
    ).resolve(filters=DatasetFilter(episode_hashes=[h]))[rec["recording_id"]]
    out = {"zarr_len": len(Z), "lance_len": len(L)}
    out["anchors_identity"] = len(L) == len(Z) and all(
        L.anchor(j) == j for j in range(len(L))
    )
    m = min(len(Z), len(L))
    idx = np.random.default_rng(0).choice(m, min(k, m), replace=False)
    idx = sorted({0, m - 1, *idx.tolist()})
    worst, img, only = {}, [], set()
    for i in idx:
        a, b = Z[i], L[i]
        only |= set(a) ^ set(b)
        for key, v in a.items():
            if key not in b or not isinstance(v, torch.Tensor):
                if key in b and v != b[key]:
                    worst[key] = "differs"
                continue
            w = b[key]
            if "images" in key and v.shape != w.shape:
                # Lance holds the 360p stream only: resize the zarr image to it.
                chw = v.shape[0] == 3
                a_hwc = (v.permute(1, 2, 0) if chw else v).float().numpy()
                hw = w.shape[1:] if chw else w.shape[:2]
                a_hwc = cv2.resize(a_hwc, hw[::-1], interpolation=cv2.INTER_AREA)
                v = torch.from_numpy(a_hwc)
                v = v.permute(2, 0, 1) if chw else v
            if v.shape != w.shape:
                worst[key] = f"shape {tuple(v.shape)} vs {tuple(w.shape)}"
                continue
            d = (v.float() - w.float()).abs()
            if "images" in key:
                img.append(float(d.mean()))
            worst[key] = max(worst.get(key, 0.0), float(d.max()))
    out.update(
        indices=len(idx),
        max_abs_diff=worst,
        image_mean_abs_diff=float(np.mean(img)) if img else None,
        keys_in_only_one=sorted(only),
    )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--zarr-dir", type=Path, required=True, help="holds just the episode"
    )
    ap.add_argument("--lance-root", type=Path, required=True)
    ap.add_argument("--every", type=int, default=5, help="compare every Nth frame")
    ap.add_argument("--samples", type=int, default=200)
    args = ap.parse_args()
    (name,) = [p.name for p in args.zarr_dir.iterdir()]
    h = episode_hash_of(name)
    z = zarr.open_group(str(args.zarr_dir / name), mode="r")
    fmt = detect_format(dict(z.attrs), set(z.keys()))
    (alias,) = (
        lance.dataset(str(args.lance_root / "aliases.lance"))
        .to_table(filter=f"episode_hash = '{h}'")
        .to_pylist()
    )
    rid = alias["recording_id"]
    (rec,) = (
        lance.dataset(str(args.lance_root / "recordings.lance"))
        .to_table(filter=f"recording_id = '{rid}'")
        .to_pylist()
    )
    report = {
        "episode_hash": h,
        "format": fmt,
        "recording_id": rid,
        "steps": check_steps(z, args.lance_root, h, rec),
        "video": check_video(z, args.lance_root, rec, args.every, fmt),
        "samples": check_samples(
            args.zarr_dir, args.lance_root, h, rec, fmt, args.samples
        ),
    }
    print(json.dumps(report, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
