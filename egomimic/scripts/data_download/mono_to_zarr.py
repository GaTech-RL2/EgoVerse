"""Turn a mono raw episode into the zarr store the mecka 6D recipe reads.

The download saved ``video.mp4``, ``hand_pose.json``, ``body_pose.json``,
``pelvis_pose.csv``, ``intrinsics.json`` and ``meta.json``. Training reads a
zarr group with world-frame ``left/right.obs_keypoints`` (21x3), matching
``obs_ee_pose`` and ``obs_wrist_pose`` (xyz + wxyz), ``obs_head_pose``,
``images.front_1`` and one annotation span of the episode's task text.

Hand landmarks in this export sit in the scene frame (a wrist is near the
table, not at a camera depth). Wrist and palm poses are built from those
points with the same relabel ``mecka_to_zarr.compute_hand_pose_xyzquat`` uses,
so the left wrist is the corrected convention and
``fix_left_wrist_convention`` stays off. ``obs_head_pose`` is the per-frame
``camera_transform`` with the same local-axis remap as
``MeckaExtractor._extract_head_poses``. Images are resized to 640x360 and the
pinhole intrinsics are scaled with them.

    python egomimic/scripts/data_download/mono_to_zarr.py convert \\
        --root /workspace/datasets/fixed_budget_opscale \\
        --manifest egomimic/hydra_configs/experiments/fixed_budget_opscale/manifest.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _stub_cv2() -> None:
    """The gateway image has no libGL. Nothing in this conversion calls cv2;
    the stub only lets the zarr writer import."""
    if "cv2" in sys.modules:
        return
    try:
        import cv2  # noqa: F401
    except Exception:
        stub = types.ModuleType("cv2")
        stub.__spec__ = ModuleSpec("cv2", loader=None)
        sys.modules["cv2"] = stub


_stub_cv2()

from egomimic.rldb.zarr.zarr_writer import ZarrWriter  # noqa: E402
from egomimic.scripts.mecka_process.mecka_to_zarr import (  # noqa: E402
    compute_hand_pose_xyzquat,
)

OUT_W = 640
OUT_H = 360
# (x, y, z) -> (-y, -z, x), right-multiplied onto the camera pose.
_HEAD_R_FIX = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
_SPLITS = (
    "op1",
    "op4",
    "op8",
    "op12",
    "op16",
    "seen_val",
    "unseen_val",
    "flagship_unseen_val",
)


def _quat_xyzw(ori: dict) -> np.ndarray:
    return np.array([ori["x"], ori["y"], ori["z"], ori["w"]], dtype=np.float64)


def head_pose_from_camera(position: dict, orientation: dict) -> np.ndarray:
    """One ``obs_head_pose`` row, xyz + wxyz, after the mecka axis remap."""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_quat(_quat_xyzw(orientation)).as_matrix()
    pose[:3, 3] = [position["x"], position["y"], position["z"]]
    fix_inv = np.eye(4, dtype=np.float64)
    fix_inv[:3, :3] = _HEAD_R_FIX.T
    adjusted = pose @ fix_inv
    quat_xyzw = Rotation.from_matrix(adjusted[:3, :3]).as_quat()
    return np.array(
        [
            adjusted[0, 3],
            adjusted[1, 3],
            adjusted[2, 3],
            quat_xyzw[3],
            quat_xyzw[0],
            quat_xyzw[1],
            quat_xyzw[2],
        ],
        dtype=np.float32,
    )


def _landmarks(hand: dict) -> np.ndarray | None:
    points = hand.get("landmarks") or []
    if len(points) != 21:
        return None
    ordered = sorted(points, key=lambda item: int(item["landmark_index"]))
    if [int(item["landmark_index"]) for item in ordered] != list(range(21)):
        return None
    return np.array(
        [[item["x"], item["y"], item["z"]] for item in ordered], dtype=np.float64
    )


def tracks_from_hand_pose(hand_pose: dict) -> dict[str, np.ndarray]:
    """World-frame keypoint, palm and wrist tracks, plus the head pose.

    A missing hand on a frame is zeros, which ``compute_hand_pose_xyzquat``
    turns into a zero position and an identity quaternion.
    """
    frames = hand_pose["frames"]
    total = int(hand_pose["total_frames"])
    left_kp = np.zeros((total, 21, 3), dtype=np.float64)
    right_kp = np.zeros((total, 21, 3), dtype=np.float64)
    left_ee = np.zeros((total, 7), dtype=np.float32)
    right_ee = np.zeros((total, 7), dtype=np.float32)
    left_wrist = np.zeros((total, 7), dtype=np.float32)
    right_wrist = np.zeros((total, 7), dtype=np.float32)
    head = np.zeros((total, 7), dtype=np.float32)
    identity = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
    left_ee[:] = identity
    right_ee[:] = identity
    left_wrist[:] = identity
    right_wrist[:] = identity
    for index in range(total):
        frame = frames.get(str(index))
        if not frame:
            continue
        camera = frame.get("camera_transform") or {}
        if camera.get("position") and camera.get("orientation"):
            head[index] = head_pose_from_camera(
                camera["position"], camera["orientation"]
            )
        else:
            head[index] = identity
        for hand in frame.get("hands") or []:
            side = str(hand.get("hand_type", "")).upper()
            if side not in ("LEFT", "RIGHT"):
                continue
            points = _landmarks(hand)
            if points is None:
                continue
            palm, wrist = compute_hand_pose_xyzquat(points, 0 if side == "LEFT" else 1)
            if side == "LEFT":
                left_kp[index] = points
                left_ee[index] = palm
                left_wrist[index] = wrist
            else:
                right_kp[index] = points
                right_ee[index] = palm
                right_wrist[index] = wrist
    return {
        "left.obs_keypoints": left_kp.reshape(total, 63).astype(np.float32),
        "right.obs_keypoints": right_kp.reshape(total, 63).astype(np.float32),
        "left.obs_ee_pose": left_ee,
        "right.obs_ee_pose": right_ee,
        "left.obs_wrist_pose": left_wrist,
        "right.obs_wrist_pose": right_wrist,
        "obs_head_pose": head,
    }


def assert_wrist_is_near_the_camera(tracks: dict[str, np.ndarray]) -> None:
    """Catch a frame mixup before it is written 2,711 times.

    A present wrist should sit a hand's reach from the camera. Table-frame
    landmarks against this camera pose land around half a metre; camera-frame
    landmarks treated as world do not.
    """
    head = tracks["obs_head_pose"][:, :3]
    distances = []
    for key in ("left.obs_keypoints", "right.obs_keypoints"):
        wrist = tracks[key][:, :3]
        present = np.linalg.norm(wrist, axis=1) > 1e-6
        if not np.any(present):
            continue
        distances.append(np.linalg.norm(wrist[present] - head[present], axis=1))
    if not distances:
        raise RuntimeError("episode has no hand landmarks")
    median = float(np.median(np.concatenate(distances)))
    if not 0.05 <= median <= 3.0:
        raise RuntimeError(
            f"median camera-to-wrist distance is {median:.3f} m; "
            "the landmark frame does not match camera_transform"
        )


def intrinsics_matrix(intrinsics: dict, width: int, height: int) -> np.ndarray:
    """3x4 pinhole K at ``width`` x ``height``, scaled from the json resolution."""
    source_w = float(intrinsics.get("w") or width)
    source_h = float(intrinsics.get("h") or height)
    matrix = np.zeros((3, 4), dtype=np.float64)
    matrix[0, 0] = float(intrinsics["fl_x"]) * width / source_w
    matrix[1, 1] = float(intrinsics["fl_y"]) * height / source_h
    matrix[0, 2] = float(intrinsics["cx"]) * width / source_w
    matrix[1, 2] = float(intrinsics["cy"]) * height / source_h
    matrix[2, 2] = 1.0
    return matrix


def _decode_video(path: Path, limit: int) -> np.ndarray:
    import av
    from PIL import Image

    container = av.open(str(path))
    frames = []
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            image = image.resize((OUT_W, OUT_H), Image.Resampling.BOX)
            frames.append(np.asarray(image, dtype=np.uint8))
            if len(frames) >= limit:
                break
    finally:
        container.close()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames, axis=0)


def convert_episode(raw_dir: Path, dest: Path, force: bool = False) -> None:
    """Write ``dest`` (a ``<id>.zarr`` directory) from one raw episode folder."""
    marker = dest.with_name(dest.name + ".ok")
    if not force and marker.exists() and (dest / "zarr.json").exists():
        return
    marker.unlink(missing_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    meta = json.loads((raw_dir / "meta.json").read_text())
    hand_pose = json.loads((raw_dir / "hand_pose.json").read_text())
    intrinsics = json.loads((raw_dir / "intrinsics.json").read_text())
    tracks = tracks_from_hand_pose(hand_pose)
    assert_wrist_is_near_the_camera(tracks)
    images = _decode_video(raw_dir / "video.mp4", len(tracks["obs_head_pose"]))
    count = len(images)
    numeric = {key: value[:count] for key, value in tracks.items()}
    text = (meta.get("task_description") or meta.get("task_id") or "").strip()
    text = " ".join(text.split())
    ZarrWriter.create_and_write(
        episode_path=dest,
        numeric_data=numeric,
        image_data={"images.front_1": images},
        annotations=[(text, 0, count)],
        fps=30,
        embodiment="human_bimanual",
        task_name=str(meta.get("task_id") or "folding_clothes"),
        task_description=text,
        intrinsics={"front_1": intrinsics_matrix(intrinsics, OUT_W, OUT_H)},
        extrinsics=None,
        metadata_override={
            "episode_id": meta.get("episode_id"),
            "user_id": meta.get("user_id"),
            "duration": meta.get("duration"),
            "task_id": meta.get("task_id"),
            "environment_id": meta.get("environment_id"),
            "scene_id": meta.get("scene_id"),
        },
    )
    marker.write_text("ok\n")


def _episode_ids(spec: dict) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    groups = [tier["episodes"] for tier in spec["tiers"].values()]
    if "op1" in spec:
        groups.append(spec["op1"]["episodes"])
    groups.append(spec["seen_val"]["episodes"])
    groups.append(spec["unseen_val"]["episodes"])
    for group in spec.get("seen_val_by_rung", {}).values():
        groups.append(group["episodes"])
    for episodes in groups:
        for episode in episodes:
            if str(episode.get("source", "")).startswith("SQL app.episodes"):
                continue  # SQL validation episodes are downloaded as processed zarr
            episode_id = str(episode["episode_id"])
            if episode_id not in seen:
                seen.add(episode_id)
                ids.append(episode_id)
    return ids


def _convert_one(
    raw_root: str, zarr_root: str, episode_id: str, force: bool = False
) -> str:
    convert_episode(
        Path(raw_root) / episode_id, Path(zarr_root) / f"{episode_id}.zarr", force=force
    )
    return episode_id


def select_shard(episode_ids: list[str], shard: int, shards: int) -> list[str]:
    """Disjoint slice so several jobs can convert the same manifest at once."""
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError(f"shard {shard} of {shards} is out of range")
    if shards == 1:
        return list(episode_ids)
    return list(episode_ids)[shard::shards]


def select_range(episode_ids: list[str], offset: int, count: int | None) -> list[str]:
    """A contiguous slice, so busier nodes can take a larger share."""
    if offset < 0:
        raise ValueError(f"offset {offset} is negative")
    if count is None:
        return list(episode_ids)[offset:]
    if count < 0:
        raise ValueError(f"count {count} is negative")
    return list(episode_ids)[offset : offset + count]


def convert_all(
    root: Path,
    manifest: Path,
    workers: int,
    shard: int = 0,
    shards: int = 1,
    offset: int = 0,
    count: int | None = None,
    force: bool = False,
) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    spec = json.loads(manifest.read_text())
    raw_root = root / "raw"
    zarr_root = root / "zarr"
    zarr_root.mkdir(parents=True, exist_ok=True)
    episode_ids = _episode_ids(spec)
    if count is not None or offset:
        episode_ids = select_range(episode_ids, offset, count)
        label = f"offset {offset} count {len(episode_ids)}"
    else:
        episode_ids = select_shard(episode_ids, shard, shards)
        label = f"shard {shard}/{shards}"
    print(
        f"{label} converting {len(episode_ids)} episodes into {zarr_root}",
        flush=True,
    )
    failures: list[str] = []
    # Threads, not processes: decode and JPEG release the GIL, and a forked
    # worker would inherit the torch state this import chain pulls in.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _convert_one, str(raw_root), str(zarr_root), episode_id, force
            ): episode_id
            for episode_id in episode_ids
        }
        done = 0
        for future in as_completed(futures):
            episode_id = futures[future]
            done += 1
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{episode_id}: {exc}")
                print(f"FAIL {episode_id}: {exc}", flush=True)
            if done % 25 == 0 or done == len(episode_ids):
                print(
                    f"converted {done}/{len(episode_ids)}  failed {len(failures)}",
                    flush=True,
                )
            if len(failures) >= 5 and done == len(failures):
                pool.shutdown(cancel_futures=True, wait=False)
                break
    if failures:
        preview = "\n".join(failures[:20])
        raise RuntimeError(f"{len(failures)} conversion(s) failed:\n{preview}")


def link_splits(root: Path, manifest: Path) -> None:
    """Point each split folder at ``zarr/<id>.zarr`` and drop the raw symlinks."""
    spec = json.loads(manifest.read_text())
    folders = {
        name: spec["tiers"][name]["episodes"] for name in ("op4", "op8", "op12", "op16")
    }
    if "op1" in spec:
        folders["op1"] = spec["op1"]["episodes"]
    folders["seen_val"] = spec["seen_val"]["episodes"]
    folders["unseen_val"] = spec["unseen_val"]["episodes"]
    for tier, group in spec.get("seen_val_by_rung", {}).items():
        folders[f"{tier}_seen_val"] = group["episodes"]
    if "flagship_unseen_val" in spec:
        folders["flagship_unseen_val"] = spec["flagship_unseen_val"]["episodes"]
    for folder, episodes in folders.items():
        folder_dir = root / folder
        folder_dir.mkdir(parents=True, exist_ok=True)
        wanted: set[str] = set()
        for episode in episodes:
            episode_id = str(episode["episode_id"])
            name = f"{episode_id}.zarr"
            wanted.add(name)
            store = root / "zarr" / name
            if not (store / "zarr.json").exists():
                raise FileNotFoundError(f"missing zarr for {folder}/{episode_id}")
            link = folder_dir / name
            target = Path("..") / "zarr" / name
            if not (link.is_symlink() and link.readlink() == target):
                if link.is_symlink() or link.exists():
                    if link.is_dir() and not link.is_symlink():
                        shutil.rmtree(link)
                    else:
                        link.unlink()
                link.symlink_to(target)
            stale = folder_dir / episode_id
            if stale.is_symlink():
                stale.unlink()
        for child in list(folder_dir.iterdir()):
            if child.name not in wanted and child.is_symlink():
                child.unlink()
        print(f"linked {folder}: {len(wanted)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    convert_cmd = sub.add_parser(
        "convert", help="Write zarr stores for every manifest episode"
    )
    convert_cmd.add_argument("--root", type=Path, required=True)
    convert_cmd.add_argument("--manifest", type=Path, required=True)
    convert_cmd.add_argument("--workers", type=int, default=16)
    convert_cmd.add_argument("--shard", type=int, default=0)
    convert_cmd.add_argument("--shards", type=int, default=1)
    convert_cmd.add_argument("--offset", type=int, default=0)
    convert_cmd.add_argument("--count", type=int, default=None)
    convert_cmd.add_argument(
        "--force",
        action="store_true",
        help="rebuild zarr stores even when conversion markers exist",
    )
    link_cmd = sub.add_parser(
        "link", help="Repoint the split folders at the zarr stores"
    )
    link_cmd.add_argument("--root", type=Path, required=True)
    link_cmd.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "convert":
        convert_all(
            args.root,
            args.manifest,
            args.workers,
            shard=args.shard,
            shards=args.shards,
            offset=args.offset,
            count=args.count,
            force=args.force,
        )
        return
    link_splits(args.root, args.manifest)


if __name__ == "__main__":
    main()
