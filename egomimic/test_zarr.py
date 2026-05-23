import json

import numpy as np
import simplejpeg
import zarr

from egomimic.rldb.zarr.zarr_dataset_multi import ZarrEpisode


def validate_episode(zarr_path: str) -> tuple[list[str], list[str]]:
    """Returns (errors, successes). Empty errors list = pass."""
    errors: list[str] = []
    successes: list[str] = []
    ep = ZarrEpisode(zarr_path)
    meta = ep.metadata
    T = meta["total_frames"]
    store = zarr.open(zarr_path, mode="r")

    # ── Metadata ────────────────────────────────────────────────────────────
    for field in ("embodiment", "total_frames", "fps", "task_name", "features"):
        if field not in meta:
            errors.append(f"Missing metadata field: {field}")
        else:
            successes.append(f"metadata field present: {field}")

    if meta.get("fps", 0) not in (30, 60):
        errors.append(f"Unexpected fps={meta['fps']}. Expected 30 or 60.")
    else:
        successes.append(f"fps={meta['fps']} is valid")

    # ── Frame counts ────────────────────────────────────────────────────────
    features = meta.get("features", {})
    for key in store.keys():
        node = store[key]
        if not isinstance(node, zarr.Array):
            continue
        if features.get(key, {}).get("dtype") == "json":
            continue
        arr_len = node.shape[0]
        if arr_len < T:
            errors.append(f"{key}: array length {arr_len} < total_frames {T}")
        else:
            successes.append(f"{key}: frame count OK ({arr_len} >= {T})")

    # ── Required keys ───────────────────────────────────────────────────────
    required = ["images.front_1", "left.obs_ee_pose", "right.obs_ee_pose"]
    for key in required:
        if key not in store:
            errors.append(f"Missing required key: {key}")
        else:
            successes.append(f"required key present: {key}")

    # ── Pose shapes and norms ───────────────────────────────────────────────
    required_poses = ("left.obs_ee_pose", "right.obs_ee_pose")
    optional_poses = (
        "left.obs_wrist_pose",
        "right.obs_wrist_pose",
        "obs_head_pose",
        "left.cmd_ee_pose",
        "right.cmd_ee_pose",
    )
    for key in required_poses + optional_poses:
        if key in store:
            arr = store[key][:]
            if arr.shape != (T, 7) and arr.shape[0] >= T:
                arr = arr[:T]
            if arr.shape[-1] != 7:
                errors.append(f"{key}: expected shape (T, 7), got {arr.shape}")
                continue
            else:
                successes.append(f"{key}: shape OK (T, 7)")
            quat = arr[:, 3:7]
            norms = np.linalg.norm(quat, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-4):
                bad = np.where(np.abs(norms - 1.0) > 1e-4)[0]
                errors.append(
                    f"{key}: {len(bad)} frames with non-unit quaternions (e.g. frame {bad[0]}, norm={norms[bad[0]]:.6f})"
                )
            else:
                successes.append(f"{key}: all quaternions unit-norm")

    # ── Gripper shapes (optional) ───────────────────────────────────────────
    for key in (
        "left.obs_gripper",
        "right.obs_gripper",
        "left.gripper",
        "right.gripper",
    ):
        if key in store:
            arr = store[key][:]
            if arr.shape[0] < T:
                errors.append(f"{key}: array length {arr.shape[0]} < total_frames {T}")
                continue
            if arr.ndim != 2 or arr.shape[-1] != 1:
                errors.append(f"{key}: expected shape (T, 1), got {arr.shape}")
            else:
                successes.append(f"{key}: gripper shape OK (T, 1)")

    # ── Keypoint shapes ─────────────────────────────────────────────────────
    for key in ("left.obs_keypoints", "right.obs_keypoints"):
        if key in store:
            arr = store[key][:]
            if arr.shape[-1] != 63:
                errors.append(
                    f"{key}: expected last dim 63 (21×3), got {arr.shape[-1]}"
                )
            else:
                successes.append(f"{key}: keypoint shape OK (last dim = 63)")

    # ── Annotation format (JSON-encoded records) ────────────────────────────
    annotation_keys = [
        k for k, f in features.items() if f.get("dtype") == "json" and k in store
    ]
    for key in annotation_keys:
        node = store[key]
        n = node.shape[0]
        bad = 0
        first_err = None
        for i in range(n):
            raw = node[i]
            # Unwrap any nested 0-d object/bytes ndarrays down to raw bytes.
            while isinstance(raw, np.ndarray):
                raw = raw.item() if raw.shape == () else raw.flat[0]
            if isinstance(raw, np.bytes_):
                raw = bytes(raw)
            try:
                rec = json.loads(
                    raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
                )
                if not isinstance(rec, dict):
                    raise ValueError(f"record is {type(rec).__name__}, expected dict")
                for field, expected in (
                    ("text", str),
                    ("start_idx", int),
                    ("end_idx", int),
                ):
                    if field not in rec:
                        raise ValueError(f"missing field '{field}'")
                    if not isinstance(rec[field], expected):
                        raise ValueError(
                            f"field '{field}' is {type(rec[field]).__name__}, expected {expected.__name__}"
                        )
                if not (0 <= rec["start_idx"] <= rec["end_idx"] <= T):
                    raise ValueError(
                        f"index range invalid: start={rec['start_idx']}, end={rec['end_idx']}, T={T}"
                    )
            except Exception as e:
                bad += 1
                if first_err is None:
                    first_err = (i, str(e))
        if bad:
            errors.append(
                f"{key}: {bad}/{n} annotations malformed (e.g. index {first_err[0]}: {first_err[1]})"
            )
        else:
            successes.append(f"{key}: all {n} annotations well-formed")

    # ── Image decodability (spot-check first frame of each JPEG key) ────────
    jpeg_keys = [
        k for k, f in features.items() if f.get("dtype") == "jpeg" and k in store
    ]
    for key in jpeg_keys:
        data = ep.read({key: (0, None)})
        try:
            frame = simplejpeg.decode_jpeg(bytes(data[key]), colorspace="RGB")
            if frame.ndim != 3 or frame.shape[2] != 3:
                errors.append(
                    f"{key}: decoded frame has unexpected shape {frame.shape}"
                )
            else:
                successes.append(f"{key}: frame 0 decoded OK, shape={frame.shape}")
        except Exception as e:
            errors.append(f"{key}: failed to decode frame 0: {e}")

    return errors, successes


# Usage
errors, successes = validate_episode(
    "/storage/project/r-dxu345-0/shared/pick_place/2026-03-17-18-09-03-000000"
)
for s in successes:
    print("OK:", s)
if errors:
    for e in errors:
        print("ERROR:", e)
else:
    print("All checks passed.")
