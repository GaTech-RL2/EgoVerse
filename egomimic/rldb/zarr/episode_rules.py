"""
Episode-level rules shared by ZarrWriter.write and the episode validator, so the
two apply the same attr rules (CONTRIBUTING_DATA.md §6). The validator also
checks arrays (required keys, poses), which the writer does not.
"""

import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

SUPPORTED_FPS = (30, 60)

# Earliest plausible recording start; earlier values are usually seconds or ns
# passed where epoch milliseconds were expected (§4).
MIN_START_TIME_MS = 1_420_070_400_000  # 2015-01-01 UTC
_EPISODE_HASH_FORMAT = "%Y-%m-%d-%H-%M-%S-%f"

# Max |‖q‖ - 1| for pose quaternions; the MCAP validator uses the same value.
QUAT_UNIT_TOL = 1e-3

# Stamped in zarr.attrs by ZarrWriter. Episodes without it predate the required
# metadata rules; bump it when a rule change should not apply to older episodes.
SCHEMA_VERSION = 1

# Must match egomimic.rldb.embodiment.embodiment.EMBODIMENT (lowercased); a unit
# test checks. Listed here so the validators don't import torch.
EMBODIMENTS = (
    "human_right_arm",
    "human_left_arm",
    "human_bimanual",
    "eva_right_arm",
    "eva_left_arm",
    "eva_bimanual",
)


def embodiment_error(embodiment: Any) -> str | None:
    """Why ``embodiment`` is not a valid embodiment name (§9), or None."""
    if isinstance(embodiment, str) and embodiment.lower() in EMBODIMENTS:
        return None
    return (
        f"embodiment must be one of {list(EMBODIMENTS)}, got {embodiment!r}. "
        "See CONTRIBUTING_DATA.md §9."
    )


def arm_sides(embodiment: str) -> tuple[str, ...]:
    """Sides the embodiment uses: ("left", "right") for *_bimanual."""
    embodiment = embodiment.lower()
    if embodiment.endswith("_left_arm"):
        return ("left",)
    if embodiment.endswith("_right_arm"):
        return ("right",)
    return ("left", "right")


def fps_error(fps: Any) -> str | None:
    """Why ``fps`` is not a supported frame rate, or None."""
    # type() rather than isinstance: bool and np.int64 would pass the `in` test
    # and then fail (or change type) when written to zarr.json.
    if type(fps) is int and fps in SUPPORTED_FPS:
        return None
    return f"fps must be an int in {SUPPORTED_FPS}, got {fps!r}. See CONTRIBUTING_DATA.md §6.3."


def start_time_error(start_time_ms: Any) -> str | None:
    """Why ``start_time_ms`` is not a plausible UTC epoch-ms recording start, or None."""
    now_ms = int(time.time() * 1000)
    if type(start_time_ms) is int and MIN_START_TIME_MS <= start_time_ms <= now_ms:
        return None
    return (
        f"recording start time {start_time_ms!r} is not UTC epoch milliseconds "
        "between 2015-01-01 and now (seconds or nanoseconds passed by mistake?). "
        "See CONTRIBUTING_DATA.md §4."
    )


def episode_hash_start_ms(name: str) -> int | None:
    """Epoch ms encoded in an episode hash like 2026-01-12-03-47-29-664000, or None."""
    try:
        dt = datetime.strptime(name, _EPISODE_HASH_FORMAT)
    except ValueError:
        return None
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _matrix_shape(value: Any) -> tuple[int, ...] | None:
    try:
        return np.asarray(value, dtype=np.float64).shape
    except (TypeError, ValueError):
        return None


def camera_errors(intrinsics: Any, extrinsics: Any) -> list[str]:
    """Problems with the intrinsics / extrinsics dicts (§6.4)."""
    errors = []
    if not isinstance(intrinsics, dict) or not intrinsics:
        errors.append(
            "intrinsics must be a non-empty {camera_key: 3x4 K matrix} dict, "
            'e.g. {"front_1": K}. See CONTRIBUTING_DATA.md §6.4.'
        )
    else:
        if not any("front" in str(k).lower() for k in intrinsics):
            errors.append(
                f"intrinsics has no front-camera entry (keys: {list(intrinsics)}); "
                'key the egocentric camera "front_1". See CONTRIBUTING_DATA.md §6.4.'
            )
        for cam, K in intrinsics.items():
            if (shape := _matrix_shape(K)) != (3, 4):
                errors.append(
                    f"intrinsics[{cam!r}] must be a 3x4 K matrix (pad a bare 3x3 "
                    f"with np.hstack([K, np.zeros((3, 1))])), got {shape or K!r}. "
                    "See CONTRIBUTING_DATA.md §6.4."
                )
                continue
            K = np.asarray(K, dtype=np.float64)
            if not np.isfinite(K).all():
                errors.append(f"intrinsics[{cam!r}] contains NaN or inf")
            elif not (K[0, 0] > 0 and K[1, 1] > 0):
                errors.append(
                    f"intrinsics[{cam!r}] must have positive focal lengths "
                    f"(fx={K[0, 0]}, fy={K[1, 1]}). See CONTRIBUTING_DATA.md §6.4."
                )
    if extrinsics is None:
        return errors
    if not isinstance(extrinsics, dict) or not extrinsics:
        errors.append(
            "extrinsics must be None or a non-empty {key: 4x4} dict (robots key "
            'per-arm, e.g. {"left": T, "right": T}). See CONTRIBUTING_DATA.md §6.4.'
        )
        return errors
    for key, T in extrinsics.items():
        if (shape := _matrix_shape(T)) != (4, 4):
            errors.append(
                f"extrinsics[{key!r}] must be a 4x4 transform matrix, got "
                f"{shape or T!r}. See CONTRIBUTING_DATA.md §6.4."
            )
        elif not np.isfinite(np.asarray(T, dtype=np.float64)).all():
            errors.append(f"extrinsics[{key!r}] contains NaN or inf")
    return errors


def matrices_to_jsonable(matrices: dict) -> dict:
    """{key: matrix} as plain nested float lists for zarr.json."""
    return {k: np.asarray(v, dtype=np.float64).tolist() for k, v in matrices.items()}
