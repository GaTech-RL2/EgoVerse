"""Smoke tests for the episode-level pause/idle precompute.

These cover:
  - the pure-numpy mask helper against the reference PauseRemovalTransform
    compress logic that lived previously in action_chunk_transforms.py,
  - the ZarrDataset integration on a synthetic single-episode store,
  - that __len__ shrinks and __getitem__ resolves chunks from the correct
    real frame, proving the dataset is actually altered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from egomimic.rldb.zarr.zarr_dataset_multi import (
    PAUSE_DETECT_KEYS,
    ZarrDataset,
    _build_pause_keep_mask,
)

# ---------------------------------------------------------------------------
# Reference compress logic — copy of the old PauseRemovalTransform algorithm.
# Used only to confirm the new mask helper matches the previous semantics.
# ---------------------------------------------------------------------------


def _reference_compress_keep_mask(chunk: np.ndarray, epsilon: float) -> np.ndarray:
    H = len(chunk)
    if H < 2:
        return np.ones(H, dtype=bool)
    deltas = np.linalg.norm(np.diff(chunk, axis=0), axis=-1)
    keep = np.ones(H, dtype=bool)
    in_pause = False
    for t in range(1, H):
        if deltas[t - 1] < epsilon:
            if in_pause:
                keep[t] = False
            else:
                in_pause = True
        else:
            in_pause = False
    return keep


# ---------------------------------------------------------------------------
# Synthetic episode helpers
# ---------------------------------------------------------------------------


def _synthetic_poses(
    *,
    n_frames: int,
    pause_spans: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build left/right obs_ee_pose arrays where frames inside any pause span
    are exactly identical (zero delta) and frames outside step by 0.1 in x.
    """
    left = np.zeros((n_frames, 7), dtype=np.float64)
    right = np.zeros((n_frames, 7), dtype=np.float64)
    left[:, 6] = 1.0  # quat w = 1
    right[:, 6] = 1.0
    motion = 0.0
    for i in range(n_frames):
        in_pause = any(s <= i < e for s, e in pause_spans)
        if not in_pause:
            motion += 0.1
        left[i, 0] = motion
        right[i, 0] = motion + 0.5  # offset right hand
    return left, right


def _write_synthetic_mecka_episode(
    path: Path,
    *,
    n_frames: int,
    pause_spans: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Write a minimal zarr v3 group at path with the keys ZarrDataset reads."""
    left, right = _synthetic_poses(n_frames=n_frames, pause_spans=pause_spans)

    head = np.zeros((n_frames, 7), dtype=np.float64)
    head[:, 6] = 1.0

    store = zarr.open(str(path), mode="w", zarr_format=3)
    store.create_array("left.obs_ee_pose", data=left, chunks=(min(100, n_frames), 7))
    store.create_array("right.obs_ee_pose", data=right, chunks=(min(100, n_frames), 7))
    store.create_array("obs_head_pose", data=head, chunks=(min(100, n_frames), 7))
    store.attrs.update(
        {
            "embodiment": "MECKA_BIMANUAL",
            "total_frames": n_frames,
            "fps": 30,
            "features": {
                "left.obs_ee_pose": {
                    "dtype": "float64",
                    "shape": [7],
                    "names": ["dim_0"],
                },
                "right.obs_ee_pose": {
                    "dtype": "float64",
                    "shape": [7],
                    "names": ["dim_0"],
                },
                "obs_head_pose": {
                    "dtype": "float64",
                    "shape": [7],
                    "names": ["dim_0"],
                },
            },
        }
    )
    return left, right


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pause_spans,n,expected_dropped",
    [
        ([], 10, 0),
        # 5-frame pause [3,8): transition frame at t=3 kept, drop t=4,5,6,7.
        ([(3, 8)], 10, 4),
        # Leading pause [0,5): transition at t=1 kept, drop t=2,3,4.
        ([(0, 5)], 10, 3),
        # Two pauses: [2,4) drops one frame; [6,9) drops two frames.
        ([(2, 4), (6, 9)], 10, 3),
    ],
)
def test_build_pause_keep_mask_matches_reference(pause_spans, n, expected_dropped):
    left, right = _synthetic_poses(n_frames=n, pause_spans=pause_spans)
    keep = _build_pause_keep_mask(left_pose=left, right_pose=right, epsilon=0.005)

    expected_left = _reference_compress_keep_mask(left, 0.005)
    expected_right = _reference_compress_keep_mask(right, 0.005)
    assert np.array_equal(keep, expected_left)
    assert np.array_equal(keep, expected_right)

    assert int((~keep).sum()) == expected_dropped


def test_build_pause_keep_mask_short_episode():
    # 0 and 1-frame edge cases — nothing to drop.
    for n in (0, 1):
        left = np.zeros((n, 7))
        right = np.zeros((n, 7))
        keep = _build_pause_keep_mask(left_pose=left, right_pose=right, epsilon=1.0)
        assert len(keep) == n
        assert keep.all()


def test_zarr_dataset_precompute_alters_length(tmp_path):
    ep = tmp_path / "ep_test.zarr"
    n_frames = 30
    pause_spans = [(5, 12), (20, 25)]  # 7-frame pause and 5-frame pause
    _write_synthetic_mecka_episode(ep, n_frames=n_frames, pause_spans=pause_spans)

    key_map = {
        "left.obs_ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "left.obs_ee_pose",
        },
        "right.action_ee_pose": {
            "key_type": "action_keys",
            "zarr_key": "right.obs_ee_pose",
            "horizon": 5,
        },
    }

    ds_off = ZarrDataset(ep, key_map=key_map, pause_removal_epsilon=None)
    assert len(ds_off) == n_frames
    assert ds_off.keep_indices is None

    ds_on = ZarrDataset(ep, key_map=key_map, pause_removal_epsilon=0.005)
    assert len(ds_on) == n_frames, (
        "Length should still match raw total_frames before precompute_pause_filter() "
        "is invoked, so PyTorch's index_map building (which calls __len__) sees the "
        "filtered count only after the resolver runs precompute."
    )

    n_raw, n_kept = ds_on.precompute_pause_filter()
    assert n_raw == n_frames
    expected_keep = _build_pause_keep_mask(
        left_pose=ds_on.episode_reader._store[PAUSE_DETECT_KEYS[0]][:],
        right_pose=ds_on.episode_reader._store[PAUSE_DETECT_KEYS[1]][:],
        epsilon=0.005,
    )
    assert n_kept == int(expected_keep.sum())
    assert len(ds_on) == n_kept
    assert n_kept < n_frames, "pause precompute must shrink an episode with pauses"


def test_zarr_dataset_getitem_uses_keep_indices(tmp_path):
    ep = tmp_path / "ep_idx.zarr"
    n_frames = 20
    pause_spans = [(4, 10)]  # frames 4-9 paused (5 dropped after transition at 4)
    _write_synthetic_mecka_episode(ep, n_frames=n_frames, pause_spans=pause_spans)

    key_map = {
        "left.obs_ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "left.obs_ee_pose",
        },
    }
    ds = ZarrDataset(ep, key_map=key_map, pause_removal_epsilon=0.005)
    ds.precompute_pause_filter()

    keep = ds.keep_indices
    assert keep is not None
    # First kept frame is always 0, last must be in-bounds.
    assert int(keep[0]) == 0
    assert int(keep[-1]) < n_frames

    # __getitem__ at logical idx 0 should return data from real frame keep[0].
    sample_0 = ds[0]
    raw_left = ds.episode_reader._store["left.obs_ee_pose"][:]
    np.testing.assert_allclose(
        sample_0["left.obs_ee_pose"].numpy(), raw_left[int(keep[0])]
    )

    # A logical idx that lands inside the post-filter range but past where the
    # original pause was must resolve to a non-paused frame in the raw episode.
    mid = len(keep) // 2
    sample_mid = ds[mid]
    np.testing.assert_allclose(
        sample_mid["left.obs_ee_pose"].numpy(), raw_left[int(keep[mid])]
    )


def test_zarr_dataset_precompute_is_idempotent(tmp_path):
    ep = tmp_path / "ep_idem.zarr"
    _write_synthetic_mecka_episode(ep, n_frames=15, pause_spans=[(5, 10)])
    key_map = {
        "left.obs_ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "left.obs_ee_pose",
        },
    }
    ds = ZarrDataset(ep, key_map=key_map, pause_removal_epsilon=0.005)
    a = ds.precompute_pause_filter()
    b = ds.precompute_pause_filter()
    assert a == b
    assert ds.keep_indices is not None
