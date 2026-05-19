"""Tests for the episode-level pause/idle precompute."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from egomimic.rldb.zarr.zarr_dataset_multi import (
    PAUSE_DETECT_KEYS,
    EpisodeResolver,
    LocalEpisodeResolver,
    ZarrDataset,
    _build_pause_keep_mask,
)


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


@pytest.mark.parametrize(
    "pause_spans,n,expected_dropped",
    [
        ([], 10, 0),
        ([(3, 8)], 10, 4),
        ([(0, 5)], 10, 3),
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
    assert len(ds_on) == n_frames

    n_raw, n_kept = ds_on.precompute_pause_filter()
    assert n_raw == n_frames
    expected_keep = _build_pause_keep_mask(
        left_pose=ds_on.episode_reader._store[PAUSE_DETECT_KEYS[0]][:],
        right_pose=ds_on.episode_reader._store[PAUSE_DETECT_KEYS[1]][:],
        epsilon=0.005,
    )
    assert n_kept == int(expected_keep.sum())
    assert len(ds_on) == n_kept
    assert n_kept < n_frames


def test_zarr_dataset_getitem_uses_keep_indices(tmp_path):
    ep = tmp_path / "ep_idx.zarr"
    n_frames = 20
    pause_spans = [(4, 10)]
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
    assert int(keep[0]) == 0
    assert int(keep[-1]) < n_frames

    sample_0 = ds[0]
    raw_left = ds.episode_reader._store["left.obs_ee_pose"][:]
    np.testing.assert_allclose(
        sample_0["left.obs_ee_pose"].numpy(), raw_left[int(keep[0])]
    )

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


@pytest.fixture
def _clean_modal_env(monkeypatch):
    monkeypatch.delenv("MODAL_IS_REMOTE", raising=False)
    monkeypatch.delenv("MODAL_TASK_ID", raising=False)
    monkeypatch.delenv("EGOMIMIC_DISABLE_MODAL_PAUSE_PRECOMPUTE", raising=False)


def _stub_dataset(path: str):
    return SimpleNamespace(episode_path=path)


def test_dispatch_off_outside_modal(_clean_modal_env):
    datasets = {"a": _stub_dataset("/mnt/zarr-data/a.zarr")}
    assert EpisodeResolver._should_use_modal_pause_precompute(datasets) is False


def test_dispatch_on_inside_modal_with_volume_paths(_clean_modal_env, monkeypatch):
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    datasets = {
        "a": _stub_dataset("/mnt/zarr-data/a.zarr"),
        "b": _stub_dataset("/mnt/zarr-data/b.zarr"),
    }
    assert EpisodeResolver._should_use_modal_pause_precompute(datasets) is True


def test_dispatch_on_when_modal_task_id_set(_clean_modal_env, monkeypatch):
    monkeypatch.setenv("MODAL_TASK_ID", "ta-xxxxxxxx")
    datasets = {"a": _stub_dataset("/mnt/zarr-data/a.zarr")}
    assert EpisodeResolver._should_use_modal_pause_precompute(datasets) is True


def test_dispatch_off_when_disabled_env(_clean_modal_env, monkeypatch):
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    monkeypatch.setenv("EGOMIMIC_DISABLE_MODAL_PAUSE_PRECOMPUTE", "1")
    datasets = {"a": _stub_dataset("/mnt/zarr-data/a.zarr")}
    assert EpisodeResolver._should_use_modal_pause_precompute(datasets) is False


def test_dispatch_off_when_paths_not_on_volume(_clean_modal_env, monkeypatch):
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    datasets = {"a": _stub_dataset("/tmp/elsewhere/a.zarr")}
    assert EpisodeResolver._should_use_modal_pause_precompute(datasets) is False


def test_run_pause_precompute_uses_local_path_outside_modal(tmp_path, _clean_modal_env):
    ep = tmp_path / "ep_dispatch.zarr"
    _write_synthetic_mecka_episode(ep, n_frames=20, pause_spans=[(4, 10)])
    key_map = {
        "left.obs_ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "left.obs_ee_pose",
        },
    }
    ds = ZarrDataset(ep, key_map=key_map, pause_removal_epsilon=0.005)
    resolver = LocalEpisodeResolver(
        folder_path=tmp_path,
        key_map=key_map,
        pause_removal_epsilon=0.005,
    )
    resolver._run_pause_precompute({"ep_dispatch": ds})
    assert ds.keep_indices is not None
    assert len(ds.keep_indices) < 20


def test_run_pause_precompute_falls_back_when_modal_lookup_fails(
    tmp_path, _clean_modal_env, monkeypatch
):
    """When dispatch picks Modal but the function lookup raises, fall back to local."""
    ep = tmp_path / "ep_fallback.zarr"
    _write_synthetic_mecka_episode(ep, n_frames=20, pause_spans=[(4, 10)])
    key_map = {
        "left.obs_ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "left.obs_ee_pose",
        },
    }
    ds = ZarrDataset(ep, key_map=key_map, pause_removal_epsilon=0.005)
    # Force the local dataset to look like it lives on the Modal volume so
    # _should_use_modal_pause_precompute returns True.
    ds.episode_path = Path("/mnt/zarr-data/ep_fallback.zarr")
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")

    resolver = LocalEpisodeResolver(
        folder_path=tmp_path,
        key_map=key_map,
        pause_removal_epsilon=0.005,
    )

    def _broken_modal_fanout(_self, _datasets):
        raise RuntimeError("simulated function lookup failure")

    monkeypatch.setattr(
        EpisodeResolver,
        "_modal_fanout_pause_precompute",
        _broken_modal_fanout,
    )
    # Path on disk is real; restore for the local fallback read.
    real_path = ep
    ds.episode_path = real_path

    # Tell dispatch to think this is on the Modal volume via the stub episode_path.
    class _PathProxy:
        def __init__(self, real, advertised):
            self._real = real
            self._advertised = advertised

        def __str__(self):
            return self._advertised

        def __fspath__(self):
            return str(self._real)

    ds.episode_path = _PathProxy(real_path, "/mnt/zarr-data/ep_fallback.zarr")
    resolver._run_pause_precompute({"ep_fallback": ds})
    assert ds.keep_indices is not None
    assert len(ds.keep_indices) < 20


def test_modal_pause_precompute_shard_is_registered():
    """Guards against accidental deletion of the deployed Modal worker."""
    pytest.importorskip("modal")
    from egomimic.modal import scan as scan_mod

    assert hasattr(scan_mod, "pause_precompute_shard")
    assert scan_mod.pause_precompute_shard.app is scan_mod.app
