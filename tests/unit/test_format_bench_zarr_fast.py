"""The format bench's FastZarrDataset emits exactly ZarrDataset's samples, alone and batched.

The episode is ramped (cell = frame * 1000 + column) and 160 frames long, so windows cross the
100-row inner-chunk boundary and a wrong offset changes numbers."""

from __future__ import annotations

import numpy as np
import pytest
import simplejpeg
import torch

from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, ZarrDataset
from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.scripts.format_bench import zarr_fast
from egomimic.scripts.format_bench.zarr_fast import FastZarrDataset

T = 160


def _ramp(width: int) -> np.ndarray:
    return (
        np.arange(T, dtype=np.float64)[:, None] * 1000.0
        + np.arange(width, dtype=np.float64)[None, :]
    )


def _write(path, seed):
    numeric = {"obs_head_pose": _ramp(7)}
    for side, off in (("left", 0.0), ("right", 0.5)):
        numeric[f"{side}.obs_wrist_pose"] = _ramp(7) + off
        numeric[f"{side}.obs_keypoints"] = _ramp(63) + off
        numeric[f"{side}.obs_ee_pose"] = _ramp(7) + off
    rng = np.random.default_rng(seed)
    return ZarrWriter.create_and_write(
        path,
        numeric_data=numeric,
        image_data={"images.front_1": rng.integers(0, 255, (T, 32, 32, 3), np.uint8)},
        embodiment="human_bimanual",
        task_name="ramp",
        intrinsics={"front_1": np.eye(3, 4)},
    )


@pytest.fixture(scope="module")
def episodes(tmp_path_factory):
    d = tmp_path_factory.mktemp("fast")
    return [_write(d / f"ep{i}.zarr", i) for i in range(2)]


def _keymap(frames8: bool):
    km = Human.get_keymap(keymap_mode="keypoints", include_ee_pose=True)
    if frames8:
        km = {
            k: {**v, "history": 8} if v.get("key_type") == "camera_keys" else v
            for k, v in km.items()
        }
    return km


def _same(a: dict, b: dict):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], torch.Tensor):
            assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


@pytest.mark.parametrize("frames8", [False, True])
@pytest.mark.parametrize("idx", [0, 5, 70, 85, 99, 100, 130, T - 1])
def test_getitem_matches_zarr_dataset(episodes, frames8, idx):
    km = _keymap(frames8)
    _same(FastZarrDataset(episodes[0], km)[idx], ZarrDataset(episodes[0], km)[idx])


@pytest.mark.parametrize("frames8", [False, True])
def test_getitems_batches_into_one_fetch_and_matches(episodes, frames8, monkeypatch):
    km = _keymap(frames8)
    fast = MultiDataset(
        datasets={f"e{i}": FastZarrDataset(p, km) for i, p in enumerate(episodes)},
        mode="total",
    )
    ref = MultiDataset(
        datasets={f"e{i}": ZarrDataset(p, km) for i, p in enumerate(episodes)},
        mode="total",
    )
    calls = []
    real = zarr_fast.fetch_many
    monkeypatch.setattr(
        zarr_fast, "fetch_many", lambda items: calls.append(len(items)) or real(items)
    )
    idxs = [0, 3, 99, 150, T, T + 85, 2 * T - 1]
    batch = fast.__getitems__(idxs)
    assert calls == [len(idxs)], "one fetch for the whole batch"
    for i, got in zip(idxs, batch):
        _same(got, ref[i])


def test_batched_jpeg_failure_falls_back_to_a_fresh_read(episodes, monkeypatch):
    km = _keymap(False)
    ds = FastZarrDataset(episodes[0], km)
    multi = MultiDataset(datasets={"e0": ds}, mode="total")
    real = simplejpeg.decode_jpeg
    n = []

    def fail_once(data, **kw):
        n.append(1)
        if len(n) == 1:
            raise ValueError("corrupt jpeg")
        return real(data, **kw)

    monkeypatch.setattr(
        "egomimic.rldb.zarr.zarr_dataset_multi.simplejpeg.decode_jpeg", fail_once
    )
    (sample,) = multi.__getitems__([60])
    frame = int(round(float(np.asarray(sample["obs_head_pose"])[0]) / 1000.0))
    assert frame != 60, "the retry moved to another frame and read it"
    np.testing.assert_array_equal(
        np.asarray(sample["obs_head_pose"]), _ramp(7)[frame].astype(np.float32)
    )
    assert ds.episode_reader._cache()["prefetched"] is None
