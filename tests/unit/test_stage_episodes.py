"""Node-local episode staging ($EGOVERSE_STAGE_DIR)."""

import os
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fixtures.synthetic_episodes import VENDORS, write_episode

from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr import zarr_dataset_multi as zdm
from egomimic.rldb.zarr.norm_cache import episode_fingerprint
from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalEpisodeResolver,
    MultiDataset,
    StagingError,
    stage_episodes,
)

VENDOR = next(v for v in sorted(VENDORS) if VENDORS[v].embodiment == "human_bimanual")


def _episodes(root, n=2):
    return {write_episode(root, VENDOR, seed=i).name[: -len(".zarr")] for i in range(n)}


def _count_copies(monkeypatch):
    calls = []
    real = shutil.copytree

    def counting(src, dst, *a, **k):
        if Path(src).name.endswith(".zarr"):  # copytree recurses into itself
            calls.append(Path(src))
        return real(src, dst, *a, **k)

    monkeypatch.setattr(zdm.shutil, "copytree", counting)
    return calls


def test_stages_and_keeps_fingerprints(tmp_path):
    src, stage = tmp_path / "nfs", tmp_path / "local"
    names = _episodes(src)
    staged = stage_episodes(src, stage, names | {"not_synced"})
    assert staged == names
    for n in names:
        assert episode_fingerprint(stage / f"{n}.zarr") == episode_fingerprint(
            src / f"{n}.zarr"
        )
    assert not list(stage.glob(".*.partial"))


def test_fresh_copies_are_reused(tmp_path, monkeypatch):
    src, stage = tmp_path / "nfs", tmp_path / "local"
    names = _episodes(src)
    stage_episodes(src, stage, names)
    calls = _count_copies(monkeypatch)
    assert stage_episodes(src, stage, names) == names
    assert calls == []


def test_changed_source_is_recopied(tmp_path, monkeypatch):
    src, stage = tmp_path / "nfs", tmp_path / "local"
    names = _episodes(src)
    stage_episodes(src, stage, names)
    changed = sorted(names)[0]
    meta = src / f"{changed}.zarr" / "zarr.json"
    st = meta.stat()
    os.utime(meta, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    calls = _count_copies(monkeypatch)
    assert stage_episodes(src, stage, names) == names
    assert [p.name for p in calls] == [f"{changed}.zarr"]


def test_concurrent_callers_copy_once(tmp_path, monkeypatch):
    src, stage = tmp_path / "nfs", tmp_path / "local"
    names = _episodes(src)
    calls = _count_copies(monkeypatch)
    results = []
    threads = [
        threading.Thread(
            target=lambda: results.append(stage_episodes(src, stage, names))
        )
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [names] * 4
    assert len(calls) == len(names)


def test_unwritable_stage_raises(tmp_path):
    src = tmp_path / "nfs"
    names = _episodes(src, n=1)
    blocker = tmp_path / "file"
    blocker.write_text("")
    with pytest.raises(StagingError, match="not writable"):
        stage_episodes(src, blocker / "stage", names)


def test_full_stage_raises(tmp_path, monkeypatch):
    src, stage = tmp_path / "nfs", tmp_path / "local"
    names = _episodes(src, n=1)
    monkeypatch.setattr(
        zdm.shutil, "disk_usage", lambda _: SimpleNamespace(total=100, used=95, free=5)
    )
    with pytest.raises(StagingError, match="free"):
        stage_episodes(src, stage, names)


def test_failed_copy_raises(tmp_path, monkeypatch):
    src, stage = tmp_path / "nfs", tmp_path / "local"
    names = _episodes(src, n=1)

    def boom(*a, **k):
        raise OSError("disk on fire")

    monkeypatch.setattr(zdm.shutil, "copytree", boom)
    with pytest.raises(StagingError, match="disk on fire"):
        stage_episodes(src, stage, names)
    assert not list(stage.glob(".*.partial"))


def test_resolver_reads_from_stage(tmp_path, monkeypatch):
    src, stage = tmp_path / "nfs", tmp_path / "local"
    names = _episodes(src)
    resolver = LocalEpisodeResolver(
        src,
        key_map=Human.get_keymap(keymap_mode="cartesian", annotation_key="annotations"),
        transform_list=Human.get_transform_list(
            mode="cartesian", stride=VENDORS[VENDOR].stride, allow_legacy_rotation=True
        ),
    )
    baseline = MultiDataset._from_resolver(resolver, mode="total")

    monkeypatch.setenv(zdm.STAGE_DIR_ENV, str(stage))
    datasets = resolver.load([(None, n) for n in names])
    assert sorted(datasets) == sorted(names)
    for ds in datasets.values():
        assert Path(ds.episode_path).parent == stage

    staged = MultiDataset._from_resolver(resolver, mode="total")
    assert len(staged) == len(baseline)
    np.testing.assert_array_equal(
        np.asarray(staged[0]["actions_cartesian"]),
        np.asarray(baseline[0]["actions_cartesian"]),
    )
