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
from egomimic.rldb.zarr.zarr_dataset_multi import (
    staged_episode_dir as ns,
)

VENDOR = next(v for v in sorted(VENDORS) if VENDORS[v].embodiment == "human_bimanual")


@pytest.fixture(autouse=True)
def _no_slurm(monkeypatch):
    for var in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_RESTART_COUNT"):
        monkeypatch.delenv(var, raising=False)


def _launch(monkeypatch, job="7", step="0"):
    monkeypatch.setenv("SLURM_JOB_ID", job)
    monkeypatch.setenv("SLURM_STEP_ID", step)


def _count_src_checks(monkeypatch, src):
    from egomimic.rldb.zarr import norm_cache

    calls = []
    real = norm_cache.episode_fingerprint

    def counting(path):
        if Path(path).parent == src:
            calls.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(norm_cache, "episode_fingerprint", counting)
    return calls


def _episodes(root, n=2):
    return {write_episode(root, VENDOR, seed=i).name[: -len(".zarr")] for i in range(n)}


def _count_copies(monkeypatch):
    calls = []
    real = zdm._tar_copy

    def counting(src_root, names, dest):
        calls.extend(Path(src_root) / n for n in names)
        return real(src_root, names, dest)

    monkeypatch.setattr(zdm, "_tar_copy", counting)
    return calls


def test_stages_and_keeps_fingerprints(tmp_path):
    src, stage = tmp_path / "nfs", tmp_path / "local"
    names = _episodes(src)
    staged = stage_episodes(src, stage, names | {"not_synced"})
    assert staged == names
    for n in names:
        assert episode_fingerprint(ns(stage, src) / f"{n}.zarr") == episode_fingerprint(
            src / f"{n}.zarr"
        )
    assert not (stage / ".incoming").exists()


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

    monkeypatch.setattr(zdm, "_tar_copy", boom)
    with pytest.raises(StagingError, match="disk on fire"):
        stage_episodes(src, stage, names)
    assert not list(stage.glob(".incoming*"))


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
        assert Path(ds.episode_path).parent == ns(stage, src)

    staged = MultiDataset._from_resolver(resolver, mode="total")
    assert len(staged) == len(baseline)
    np.testing.assert_array_equal(
        np.asarray(staged[0]["actions_cartesian"]),
        np.asarray(baseline[0]["actions_cartesian"]),
    )


def test_launch_checks_the_source_once(tmp_path, monkeypatch):
    """The node's other ranks and later heads of one launch reuse the first
    caller's check instead of re-reading the source over NFS."""
    src, stage = tmp_path / "nfs", tmp_path / "local"
    names = _episodes(src, n=3)
    _launch(monkeypatch)
    checks = _count_src_checks(monkeypatch, src)
    assert stage_episodes(src, stage, names | {"not_synced"}) == names
    assert len(checks) == 3
    subset = set(sorted(names)[:2])
    assert stage_episodes(src, stage, subset) == subset
    assert stage_episodes(src, stage, names | {"not_synced"}) == names
    assert len(checks) == 3


def test_new_launch_rechecks_the_source(tmp_path, monkeypatch):
    src, stage = tmp_path / "nfs", tmp_path / "local"
    names = _episodes(src)
    _launch(monkeypatch, step="0")
    stage_episodes(src, stage, names)
    changed = sorted(names)[0]
    meta = src / f"{changed}.zarr" / "zarr.json"
    st = meta.stat()
    os.utime(meta, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    calls = _count_copies(monkeypatch)
    assert stage_episodes(src, stage, names) == names
    assert calls == []
    _launch(monkeypatch, step="1")
    assert stage_episodes(src, stage, names) == names
    assert [p.name for p in calls] == [f"{changed}.zarr"]


def test_launch_record_is_per_source(tmp_path, monkeypatch):
    stage = tmp_path / "local"
    names = _episodes(tmp_path / "a")
    _launch(monkeypatch)
    assert stage_episodes(tmp_path / "a", stage, names) == names
    assert stage_episodes(tmp_path / "b", stage, names) == set()


def _bump_mtime(path):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))


def test_versioned_copy_is_trusted_without_reading_the_source(tmp_path, monkeypatch):
    src, stage = tmp_path / "nfs", tmp_path / "local"
    names = _episodes(src)
    versions = {n: "v1" for n in names}
    _launch(monkeypatch, step="0")
    stage_episodes(src, stage, names, versions)
    _launch(monkeypatch, step="1")
    checks = _count_src_checks(monkeypatch, src)
    assert stage_episodes(src, stage, names, versions) == names
    assert checks == []


def test_new_version_rechecks_and_recopies(tmp_path, monkeypatch):
    src, stage = tmp_path / "nfs", tmp_path / "local"
    names = _episodes(src)
    stage_episodes(src, stage, names, {n: "v1" for n in names})
    changed = sorted(names)[0]
    _bump_mtime(src / f"{changed}.zarr" / "zarr.json")
    calls = _count_copies(monkeypatch)
    assert stage_episodes(src, stage, names, {n: "v1" for n in names}) == names
    assert calls == []
    versions = {n: "v1" for n in names} | {changed: "v2"}
    assert stage_episodes(src, stage, names, versions) == names
    assert [p.name for p in calls] == [f"{changed}.zarr"]


def test_damaged_local_copy_is_restaged(tmp_path, monkeypatch):
    src, stage = tmp_path / "nfs", tmp_path / "local"
    names = _episodes(src)
    versions = {n: "v1" for n in names}
    stage_episodes(src, stage, names, versions)
    damaged = sorted(names)[0]
    _bump_mtime(ns(stage, src) / f"{damaged}.zarr" / "zarr.json")
    calls = _count_copies(monkeypatch)
    assert stage_episodes(src, stage, names, versions) == names
    assert [p.name for p in calls] == [f"{damaged}.zarr"]


def test_s3_sync_runs_once_per_launch(tmp_path, monkeypatch):
    monkeypatch.setenv(zdm.STAGE_DIR_ENV, str(tmp_path / "local"))
    synced = []
    monkeypatch.setattr(
        zdm.S3EpisodeResolver,
        "_sync_s3_to_local_unlocked",
        classmethod(lambda cls, b, paths, d, n: synced.append([h for _, h in paths])),
    )
    paths = [("s3://b/p/a", "a"), ("s3://b/p/b", "b")]

    def sync(p):
        zdm.S3EpisodeResolver._sync_s3_to_local("b", p, tmp_path / "nfs")

    _launch(monkeypatch)
    sync(paths)
    sync(paths)
    sync(paths + [("s3://b/p/c", "c")])
    assert synced == [["a", "b"], [], ["c"]]
    _launch(monkeypatch, step="1")
    sync(paths)
    assert synced[-1] == ["a", "b"]


def test_s3_sync_outside_a_step_needs_no_stage_dir(tmp_path, monkeypatch):
    """The download script runs on the login node, which has no /ephemeral."""
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv(zdm.STAGE_DIR_ENV, str(blocker / "stage"))
    synced = []
    monkeypatch.setattr(
        zdm.S3EpisodeResolver,
        "_sync_s3_to_local_unlocked",
        classmethod(lambda cls, b, paths, d, n: synced.append([h for _, h in paths])),
    )
    zdm.S3EpisodeResolver._sync_s3_to_local("b", [("s3://b/p/a", "a")], tmp_path)
    _launch(monkeypatch)
    zdm.S3EpisodeResolver._sync_s3_to_local("b", [("s3://b/p/a", "a")], tmp_path)
    assert synced == [["a"], ["a"]]


def test_symlinked_source_is_staged_as_data(tmp_path):
    real, src, stage = tmp_path / "real", tmp_path / "nfs", tmp_path / "local"
    names = _episodes(real, n=1)
    src.mkdir()
    for n in names:
        (src / f"{n}.zarr").symlink_to(real / f"{n}.zarr")
    assert stage_episodes(src, stage, names) == names
    for n in names:
        staged = ns(stage, src) / f"{n}.zarr"
        assert staged.is_dir() and not staged.is_symlink()
        assert not any(p.is_symlink() for p in staged.rglob("*"))


def test_trust_is_per_source(tmp_path, monkeypatch):
    stage = tmp_path / "local"
    names = _episodes(tmp_path / "a", n=1)
    versions = {n: "v1" for n in names}
    assert stage_episodes(tmp_path / "a", stage, names, versions) == names
    assert stage_episodes(tmp_path / "missing", stage, names, versions) == set()


def test_trust_expires(tmp_path, monkeypatch):
    src, stage = tmp_path / "nfs", tmp_path / "local"
    names = _episodes(src, n=1)
    versions = {n: "v1" for n in names}
    stage_episodes(src, stage, names, versions)
    monkeypatch.setattr(zdm, "_STAGED_TRUST_S", -1)
    checks = _count_src_checks(monkeypatch, src)
    assert stage_episodes(src, stage, names, versions) == names
    assert len(checks) == 1


def test_failed_chunk_keeps_earlier_chunks(tmp_path, monkeypatch):
    src, stage = tmp_path / "nfs", tmp_path / "local"
    names = _episodes(src, n=3)
    monkeypatch.setattr(zdm, "_STAGE_CHUNK", 1)
    real = zdm._tar_copy
    calls = []

    def flaky(src_root, chunk, dest):
        calls.append(chunk)
        if len(calls) == 2:
            raise OSError("disk on fire")
        return real(src_root, chunk, dest)

    monkeypatch.setattr(zdm, "_tar_copy", flaky)
    versions = {n: "v1" for n in names}
    with pytest.raises(StagingError, match="disk on fire"):
        stage_episodes(src, stage, names, versions)
    first = calls[0][0][: -len(".zarr")]
    monkeypatch.setattr(zdm, "_tar_copy", real)
    recopied = _count_copies(monkeypatch)
    assert stage_episodes(src, stage, names, versions) == names
    assert f"{first}.zarr" not in [p.name for p in recopied]
    assert len(recopied) == 2


def test_leftover_incoming_is_removed(tmp_path):
    src, stage = tmp_path / "nfs", tmp_path / "local"
    names = _episodes(src, n=1)
    (stage / ".incoming" / "junk").mkdir(parents=True)
    assert stage_episodes(src, stage, names) == names
    assert not (stage / ".incoming").exists()


def test_tar_reports_errors_without_hanging(tmp_path):
    """Thousands of per-file warnings must not fill a pipe and deadlock."""
    if os.geteuid() == 0:
        pytest.skip("root reads mode-000 files")
    src = tmp_path / "nfs" / "ep.zarr"
    src.mkdir(parents=True)
    for i in range(3000):
        f = src / f"f{i:05d}_{'x' * 40}"
        f.write_text("")
        f.chmod(0)
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(OSError, match="Permission denied"):
        zdm._tar_copy(tmp_path / "nfs", ["ep.zarr"], dest)


def test_same_hash_from_two_sources_keeps_both(tmp_path):
    stage = tmp_path / "local"
    names = _episodes(tmp_path / "a", n=1)
    _episodes(tmp_path / "b", n=1)
    versions = {n: "v1" for n in names}
    assert stage_episodes(tmp_path / "a", stage, names, versions) == names
    assert stage_episodes(tmp_path / "b", stage, names, versions) == names
    calls = []
    for src in ("a", "b"):
        assert stage_episodes(tmp_path / src, stage, names, versions) == names
        calls += [ns(stage, tmp_path / src) / f"{n}.zarr" for n in names]
    assert all(p.is_dir() for p in calls)


def test_verified_name_needs_a_local_copy(tmp_path, monkeypatch):
    src, stage = tmp_path / "nfs", tmp_path / "local"
    names = _episodes(src)
    _launch(monkeypatch)
    stage_episodes(src, stage, names)
    gone = sorted(names)[0]
    shutil.rmtree(ns(stage, src) / f"{gone}.zarr")
    assert stage_episodes(src, stage, names) == names - {gone}


def test_restage_swaps_by_rename(tmp_path, monkeypatch):
    """The old copy is never deleted in place, so a kill can't leave a
    half-deleted episode that still fingerprints as intact."""
    src, stage = tmp_path / "nfs", tmp_path / "local"
    names = _episodes(src, n=1)
    stage_episodes(src, stage, names)
    (name,) = names
    _bump_mtime(src / f"{name}.zarr" / "zarr.json")
    deleted = []
    real = shutil.rmtree

    def watching(path, *a, **k):
        deleted.append(Path(path))
        return real(path, *a, **k)

    monkeypatch.setattr(zdm.shutil, "rmtree", watching)
    assert stage_episodes(src, stage, names) == names
    assert ns(stage, src) / f"{name}.zarr" not in deleted
    assert not (stage / ".trash").exists()


def test_full_disk_evicts_stale_unused_episodes(tmp_path, monkeypatch):
    src, stage = tmp_path / "nfs", tmp_path / "local"
    old = _episodes(tmp_path / "old", n=1)
    stage_episodes(tmp_path / "old", stage, old)
    (old_name,) = old
    old_dir = ns(stage, tmp_path / "old") / f"{old_name}.zarr"
    os.utime(old_dir, (0, 0))
    new = {write_episode(src, VENDOR, seed=7).name[: -len(".zarr")]}

    def usage(_):
        full = old_dir.exists()
        return SimpleNamespace(
            total=100, used=95 if full else 50, free=5 if full else 50
        )

    monkeypatch.setattr(zdm.shutil, "disk_usage", usage)
    assert stage_episodes(src, stage, new) == new
    assert not old_dir.exists()


def test_launch_key_without_a_step_uses_the_process_group(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    assert zdm._launch_key() == f"7.pg{os.getpgid(0)}.0"
    monkeypatch.setenv("SLURM_STEP_ID", "3")
    assert zdm._launch_key() == "7.s3.0"
