from pathlib import Path

import pytest

from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalEpisodeResolver,
    S3EpisodeResolver,
)


class _DummyDataset:
    def __len__(self) -> int:
        return 4


def _resolver_key_map() -> dict:
    return {"observations.state": {"zarr_key": "state"}}


def _write_episode_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_dummy_datasets(
    _self, search_path: Path, valid_folder_names: set[str]
) -> dict:
    return {name: _DummyDataset() for name in valid_folder_names}


@pytest.mark.parametrize(
    ("processed_path", "expected"),
    [
        ("s3://rldb/path/episode_1", "s3://rldb/path/episode_1/*"),
        ("processed_v3/scale/episode_1", "s3://rldb/processed_v3/scale/episode_1/*"),
    ],
)
def test_build_src_prefix(processed_path: str, expected: str) -> None:
    assert S3EpisodeResolver._build_src_prefix("rldb", processed_path) == expected


def test_build_sync_jobs_uses_final_and_partial_paths(tmp_path: Path) -> None:
    jobs = S3EpisodeResolver._build_sync_jobs(
        bucket_name="rldb",
        s3_paths=[("processed_v3/scale/episode_1", "episode_1")],
        local_dir=tmp_path,
    )

    assert len(jobs) == 1
    job = jobs[0]
    assert job.episode_hash == "episode_1"
    assert job.src_prefix == "s3://rldb/processed_v3/scale/episode_1/*"
    assert job.final_path == tmp_path / "episode_1"
    assert job.partial_path == tmp_path / "episode_1.partial"


def test_episode_already_present_rejects_missing_final(tmp_path: Path) -> None:
    assert S3EpisodeResolver._episode_already_present(tmp_path, "episode_1") is False


def test_episode_already_present_accepts_final_without_partial(tmp_path: Path) -> None:
    final_path = tmp_path / "episode_1"
    _write_episode_dir(final_path)

    assert S3EpisodeResolver._episode_already_present(tmp_path, "episode_1") is True
    assert final_path.is_dir()


def test_episode_already_present_rejects_when_partial_exists(tmp_path: Path) -> None:
    final_path = tmp_path / "episode_1"
    partial_path = tmp_path / "episode_1.partial"
    _write_episode_dir(final_path)
    _write_episode_dir(partial_path)

    assert S3EpisodeResolver._episode_already_present(tmp_path, "episode_1") is False
    assert final_path.is_dir()
    assert partial_path.is_dir()


def test_prepare_sync_job_removes_stale_partial_and_existing_final(
    tmp_path: Path,
) -> None:
    _write_episode_dir(tmp_path / "episode_1.partial")
    _write_episode_dir(tmp_path / "episode_1")
    job = S3EpisodeResolver._build_sync_jobs(
        bucket_name="rldb",
        s3_paths=[("processed_v3/scale/episode_1", "episode_1")],
        local_dir=tmp_path,
    )[0]

    S3EpisodeResolver._prepare_sync_job(job)

    assert not job.partial_path.exists()
    assert not job.final_path.exists()


def test_promote_synced_jobs_promotes_existing_partial_and_collects_missing(
    tmp_path: Path,
) -> None:
    valid_job, missing_job = S3EpisodeResolver._build_sync_jobs(
        bucket_name="rldb",
        s3_paths=[
            ("processed_v3/scale/episode_valid", "episode_valid"),
            ("processed_v3/scale/episode_missing", "episode_missing"),
        ],
        local_dir=tmp_path,
    )
    _write_episode_dir(valid_job.partial_path)

    failures = S3EpisodeResolver._promote_synced_jobs([valid_job, missing_job])

    assert valid_job.final_path.is_dir()
    assert not valid_job.partial_path.exists()
    assert len(failures) == 1
    assert failures[0][0].episode_hash == "episode_missing"
    assert "missing_partial_dir" in failures[0][1]
    assert not missing_job.partial_path.exists()
    assert not missing_job.final_path.exists()


def test_sync_s3_to_local_noops_for_empty_paths(tmp_path: Path) -> None:
    S3EpisodeResolver._sync_s3_to_local(
        bucket_name="rldb",
        s3_paths=[],
        local_dir=tmp_path,
        numworkers=4,
    )

    assert list(tmp_path.iterdir()) == []


def test_sync_s3_to_local_retries_missing_partial_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: dict[str, int] = {}

    def fake_run(cls, jobs, numworkers=10):
        for job in jobs:
            attempts[job.episode_hash] = attempts.get(job.episode_hash, 0) + 1
            if attempts[job.episode_hash] == 2:
                _write_episode_dir(job.partial_path)
        return 0

    monkeypatch.setattr(
        S3EpisodeResolver,
        "_run_s5cmd_batch",
        classmethod(fake_run),
    )

    S3EpisodeResolver._sync_s3_to_local(
        bucket_name="rldb",
        s3_paths=[("s3://rldb/path/episode_1", "episode_1")],
        local_dir=tmp_path,
        numworkers=4,
    )

    assert attempts == {"episode_1": 2}
    assert (tmp_path / "episode_1").is_dir()
    assert not (tmp_path / "episode_1.partial").exists()


def test_sync_s3_to_local_raises_after_missing_partial_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: dict[str, int] = {}

    def fake_run(cls, jobs, numworkers=10):
        for job in jobs:
            attempts[job.episode_hash] = attempts.get(job.episode_hash, 0) + 1
        return 0

    monkeypatch.setattr(
        S3EpisodeResolver,
        "_run_s5cmd_batch",
        classmethod(fake_run),
    )

    with pytest.raises(RuntimeError, match="episode_1"):
        S3EpisodeResolver._sync_s3_to_local(
            bucket_name="rldb",
            s3_paths=[("s3://rldb/path/episode_1", "episode_1")],
            local_dir=tmp_path,
            numworkers=4,
        )

    assert attempts == {"episode_1": 2}
    assert not (tmp_path / "episode_1").exists()
    assert not (tmp_path / "episode_1.partial").exists()


def test_sync_s3_to_local_only_syncs_partial_and_missing_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_episode_dir(tmp_path / "episode_keep")
    _write_episode_dir(tmp_path / "episode_corrupt_but_trusted")
    _write_episode_dir(tmp_path / "episode_replace")
    _write_episode_dir(tmp_path / "episode_replace.partial")
    sync_calls: list[list[str]] = []

    def fake_run(cls, jobs, numworkers=10):
        sync_calls.append([job.episode_hash for job in jobs])
        for job in jobs:
            _write_episode_dir(job.partial_path)
        return 0

    monkeypatch.setattr(
        S3EpisodeResolver,
        "_run_s5cmd_batch",
        classmethod(fake_run),
    )

    S3EpisodeResolver._sync_s3_to_local(
        bucket_name="rldb",
        s3_paths=[
            ("processed_v3/scale/episode_keep", "episode_keep"),
            (
                "processed_v3/scale/episode_corrupt_but_trusted",
                "episode_corrupt_but_trusted",
            ),
            ("processed_v3/scale/episode_replace", "episode_replace"),
            ("processed_v3/scale/episode_new", "episode_new"),
        ],
        local_dir=tmp_path,
        numworkers=4,
    )

    assert sync_calls == [["episode_replace", "episode_new"]]
    assert (tmp_path / "episode_keep").is_dir()
    assert (tmp_path / "episode_corrupt_but_trusted").is_dir()
    assert (tmp_path / "episode_replace").is_dir()
    assert (tmp_path / "episode_new").is_dir()
    assert not (tmp_path / "episode_replace.partial").exists()
    assert not (tmp_path / "episode_new.partial").exists()


def test_sync_from_filters_returns_empty_without_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync_calls: list[tuple] = []

    monkeypatch.setattr(
        S3EpisodeResolver,
        "_get_filtered_paths",
        staticmethod(lambda filters=None, debug=False: []),
    )
    monkeypatch.setattr(
        S3EpisodeResolver,
        "_sync_s3_to_local",
        classmethod(
            lambda cls,
            bucket_name,
            s3_paths,
            local_dir,
            numworkers=10: sync_calls.append(
                (bucket_name, s3_paths, local_dir, numworkers)
            )
        ),
    )

    result = S3EpisodeResolver.sync_from_filters(
        bucket_name="rldb",
        filters={"episode_hash": "missing"},
        local_dir=tmp_path,
        numworkers=17,
    )

    assert result == []
    assert sync_calls == []


def test_sync_from_filters_forwards_numworkers_and_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict = {}
    filtered_paths = [("processed_v3/scale/episode_1", "episode_1")]

    monkeypatch.setattr(
        S3EpisodeResolver,
        "_get_filtered_paths",
        staticmethod(lambda filters=None, debug=False: filtered_paths),
    )

    def fake_sync(cls, bucket_name, s3_paths, local_dir, numworkers=10):
        observed.update(
            {
                "bucket_name": bucket_name,
                "s3_paths": s3_paths,
                "local_dir": local_dir,
                "numworkers": numworkers,
            }
        )

    monkeypatch.setattr(
        S3EpisodeResolver,
        "_sync_s3_to_local",
        classmethod(fake_sync),
    )

    result = S3EpisodeResolver.sync_from_filters(
        bucket_name="rldb",
        filters={"episode_hash": "episode_1"},
        local_dir=tmp_path,
        numworkers=23,
    )

    assert result == filtered_paths
    assert observed == {
        "bucket_name": "rldb",
        "s3_paths": filtered_paths,
        "local_dir": tmp_path,
        "numworkers": 23,
    }


def test_resolve_skips_sync_for_cached_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_episode_dir(tmp_path / "episode_1")
    sync_calls: list[list[str]] = []

    monkeypatch.setattr(
        S3EpisodeResolver,
        "_get_filtered_paths",
        staticmethod(
            lambda filters=None, debug=False: [
                ("s3://rldb/path/episode_1", "episode_1")
            ]
        ),
    )
    monkeypatch.setattr(
        S3EpisodeResolver,
        "_run_s5cmd_batch",
        classmethod(
            lambda cls, jobs, numworkers=10: sync_calls.append(
                [job.episode_hash for job in jobs]
            )
        ),
    )
    monkeypatch.setattr(
        S3EpisodeResolver,
        "_load_zarr_datasets",
        _load_dummy_datasets,
    )

    resolver = S3EpisodeResolver(folder_path=tmp_path, key_map=_resolver_key_map())
    datasets = resolver.resolve(filters={"episode_hash": "episode_1"})

    assert set(datasets.keys()) == {"episode_1"}
    assert len(datasets["episode_1"]) == 4
    assert sync_calls == []


def test_resolve_resyncs_cached_episode_when_partial_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_episode_dir(tmp_path / "episode_1")
    _write_episode_dir(tmp_path / "episode_1.partial")
    sync_calls: list[list[str]] = []

    monkeypatch.setattr(
        S3EpisodeResolver,
        "_get_filtered_paths",
        staticmethod(
            lambda filters=None, debug=False: [
                ("s3://rldb/path/episode_1", "episode_1")
            ]
        ),
    )

    def fake_run(cls, jobs, numworkers=10):
        sync_calls.append([job.episode_hash for job in jobs])
        for job in jobs:
            _write_episode_dir(job.partial_path)
        return 0

    monkeypatch.setattr(
        S3EpisodeResolver,
        "_run_s5cmd_batch",
        classmethod(fake_run),
    )
    monkeypatch.setattr(
        S3EpisodeResolver,
        "_load_zarr_datasets",
        _load_dummy_datasets,
    )

    resolver = S3EpisodeResolver(folder_path=tmp_path, key_map=_resolver_key_map())
    datasets = resolver.resolve(filters={"episode_hash": "episode_1"})

    assert set(datasets.keys()) == {"episode_1"}
    assert len(datasets["episode_1"]) == 4
    assert sync_calls == [["episode_1"]]
    assert not (tmp_path / "episode_1.partial").exists()


def test_resolve_raises_when_no_filtered_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        S3EpisodeResolver,
        "_get_filtered_paths",
        staticmethod(lambda filters=None, debug=False: []),
    )

    resolver = S3EpisodeResolver(folder_path=tmp_path, key_map=_resolver_key_map())

    with pytest.raises(ValueError, match="filters matched no episodes"):
        resolver.resolve(filters={"episode_hash": "missing"})


def test_local_resolver_resolve_filters_and_loads_valid_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        LocalEpisodeResolver,
        "_get_local_filtered_paths",
        staticmethod(
            lambda search_path, filters=None, debug=False: [
                (str(search_path / "episode_1"), "episode_1")
            ]
        ),
    )
    monkeypatch.setattr(
        LocalEpisodeResolver,
        "_load_zarr_datasets",
        _load_dummy_datasets,
    )

    resolver = LocalEpisodeResolver(folder_path=tmp_path, key_map=_resolver_key_map())
    datasets = resolver.resolve(filters={"task_name": "pick"})

    assert set(datasets.keys()) == {"episode_1"}
    assert len(datasets["episode_1"]) == 4
