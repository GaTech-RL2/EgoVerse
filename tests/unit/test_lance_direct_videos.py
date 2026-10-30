import builtins
import datetime
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

lance = pytest.importorskip("lance")
pytest.importorskip("av")

from fixtures.lance_tables import EVA_DELIVERY_FRAMES, eva_delivery  # noqa: E402

from egomimic.rldb.lance import commit, mp4_ranges  # noqa: E402
from egomimic.scripts import lance_commit  # noqa: E402
from egomimic.scripts.lance_commit import REJECTED, commit_staged  # noqa: E402

N = EVA_DELIVERY_FRAMES
ZERO = datetime.timedelta(0)


@pytest.fixture
def env(tmp_path):
    staging, root = tmp_path / "staging", tmp_path / "lance"
    built = iter(range(100))

    def build(vid, **kw):
        return eva_delivery(tmp_path / f"built{next(built)}", vid, **kw)

    def stage(vid, to=root, **kw):
        src = build(vid, **kw)
        commit.stage_direct(src, commit.join(staging, "eva", vid), str(to))
        meta = json.loads((staging / "eva" / vid / "meta.json").read_text())
        return meta["lance_videos"]

    def run(**kw):
        return commit_staged(str(staging), "eva", str(root), tmp_path / "work", **kw)

    return staging, root, build, stage, run


def test_place_videos_uploads_once_to_the_content_addressed_path(env, monkeypatch):
    staging, root, build, stage, run = env
    src = build("100")
    placed = commit.place_videos(src, str(root))
    rid = json.loads((src / "meta.json").read_text())["recording_id"]
    digest = commit.content_digest(
        pq.read_table(src / "steps.parquet"), {"front_1": src / "front_1.mp4"}
    )
    path = commit.video_path(rid, digest, "front_1")
    assert placed == {
        "root": str(root),
        "content_digest": digest,
        "videos": {
            "front_1": {
                "path": path,
                "size": (src / "front_1.mp4").stat().st_size,
                "num_frames": N,
            }
        },
    }
    assert (root / path).read_bytes() == (src / "front_1.mp4").read_bytes()

    uploads = []
    real_put = commit.put_file
    monkeypatch.setattr(
        commit, "put_file", lambda s, d: (uploads.append(d), real_put(s, d))
    )
    assert commit.place_videos(src, str(root)) == placed
    assert uploads == []
    (root / path).write_bytes(b"partial")
    assert commit.place_videos(src, str(root)) == placed
    assert uploads == [commit.join(str(root), path)]


def test_committer_checks_placed_videos_without_reading_them(env, monkeypatch):
    staging, root, build, stage, run = env
    placed = stage("100")
    stage("200")
    assert sorted(commit.list_files(staging / "eva" / "100")) == [
        "meta.json",
        "steps.parquet",
    ]

    touched = []

    def spy(mod, name):
        real = getattr(mod, name)

        def f(*a, **kw):
            touched.append((name, str(a[0])))
            return real(*a, **kw)

        monkeypatch.setattr(mod, name, f)

    for mod, name in (
        (commit, "get_file"),
        (commit, "get_bytes"),
        (commit, "put_file"),
        (shutil, "copy2"),
        (shutil, "copyfile"),
        (lance_commit, "read_range"),
        (mp4_ranges, "read_index"),
        (builtins, "open"),
    ):
        spy(mod, name)

    report = run()
    (batch,) = report["batches"]
    assert len(batch["added"]) == 2 and report["rejected"] == {}
    assert not [t for t in touched if t[1].endswith(".mp4") or t[0] == "read_index"]
    rec = (
        lance.dataset(root / "recordings.lance")
        .to_table(filter="vendor_recording_id = '100'")
        .to_pylist()[0]
    )
    assert rec["video_path"] == [("front_1", placed["videos"]["front_1"]["path"])]
    assert rec["content_digest"] == placed["content_digest"]


def test_placed_video_of_the_wrong_size_or_root_is_rejected(env, tmp_path):
    staging, root, build, stage, run = env
    placed = stage("100")
    stage("200", to=tmp_path / "other")
    (root / placed["videos"]["front_1"]["path"]).write_bytes(b"truncated")
    report = run()
    assert report["batches"] == []
    assert "is 9 bytes, meta.json says" in report["rejected"]["100"][0]
    assert "placed under" in report["rejected"]["200"][0]
    note = json.loads((staging / "eva" / "100" / REJECTED).read_text())
    assert note["errors"] == report["rejected"]["100"]
    assert run() == {"batches": [], "rejected": {}}


def test_frame_count_comes_from_meta(env):
    staging, root, build, stage, run = env
    stage("100", frames_in_video=N - 2)
    report = run()
    assert f"front_1.mp4 has {N - 2} frames" in report["rejected"]["100"][0]


def test_orphan_videos_lists_uncommitted_uploads_and_deletes_only_when_asked(env):
    staging, root, build, stage, run = env
    committed = stage("100")
    rejected = stage("200")
    (root / rejected["videos"]["front_1"]["path"]).write_bytes(b"truncated")
    run()
    pending = stage("300")
    unstaged = commit.place_videos(build("400"), str(root))

    def path(p):
        return commit.join(str(root), p["videos"]["front_1"]["path"])

    orphans = sorted(map(path, (rejected, unstaged)))
    listed = commit.orphan_videos(str(root), str(staging), upload_grace=ZERO)
    assert sorted(listed) == orphans
    assert commit.orphan_videos(str(root), str(staging)) == []
    assert all(Path(p).exists() for p in orphans)

    assert (
        sorted(
            commit.orphan_videos(
                str(root), str(staging), upload_grace=ZERO, delete=True
            )
        )
        == orphans
    )
    assert not any(Path(p).exists() for p in orphans)
    assert Path(path(committed)).exists() and Path(path(pending)).exists()


def test_restaging_refreshes_a_placed_video_and_cleanup_spares_pending_ones(env):
    staging, root, build, stage, run = env
    placed = stage("100")
    video = root / placed["videos"]["front_1"]["path"]
    old = video.stat().st_mtime - 30 * 86400
    os.utime(video, (old, old))
    assert commit.orphan_videos(str(root), str(staging)) == []
    (staging / "eva" / "100" / "meta.json").unlink()
    assert commit.orphan_videos(str(root), str(staging)) == [str(video)]

    assert stage("100") == placed
    assert video.stat().st_mtime > old + 86400
    os.utime(video, (old, old))
    commit.cleanup(root, staging, older_than=ZERO)
    assert video.exists()
    rejected = stage("200")
    (root / rejected["videos"]["front_1"]["path"]).write_bytes(b"truncated")
    run()
    commit.cleanup(root, staging, older_than=ZERO, upload_grace=ZERO)
    assert not (root / rejected["videos"]["front_1"]["path"]).exists()
