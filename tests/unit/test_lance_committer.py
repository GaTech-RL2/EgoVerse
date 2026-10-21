import json

import pytest

lance = pytest.importorskip("lance")
pytest.importorskip("av")

from fixtures.lance_tables import EVA_DELIVERY_FRAMES, eva_delivery  # noqa: E402

from egomimic.rldb.lance import commit  # noqa: E402
from egomimic.rldb.lance.dataset import LanceEpisodeResolver  # noqa: E402
from egomimic.rldb.lance.schema import recording_id  # noqa: E402
from egomimic.scripts.lance_commit import REJECTED, commit_staged  # noqa: E402

N = EVA_DELIVERY_FRAMES


@pytest.fixture
def env(tmp_path):
    staging, root = tmp_path / "staging", tmp_path / "lance"
    built = iter(range(100))

    def stage(vid, **kw):
        src = eva_delivery(tmp_path / f"built{next(built)}", vid, **kw)
        commit.stage(src, commit.join(staging, "eva", vid))

    def run(**kw):
        return commit_staged(str(staging), "eva", str(root), tmp_path / "work", **kw)

    return staging, root, stage, run


def _changes(root):
    return lance.dataset(root / "changes.lance").to_table().to_pylist()


def test_staged_deliveries_commit_as_one_tagged_batch(env):
    staging, root, stage, run = env
    stage("100")
    stage("200")
    report = run()
    (batch,) = report["batches"]
    assert sorted(batch["added"]) == sorted(
        recording_id("eva", v) for v in ("100", "200")
    )
    assert batch["tag"].startswith("eva-")
    assert {"steps/eva.lance", "recordings.lance", "changes.lance"} <= set(
        batch["versions"]
    )
    (row,) = _changes(root)
    assert row["tag"] == batch["tag"] and sorted(row["added"]) == sorted(batch["added"])
    assert commit.list_files(staging / "eva") == {}
    leaves = LanceEpisodeResolver(
        root, lab="eva", key_map={}, cameras={}, tag=batch["tag"]
    ).resolve()
    assert len(leaves) == 2
    assert run() == {"batches": [], "rejected": {}}


def test_a_bad_delivery_is_skipped_until_its_files_change(env):
    staging, root, stage, run = env
    stage("100")
    stage("200", frames_in_video=N - 2)
    report = run()
    assert [b["added"] for b in report["batches"]] == [[recording_id("eva", "100")]]
    assert "front_1.mp4 has 4 frames" in report["rejected"]["200"][0]
    note = json.loads((staging / "eva" / "200" / REJECTED).read_text())
    assert note["errors"] == report["rejected"]["200"]

    assert run() == {"batches": [], "rejected": {}}
    assert len(_changes(root)) == 1

    commit.remove([str(staging / "eva" / "200" / "front_1.mp4")])
    stage("200")
    report = run()
    assert [b["added"] for b in report["batches"]] == [[recording_id("eva", "200")]]
    assert not (staging / "eva" / "200").exists() or not any(
        (staging / "eva" / "200").iterdir()
    )


def test_restaging_a_committed_recording_needs_replace(env, tmp_path):
    staging, root, stage, run = env
    stage("100")
    run()
    commit.stage(
        eva_delivery(tmp_path / "v2", "100", offset=50),
        commit.join(staging, "eva", "100"),
    )
    report = run()
    assert (
        report["batches"] == [] and "already committed" in report["rejected"]["100"][0]
    )
    report = run(replace=True)
    assert [b["replaced"] for b in report["batches"]] == [[recording_id("eva", "100")]]
    rows = lance.dataset(root / "steps" / "eva.lance").to_table().to_pylist()
    assert len(rows) == N and rows[0]["left"]["obs_ee_pose"][0] == 50.5
