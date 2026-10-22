import json

import pytest

lance = pytest.importorskip("lance")
pytest.importorskip("av")

from fixtures.lance_tables import EVA_DELIVERY_FRAMES, eva_delivery  # noqa: E402

from egomimic.rldb.lance import commit  # noqa: E402
from egomimic.rldb.lance.dataset import LanceEpisodeResolver  # noqa: E402
from egomimic.rldb.lance.schema import recording_id, steps_path  # noqa: E402
from egomimic.scripts.lance_commit import REJECTED, commit_staged  # noqa: E402

EVA_STEPS = steps_path("rl2", "eva_bimanual", "parallel_jaw_gripper")

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
    assert {EVA_STEPS, "recordings.lance", "changes.lance"} <= set(batch["versions"])
    (row,) = _changes(root)
    assert row["tag"] == batch["tag"] and sorted(row["added"]) == sorted(batch["added"])
    assert row["source"] == "eva" and row["steps_tables"] == [EVA_STEPS]
    assert commit.list_files(staging / "eva") == {}
    leaves = LanceEpisodeResolver(
        root, key_map={}, cameras={}, tag=batch["tag"]
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


@pytest.mark.parametrize("kind", ["sql", "socket"])
def test_an_infrastructure_error_aborts_the_run_without_rejecting(
    env, monkeypatch, kind
):
    from sqlalchemy.exc import OperationalError

    from egomimic.scripts import lance_commit

    staging, root, stage, run = env
    stage("100")
    err = (
        OperationalError("select", {}, Exception("down"))
        if kind == "sql"
        else ConnectionError("database unreachable")
    )

    def down(dirs):
        raise err

    monkeypatch.setitem(lance_commit.LOADERS, "eva", down)
    with pytest.raises(type(err)):
        run()
    assert not (staging / "eva" / "100" / REJECTED).exists()


def test_a_missing_sql_row_is_retried_on_the_next_run(env, monkeypatch):
    from egomimic.scripts import lance_commit

    staging, root, stage, run = env
    stage("100")
    load = lance_commit.LOADERS["eva"]
    monkeypatch.setitem(
        lance_commit.LOADERS, "eva", lambda dirs: ([], {"100": ["no SQL row"]})
    )
    assert run()["rejected"] == {"100": ["no SQL row"]}
    assert (staging / "eva" / "100" / REJECTED).exists()
    monkeypatch.setitem(lance_commit.LOADERS, "eva", load)
    (batch,) = run()["batches"]
    assert batch["added"] == [recording_id("eva", "100")]


def test_a_delivery_staged_under_another_name_is_rejected(env, tmp_path):
    staging, root, stage, run = env
    stage("100")
    commit.stage(
        eva_delivery(tmp_path / "x", "200"), commit.join(staging, "eva", "300")
    )
    report = run()
    assert [b["added"] for b in report["batches"]] == [[recording_id("eva", "100")]]
    assert "!= staged directory 300" in report["rejected"]["300"][0]


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
    rows = lance.dataset(root / EVA_STEPS).to_table().to_pylist()
    assert len(rows) == N and rows[0]["left"]["obs_ee_pose"][0] == 50.5


def test_delivery_provenance_is_kept_in_changes(env, tmp_path):
    staging, root, stage, run = env
    src = eva_delivery(tmp_path / "prov", "100")
    meta = json.loads((src / "meta.json").read_text())
    meta["provenance"] = {"converter": "abc123"}
    (src / "meta.json").write_text(json.dumps(meta))
    commit.stage(src, commit.join(staging, "eva", "100"))
    stage("200")
    run()
    (row,) = _changes(root)
    assert json.loads(row["provenance"]) == {
        recording_id("eva", "100"): {"converter": "abc123"}
    }
