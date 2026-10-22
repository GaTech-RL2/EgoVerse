"""Staging (egomimic/ingest/stage.py): convert now, commit exactly that result later."""

import shutil
import subprocess

import pytest
from fixtures.mcap_episodes import write_mcap_episode

lance = pytest.importorskip("lance")
if not (
    shutil.which("ffmpeg")
    and "libx264"
    in subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True
    ).stdout.decode()
):
    pytest.skip("needs ffmpeg with libx264", allow_module_level=True)

from egomimic.ingest.stage import (  # noqa: E402
    StageCorrupt,
    StaleStage,
    commit_staged,
    committed_state,
    load,
    prepare,
    read_pin,
    stage_digest,
    table_hashes,
)
from egomimic.rldb.lance import commit  # noqa: E402
from egomimic.scripts.mcap_process.mcap_to_lance import (  # noqa: E402
    convert_and_commit,
)

OPTS = dict(codec="libx264", mano=False)


def stage(tmp_path, name="s", fingerprint="fp1", root="lance", **kw):
    ep = write_mcap_episode(tmp_path / f"eps_{name}", **kw)
    return prepare(
        ep, tmp_path / "staged" / name, tmp_path / root, fingerprint=fingerprint, **OPTS
    )


def approved(staged, root, **kw):
    """Commit a stage with its own approval (what the approve button will pass)."""
    return commit_staged(staged, root, approved_stage=stage_digest(staged), **kw)


def rows(root, table, drop=()):
    t = lance.dataset(root / table).to_table().to_pylist()
    return [{k: v for k, v in r.items() if k not in drop} for r in t]


def test_prepare_then_commit_matches_convert_and_commit(tmp_path):
    p = stage(tmp_path)
    assert p.status == "staged"
    assert not (tmp_path / "lance").exists()  # preparing writes nothing to Lance
    out = approved(p.staged, tmp_path / "lance", fingerprint="fp1")
    assert out["status"] == "committed" and out["annotations_added"] == 2
    assert {"recordings.lance", "steps/aria.lance"} <= set(out["versions"])

    ep = write_mcap_episode(tmp_path / "eps_direct")
    assert convert_and_commit(ep, tmp_path / "direct", **OPTS).status == "committed"
    a, b = tmp_path / "lance", tmp_path / "direct"
    video = ("video_path", "content_digest")  # separate encodes of the same frames
    assert rows(a, "recordings.lance", video) == rows(b, "recordings.lance", video)
    for table in (
        "steps/aria.lance",
        "tasks.lance",
        "aliases.lance",
        "annotations.lance",
    ):
        assert rows(a, table) == rows(b, table), table


def test_rejected_and_failed_deliveries_stage_nothing(tmp_path):
    p = stage(tmp_path, "bad", imu_hz=100)
    assert p.status == "rejected" and p.problems == ["[imu] /imu: 100 Hz, need >= 200"]
    p = stage(tmp_path, "robot", embodiment="eva_bimanual")
    assert p.status == "failed" and "no robot steps schema" in p.error
    assert sorted(x.name for x in (tmp_path / "staged").iterdir()) == []


def test_the_pin_records_inputs_converter_and_every_file(tmp_path):
    p = stage(tmp_path)
    pin = read_pin(p.staged)
    assert pin["fingerprint"] == "fp1" and pin["format"] == "mcap"
    assert pin["settings"] == {"mano": False, "mano_iters": 2400, "codec": "libx264"}
    assert len(pin["converter"]["revision"]) == 40  # this checkout's commit
    assert {
        "steps.parquet",
        "rows.json",
        "videos/front_1.mp4",
        "videos/front_1_480.mp4",
    } <= set(pin["files"])
    assert "diff.json" in pin["files"]  # the reviewed diff is pinned too
    empty = table_hashes(committed_state(tmp_path / "lance", p.recording_id, "aria"))
    assert pin["base_state"] == empty
    assert set(pin["staged_state"]) == {
        "steps",
        "recording",
        "alias",
        "tasks",
        "annotations",
    }
    assert all(pin["staged_state"][k] != empty[k] for k in empty)
    assert pin["report"]["num_frames"] == 60 and pin["report"]["labels"] == 2
    c = load(p.staged)
    assert c.delivery.recording_id == p.recording_id and len(c.labels) == 2


@pytest.mark.parametrize("how", ["edit", "extra", "delete", "diff"])
def test_a_changed_stage_refuses_to_commit(tmp_path, how):
    p = stage(tmp_path)
    if how == "edit":
        (p.staged / "videos" / "front_1.mp4").write_bytes(b"not the staged video")
    elif how == "extra":
        (p.staged / "videos" / "smuggled.mp4").write_bytes(b"x")
    elif how == "delete":
        (p.staged / "rows.json").unlink()
    else:  # what the reviewer saw is part of what gets committed
        (p.staged / "diff.json").write_text('{"action": "new"}')
    with pytest.raises(StageCorrupt):
        approved(p.staged, tmp_path / "lance")
    assert not (tmp_path / "lance" / "recordings.lance").exists()


def test_a_newer_upload_makes_the_stage_stale(tmp_path):
    p = stage(tmp_path)
    with pytest.raises(StaleStage):
        approved(p.staged, tmp_path / "lance", fingerprint="fp2")


def test_lance_changing_since_staging_makes_the_stage_stale(tmp_path):
    a = stage(tmp_path, "a")
    b = stage(tmp_path, "b", fingerprint="fp2", drop_video_frame=11)  # same recording
    assert (
        a.recording_id == b.recording_id
        and a.diff["action"] == b.diff["action"] == "new"
    )
    approved(a.staged, tmp_path / "lance")
    with pytest.raises(StaleStage):  # b was diffed against a Lance without a
        approved(b.staged, tmp_path / "lance")
    b = stage(tmp_path, "b", fingerprint="fp2", drop_video_frame=11)  # re-stage
    assert b.diff["action"] == "replace"
    assert approved(b.staged, tmp_path / "lance")["status"] == "replaced"
    assert len(rows(tmp_path / "lance", "recordings.lance")) == 1


def test_committing_twice_is_a_no_op(tmp_path):
    p = stage(tmp_path)
    first = approved(p.staged, tmp_path / "lance")
    again = approved(p.staged, tmp_path / "lance")
    assert again["status"] == "unchanged" and again["annotations_added"] == 0
    assert again["versions"] == first["versions"]  # nothing was written


def test_diff_of_a_new_recording(tmp_path):
    d = stage(tmp_path).diff
    assert d["action"] == "new"
    assert d["frames"] == {"new": 60, "old": None}
    assert d["videos"] == {"new": ["front_1", "front_1_480"], "old": []}
    assert d["tasks"] == {
        "added": ["fold_clothes", "pack_clothes"],
        "removed": [],
        "changed": [],
    }
    assert d["labels"] == {
        "added": ["drop it in the basket", "pick up the shirt"],
        "removed": [],
    }
    assert d["valid_frac"]["old"] is None


def test_diff_of_a_labels_only_correction(tmp_path):
    approved(stage(tmp_path, "v1").staged, tmp_path / "lance")

    def reword(topic, msg, t):
        if topic == "/subtask-annotation" and msg.data == "pick up the shirt":
            msg.data = "pick up the red shirt"
            return msg, t

    d = stage(tmp_path, "v2", fingerprint="fp2", tamper=reword).diff
    assert d["action"] == "replace" and d["frames"] == {"new": 60, "old": 60}
    assert d["tasks"] == {"added": [], "removed": [], "changed": []}
    assert d["labels"] == {
        "added": ["pick up the red shirt"],
        "removed": ["pick up the shirt"],
    }
    assert d["valid_frac"]["old"] == {"head": 1.0, "left": 1.0, "right": 1.0}


def test_restaging_replaces_the_folder_entirely(tmp_path):
    p = stage(tmp_path, "s", slam_camera=True)
    assert (p.staged / "videos" / "slam_left.mp4").exists()
    p = stage(tmp_path, "s")  # the second camera is gone in the new version
    assert sorted(x.name for x in (p.staged / "videos").iterdir()) == [
        "front_1.mp4",
        "front_1_480.mp4",
    ]
    load(p.staged)  # consistent with its own pin
    assert sorted(x.name for x in (tmp_path / "staged").iterdir()) == ["s"]  # no temps


def test_a_task_only_correction_is_committed_not_skipped(tmp_path):
    approved(stage(tmp_path, "v1").staged, tmp_path / "lance")

    def redescribe(topic, msg, t):
        if topic == "/task" and msg.name == "pack_clothes":
            msg.description = "pack the folded shirt into the basket"
            return msg, t

    v2 = stage(tmp_path, "v2", fingerprint="fp2", tamper=redescribe)
    assert v2.diff["tasks"]["changed"] == ["pack_clothes"]
    assert (
        read_pin(v2.staged)["content_digest"]
        == read_pin(tmp_path / "staged" / "v1")["content_digest"]
    )  # same steps and videos
    assert approved(v2.staged, tmp_path / "lance")["status"] == "replaced"
    tasks = {t["task_name"]: t for t in rows(tmp_path / "lance", "tasks.lance")}
    assert (
        tasks["pack_clothes"]["task_description"]
        == "pack the folded shirt into the basket"
    )


def test_a_metadata_only_correction_is_committed(tmp_path):
    approved(stage(tmp_path, "v1").staged, tmp_path / "lance")
    v2 = stage(
        tmp_path, "v2", fingerprint="fp2", metadata_override={"scene": "kitchen_b"}
    )
    assert approved(v2.staged, tmp_path / "lance")["status"] == "replaced"
    (rec,) = rows(tmp_path / "lance", "recordings.lance")
    assert rec["scene"] == "kitchen_b"


def test_a_task_edited_in_lance_after_staging_makes_the_stage_stale(tmp_path):
    root = tmp_path / "lance"
    approved(stage(tmp_path, "v1").staged, root)
    v2 = stage(
        tmp_path, "v2", fingerprint="fp2", metadata_override={"scene": "kitchen_b"}
    )
    tid = rows(root, "tasks.lance")[0]["task_instance_id"]
    import pyarrow as pa

    commit.edit(  # someone fixes a task description directly; same task id
        root,
        "tasks.lance",
        pa.table({"task_instance_id": [tid], "task_description": ["edited in Lance"]}),
        on="task_instance_id",
    )
    with pytest.raises(StaleStage):
        approved(v2.staged, root)


def test_a_label_added_after_staging_makes_the_stage_stale(tmp_path):
    from egomimic.rldb.lance.schema import ANNOTATIONS, annotation_id

    root = tmp_path / "lance"
    v1 = stage(tmp_path, "v1")
    approved(v1.staged, root)
    v2 = stage(
        tmp_path, "v2", fingerprint="fp2", metadata_override={"scene": "kitchen_b"}
    )
    task = rows(root, "tasks.lance")[0]
    row = dict(
        recording_id=v1.recording_id,
        t_start_ns=task["t_start_ns"],
        t_end_ns=task["t_start_ns"] + 10**8,
        annotation_type="language",
        text="added by a grader after staging",
        annotation_vendor="scale",
        quality_score=None,
        graded_by=None,
        rubric_version=None,
    )
    row["annotation_id"] = annotation_id(
        row["recording_id"],
        "scale",
        "language",
        row["t_start_ns"],
        row["t_end_ns"],
        row["text"],
    )
    import pyarrow as pa

    commit.apply_annotations(root, pa.Table.from_pylist([row], schema=ANNOTATIONS), [])
    with pytest.raises(StaleStage):
        approved(v2.staged, root)


def test_an_approval_applies_to_exactly_the_reviewed_stage(tmp_path):
    p = stage(tmp_path)
    with pytest.raises(StaleStage):
        commit_staged(p.staged, approved_stage="0" * 64, lance_root=tmp_path / "lance")
    out = approved(p.staged, tmp_path / "lance")
    assert out["status"] == "committed"


def test_committing_needs_an_approval(tmp_path):
    p = stage(tmp_path)
    with pytest.raises(TypeError):
        commit_staged(p.staged, tmp_path / "lance")  # no approved_stage: not possible


class Crash(Exception):
    pass


def crash_after(monkeypatch, n: int):
    """Make the n-th Lance table write (append or delete+append) succeed and then crash
    the process, as a kill between Ryan's per-table commits would."""
    count = {"writes": 0}
    for name in ("_append", "_replace"):
        real = getattr(commit, name)

        def wrapped(*a, _real=real, **kw):
            _real(*a, **kw)
            count["writes"] += 1
            if count["writes"] == n:
                raise Crash(f"killed after write {n}")

        monkeypatch.setattr(commit, name, wrapped)
    return count


def check_finished(root, p):
    """Lance holds exactly the staged result: no duplicated or orphan rows."""
    pin = read_pin(p.staged)
    assert (
        table_hashes(committed_state(root, p.recording_id, "aria"))
        == pin["staged_state"]
    )
    assert len(rows(root, "steps/aria.lance")) == pin["report"]["num_frames"]
    assert len(rows(root, "recordings.lance")) == 1


# A new recording writes: steps, aliases, tasks, recordings (published), the annotations
# table, then the labels. A crash after any of them must be finishable by a retry.
@pytest.mark.parametrize("crash_at", [1, 2, 3, 4, 5, 6])
def test_a_new_recording_cut_off_between_tables_is_finished_on_retry(
    tmp_path, monkeypatch, crash_at
):
    root = tmp_path / "lance"
    p = stage(tmp_path)
    crash_after(monkeypatch, crash_at)
    with pytest.raises(Crash):
        approved(p.staged, root)
    monkeypatch.undo()
    out = approved(p.staged, root)
    assert out["status"] == ("unchanged" if crash_at == 6 else "resumed")
    check_finished(root, p)


# A replacement writes: steps, aliases, tasks, recordings, then the labels.
@pytest.mark.parametrize("crash_at", [1, 2, 3, 4, 5])
def test_a_replacement_cut_off_between_tables_is_finished_on_retry(
    tmp_path, monkeypatch, crash_at
):
    root = tmp_path / "lance"
    approved(stage(tmp_path, "v1").staged, root)

    def reword(topic, msg, t):
        if topic == "/subtask-annotation" and msg.data == "pick up the shirt":
            msg.data = "pick up the red shirt"
            return msg, t

    v2 = stage(
        tmp_path, "v2", fingerprint="fp2", tamper=reword, drop_video_frame=11,
        metadata_override={"scene": "kitchen_b"},
    )  # fmt: skip
    crash_after(monkeypatch, crash_at)
    with pytest.raises(Crash):
        approved(v2.staged, root)
    monkeypatch.undo()
    out = approved(v2.staged, root)
    assert out["status"] == ("unchanged" if crash_at == 5 else "resumed")
    check_finished(root, v2)
    texts = sorted(a["text"] for a in rows(root, "annotations.lance"))
    assert texts == ["drop it in the basket", "pick up the red shirt"]


def test_a_deployed_copy_pins_its_revision_file(tmp_path, monkeypatch):
    """deploy.sh copies the code without .git and writes egomimic/ingest/REVISION."""
    import egomimic.ingest.stage as st

    copy = tmp_path / "code" / "egomimic" / "ingest"
    copy.mkdir(parents=True)
    (copy / "REVISION").write_text("abc123\n")
    monkeypatch.setattr(st, "__file__", str(copy / "stage.py"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))  # no repo above
    assert st._git() == {"revision": "abc123", "dirty": False}
