"""Staging (egomimic/ingest/stage.py): convert now into a staged delivery the committer
takes as it is, pinned so an approval applies to exactly that result."""

import hashlib
import json
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
    REVIEW_ONLY,
    StageCorrupt,
    prepare,
    read_pin,
    stage_digest,
    verify,
)
from egomimic.rldb.lance.schema import steps_path  # noqa: E402
from egomimic.scripts.lance_commit import commit_staged  # noqa: E402

OPTS = dict(codec="libx264", mano=False)


def stage(tmp_path, name="s", fingerprint="f" * 64, **kw):
    ep = write_mcap_episode(tmp_path / f"eps_{name}", **kw)
    return prepare(
        ep,
        tmp_path / "staged" / name,
        tmp_path / "lance",
        fingerprint=fingerprint,
        **OPTS,
    )


def land(tmp_path, staged, replace=False):
    """Hand a stage to the committer as the watcher does, and commit it."""
    pin = read_pin(staged)
    dst = tmp_path / "lance_staging" / pin["source"] / pin["vendor_recording_id"]
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(staged.iterdir(), key=lambda p: p.name == "meta.json"):
        if p.name not in REVIEW_ONLY:
            shutil.copyfile(p, dst / p.name)
    return commit_staged(
        str(tmp_path / "lance_staging"),
        pin["source"],
        str(tmp_path / "lance"),
        tmp_path / "work",
        replace=replace,
    )


def rows(tmp_path, table):
    return lance.dataset(tmp_path / "lance" / table).to_table().to_pylist()


def test_a_stage_is_a_delivery_the_committer_takes_as_it_is(tmp_path):
    p = stage(tmp_path)
    assert p.status == "staged"
    assert not (tmp_path / "lance").exists()  # preparing writes nothing to Lance
    report = land(tmp_path, p.staged)
    assert [b["added"] for b in report["batches"]][0] == [p.recording_id]
    (rec,) = rows(tmp_path, "recordings.lance")
    assert rec["recording_id"] == p.recording_id and rec["num_frames"] == 60
    assert len(rows(tmp_path, "tasks.lance")) == 2
    assert len(rows(tmp_path, "annotations.lance")) == 2
    assert len(rows(tmp_path, steps_path(rec["lab"], rec["embodiment"], "hand"))) == 60


def test_rejected_and_failed_deliveries_stage_nothing(tmp_path):
    p = stage(tmp_path, "bad", imu_hz=100)
    assert p.status == "rejected" and p.problems == ["[imu] /imu: 100 Hz, need >= 200"]
    p = stage(tmp_path, "robot", embodiment="eva_bimanual")
    assert p.status == "failed" and "no robot steps schema" in p.error
    assert sorted(x.name for x in (tmp_path / "staged").iterdir()) == []


def test_the_pin_records_inputs_converter_and_every_file(tmp_path):
    p = stage(tmp_path)
    pin = read_pin(p.staged)
    assert pin["fingerprint"] == "f" * 64 and pin["format"] == "mcap"
    assert pin["settings"] == {"mano": False, "mano_iters": 2400, "codec": "libx264"}
    assert len(pin["converter"]["revision"]) == 40  # this checkout's commit
    assert set(pin["files"]) == {
        "steps.parquet",
        "recordings.parquet",
        "tasks.parquet",
        "aliases.parquet",
        "labels.parquet",
        "front_1.mp4",
        "front_1_480.mp4",
        "meta.json",
    }
    f = pin["files"]["front_1.mp4"]
    data = (p.staged / "front_1.mp4").read_bytes()
    assert f == {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    assert pin["report"]["num_frames"] == 60 and pin["report"]["labels"] == 2
    meta = json.loads((p.staged / "meta.json").read_text())
    assert meta["format"] == "delivery" and meta["label_vendors"] == ["acme", "rl2"]
    assert meta["provenance"]["fingerprint"] == "f" * 64
    assert (
        stage_digest(p.staged)
        == hashlib.sha256((p.staged / "stage.json").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("how", ["resize", "extra", "missing"])
def test_a_stored_stage_must_match_its_pin(tmp_path, how):
    p = stage(tmp_path)
    pin = read_pin(p.staged)
    sizes = {n: (p.staged / n).stat().st_size for n in pin["files"]}
    verify(pin, {**sizes, "stage.json": 1, "diff.json": 1})
    if how == "resize":
        sizes["front_1.mp4"] += 1
    elif how == "extra":
        sizes["evil.mp4"] = 3
    else:
        del sizes["labels.parquet"]
    with pytest.raises(StageCorrupt):
        verify(pin, sizes)


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
    land(tmp_path, stage(tmp_path, "v1").staged)

    def reword(topic, msg, t):
        if topic == "/subtask-annotation" and msg.data == "pick up the shirt":
            msg.data = "pick up the red shirt"
            return msg, t

    p = stage(tmp_path, "v2", fingerprint="e" * 64, tamper=reword)
    d = p.diff
    assert d["action"] == "replace" and d["frames"] == {"new": 60, "old": 60}
    assert d["tasks"] == {"added": [], "removed": [], "changed": []}
    assert d["labels"] == {
        "added": ["pick up the red shirt"],
        "removed": ["pick up the shirt"],
    }
    assert d["valid_frac"]["old"] == {"head": 1.0, "left": 1.0, "right": 1.0}
    land(tmp_path, p.staged, replace=True)
    texts = {a["text"] for a in rows(tmp_path, "annotations.lance")}
    assert texts == {"pick up the red shirt", "drop it in the basket"}


def test_restaging_replaces_the_folder_entirely(tmp_path):
    p = stage(tmp_path, "s", slam_camera=True)
    assert (p.staged / "slam_left.mp4").exists()
    p = stage(tmp_path, "s")  # the second camera is gone in the new version
    assert not (p.staged / "slam_left.mp4").exists()
    pin = read_pin(p.staged)
    verify(pin, {x.name: x.stat().st_size for x in p.staged.iterdir()})
    assert sorted(x.name for x in (tmp_path / "staged").iterdir()) == ["s"]  # no temps


def test_a_deployed_copy_pins_its_revision_file(tmp_path, monkeypatch):
    """The converter image copies the code without .git and writes REVISION."""
    import egomimic.ingest.stage as st

    copy = tmp_path / "code" / "egomimic" / "ingest"
    copy.mkdir(parents=True)
    (copy / "REVISION").write_text("abc123\n")
    monkeypatch.setattr(st, "__file__", str(copy / "stage.py"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))  # no repo above
    assert st.converter_revision() == {"revision": "abc123", "dirty": False}
