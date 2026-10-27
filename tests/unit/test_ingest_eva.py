"""EVA uploads through ingest (egomimic/ingest/eva.py) on synthetic demos."""

import json
import shutil
import subprocess

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
lance = pytest.importorskip("lance")

from egomimic.ingest.formats import detect, prepare_delivery  # noqa: E402
from egomimic.ingest.stage import read_pin  # noqa: E402
from egomimic.rldb.lance.schema import recording_id  # noqa: E402
from egomimic.scripts.lance_commit import commit_staged  # noqa: E402

VID = "1766709615098"
N = 12
META = {
    "task_name": "fold_clothes",
    "task_description": "fold the shirt",
    "operator": "op_7",
    "lab": "rl2",
    "scene": "kitchen_A",
    "objects": ["shirt"],
    "embodiment": "eva_bimanual",
    "rig_name": "eva_2",
    "device_serial": "SN0001",
    "license": "CC-BY-4.0",
}
CAMERAS = ("front_img_1", "left_wrist_img", "right_wrist_img")


def _ffmpeg_has(codec: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    out = subprocess.run(
        [ffmpeg or "ffmpeg", "-hide_banner", "-encoders"], capture_output=True
    )
    return codec in out.stdout.decode()


@pytest.fixture
def encoder(monkeypatch):
    """The delivery-spec AV1 encoder, or libx264 without B-frames where ffmpeg has no
    SVT-AV1 (same GOP 2 and faststart, so the committer's index reads it the same)."""
    if _ffmpeg_has("libsvtav1"):
        return
    if not _ffmpeg_has("libx264"):
        pytest.skip("needs ffmpeg with libsvtav1 or libx264")
    import egomimic.scripts.eva_process.eva_to_lance as m

    def x264(path, h, w, fps, ffmpeg):
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(int(fps)), "-i", "-",
            "-c:v", "libx264", "-g", "2", "-bf", "0", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ]  # fmt: skip
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    monkeypatch.setattr(m, "open_av1_encoder", x264)


def upload(tmp_path, meta=META, cameras=CAMERAS, vid=VID):
    d = tmp_path / "delivery"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with h5py.File(d / f"{vid}.hdf5", "w") as f:
        for group in ("observations", "actions"):
            for key in ("eepose", "joints"):
                a = rng.uniform(0.1, 0.5, (N, 14)).astype(np.float32)
                a[:, 6] = a[:, 13] = np.linspace(0.002, 0.09, N)
                f[f"{group}/{key}"] = a
        for cam in cameras:
            f[f"observations/images/{cam}"] = np.zeros((N, 48, 64, 3), np.uint8)
    (d / f"{vid}_metadata.json").write_text(json.dumps(meta))
    return d


def prepare(tmp_path, d):
    return prepare_delivery(
        d, tmp_path / "staged", tmp_path / "lance", fingerprint="f" * 64,
        vendor="rl2-robots", codec="libx264", mano=False,
    )  # fmt: skip


def test_an_eva_upload_is_detected_by_its_files(tmp_path):
    d = upload(tmp_path)
    names = [p.name for p in d.iterdir()]
    assert detect(names).name == "eva"
    assert detect([f"{VID}.hdf5", "1766709615099_metadata.json"]) is None


def test_an_eva_upload_stages_and_commits(tmp_path, encoder):
    p = prepare(tmp_path, upload(tmp_path))
    assert p.status == "staged", (p.problems, p.error)
    assert p.recording_id == recording_id("eva", VID)  # as the backfill names it
    pin = read_pin(p.staged)
    assert (pin["format"], pin["source"], pin["settings"]) == (
        "eva",
        "eva",
        {"fps": 30.0},
    )
    assert pin["report"]["num_frames"] == N and "waived" not in pin["report"]
    assert pin["report"]["videos"] == ["front_1", "left_wrist", "right_wrist"]

    dst = tmp_path / "lance_staging" / pin["source"] / pin["recording_id"]
    shutil.copytree(
        p.staged, dst, ignore=shutil.ignore_patterns("stage.json", "diff.json")
    )
    report = commit_staged(
        str(tmp_path / "lance_staging"), "eva", str(tmp_path / "lance"), tmp_path / "w"
    )
    assert report["rejected"] == {}
    (rec,) = (
        lance.dataset(tmp_path / "lance" / "recordings.lance").to_table().to_pylist()
    )
    assert (rec["lab"], rec["embodiment"], rec["end_effector"]) == (
        "rl2", "eva_bimanual", "gripper",
    )  # fmt: skip
    assert (rec["rig_name"], rec["operator"], rec["license"]) == (
        "eva_2",
        "op_7",
        "CC-BY-4.0",
    )
    (task,) = lance.dataset(tmp_path / "lance" / "tasks.lance").to_table().to_pylist()
    assert task["task_name"] == "fold_clothes" and task["objects"] == ["shirt"]


def test_the_rigs_legacy_keys_are_read(tmp_path, encoder):
    legacy = {k: v for k, v in META.items() if k not in ("task_name", "embodiment")}
    legacy |= {"task": "fold_clothes", "robot_name": "eva_bimanual"}
    p = prepare(tmp_path, upload(tmp_path, meta=legacy))
    assert p.status == "staged", (p.problems, p.error)


def test_a_waiver_goes_to_review(tmp_path, encoder):
    meta = {k: v for k, v in META.items() if k != "scene"} | {
        "missing_metadata": "scene"
    }
    p = prepare(tmp_path, upload(tmp_path, meta=meta))
    assert p.status == "staged"
    assert p.report["waived"] == ["scene: marked as not available (MissingMetadata)"]


def test_missing_metadata_is_rejected_with_reasons(tmp_path):
    meta = {k: v for k, v in META.items() if k not in ("operator", "device_serial")}
    p = prepare(tmp_path, upload(tmp_path, meta=meta))
    assert p.status == "rejected"
    assert any("operator" in m for m in p.problems)
    assert "device_serial is required, got None" in p.problems
    assert not (tmp_path / "staged").exists()


def test_a_missing_wrist_camera_is_rejected_before_converting(tmp_path):
    p = prepare(tmp_path, upload(tmp_path, cameras=("front_img_1", "left_wrist_img")))
    assert p.status == "rejected"
    assert p.problems == [
        f"{VID}.hdf5 has no right_wrist_img: a eva_bimanual demo needs "
        "front_img_1, left_wrist_img, right_wrist_img"
    ]


def test_a_one_arm_robot_needs_only_its_wrist(tmp_path, encoder):
    meta = META | {"embodiment": "eva_right_arm"}
    p = prepare(
        tmp_path,
        upload(tmp_path, meta=meta, cameras=("front_img_1", "right_wrist_img")),
    )
    assert p.status == "staged", (p.problems, p.error)


def test_an_id_that_is_not_epoch_ms_is_rejected(tmp_path):
    p = prepare(tmp_path, upload(tmp_path, vid="17667096"))  # seconds, not ms
    assert p.status == "rejected" and "epoch milliseconds" in p.problems[0]


def test_an_eva_upload_goes_through_the_watcher_review_and_committer(tmp_path, encoder):
    from egomimic.ingest.launch import LocalLauncher
    from egomimic.ingest.run import commit_round
    from egomimic.ingest.store import LocalStore, Ref
    from egomimic.ingest.watch import run_pass

    store = LocalStore(tmp_path / "buckets")
    ref = Ref("rl2-robots", "demo-1")
    store.upload_delivery(ref, upload(tmp_path), {"by": "test"})
    launcher = LocalLauncher(
        store, str(tmp_path / "lance"), tmp_path / "jobs", codec="libx264", mano=False
    )

    def run():
        return run_pass(
            store, launcher, tmp_path / "work", raw_root=str(tmp_path / "raw"),
            lance_staging=str(tmp_path / "staging"), revision="r1",
        )  # fmt: skip

    assert run().staged == ["rl2-robots/demo-1"]
    st = store.get_status(ref)
    assert (st["state"], st["recording_id"]) == ("ready", recording_id("eva", VID))
    store.add_decision(ref, "approve", "simar", stage=st["stage"])
    run()  # hand-off to <staging>/eva/<recording id>/
    commit_round(str(tmp_path / "staging"), str(tmp_path / "lance"), tmp_path / "work")
    assert run().committed == ["rl2-robots/demo-1"]
    (rec,) = (
        lance.dataset(tmp_path / "lance" / "recordings.lance").to_table().to_pylist()
    )
    assert rec["recording_id"] == recording_id("eva", VID)
