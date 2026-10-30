import json
from datetime import datetime, timezone

import numpy as np
import pytest

lance = pytest.importorskip("lance")
av = pytest.importorskip("av")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

from egomimic.rldb.embodiment.end_effector import END_EFFECTOR  # noqa: E402
from egomimic.rldb.lance import commit  # noqa: E402
from egomimic.rldb.lance.dataset import LanceEpisodeResolver  # noqa: E402
from egomimic.rldb.lance.metadata import read_manifest  # noqa: E402
from egomimic.rldb.lance.schema import (  # noqa: E402
    DEXMATE_STEPS,
    EVA_STEPS,
    recording_id,
    steps_path,
    zarr_columns,
)
from egomimic.rldb.zarr.required_metadata import MissingMetadata  # noqa: E402
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset  # noqa: E402
from egomimic.rldb.zarr.zarr_writer import ZarrWriter  # noqa: E402
from egomimic.scripts import zarr_to_lance  # noqa: E402
from egomimic.scripts.lance_commit import commit_staged  # noqa: E402

HASH = "2025-11-14-17-16-59-204000"
N = 12
H, W = 48, 64
META = {"lab": "rl2", "operator": "op", "scene": "3"}


class _PyAVEncoder:
    """Stands in for the ffmpeg subprocess of open_av1_encoder (stdin.write/close, wait)."""

    def __init__(self, path, h, w, fps, ffmpeg):
        self.out = av.open(str(path), "w", options={"movflags": "+faststart"})
        self.s = self.out.add_stream("libaom-av1", rate=int(fps))
        self.s.width, self.s.height, self.s.pix_fmt = w, h, "yuv420p"
        self.s.options = {"cpu-used": "8", "crf": "30"}
        self.s.codec_context.gop_size = 2
        self.h, self.w, self.stdin = h, w, self

    def write(self, data):
        img = np.frombuffer(data, np.uint8).reshape(self.h, self.w, 3)
        for p in self.s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            self.out.mux(p)

    def close(self):
        for p in self.s.encode():
            self.out.mux(p)
        self.out.close()

    def wait(self):
        return 0


@pytest.fixture(autouse=True)
def _encoder(monkeypatch):
    monkeypatch.setattr(zarr_to_lance, "open_av1_encoder", _PyAVEncoder)
    # Tiny frames: a 24 px stream plus the full-size front_1_48, as front_1 / front_1_480.
    monkeypatch.setattr(zarr_to_lance, "SHORT_SIDE", 24)


# A microagi v2 episode: per-frame times, sentinels and dropped frames.
MICROAGI = {"episode_id": "e", "source_mcap": "m.mcap"}


def _human_zarr(
    tmp_path, embodiment="human_bimanual", attrs=MICROAGI, annotations=None
):
    rng = np.random.default_rng(0)
    num = {
        f"{s}.{k}": rng.normal(size=(N, d))
        for s in ("left", "right")
        for k, d in (
            ("obs_ee_pose", 7),
            ("obs_wrist_pose", 7),
            ("obs_keypoints", 63),
            ("obs_aria_keypoints", 63),
        )
    }
    num["obs_head_pose"] = rng.normal(size=(N, 7))
    num["obs_eye_gaze"] = rng.normal(size=(N, 3))
    num["obs_eye_gaze"][5] = 1e9
    num["right.obs_keypoints"][4] = 1e9
    # Frames 6-7 of the raw stream were dropped by the zarr converter.
    ts = 1_000_000_000 + np.r_[np.arange(6), np.arange(8, N + 2)] * 33_333_333
    num["obs_rgb_timestamps_ns"] = ts.astype(np.int64)
    imgs = np.zeros((N, H, W, 3), np.uint8)
    for i in range(N):
        imgs[i, :, : 4 * (i + 1)] = 255
    k = np.array([[50.0, 0, 32, 0], [0, 50.0, 24, 0], [0, 0, 1, 0]])
    path = tmp_path / "zarr" / f"{HASH}.zarr"
    ZarrWriter.create_and_write(
        path,
        numeric_data=num,
        image_data={"images.front_1": imgs},
        embodiment=embodiment,
        task_name="fold_clothes",
        task_description="folding clothes",
        operator="op",
        lab="rl2",
        scene=MissingMetadata(),
        objects=MissingMetadata(),
        vendor_recording_id=HASH,
        chunk_timesteps=5,
        intrinsics={"front_1": k},
        annotations=annotations,
        metadata_override=attrs or None,
    )
    return path, num


def test_human_zarr_commits_its_zarr_values(tmp_path):
    path, num = _human_zarr(tmp_path)
    staging, root = tmp_path / "staging", tmp_path / "lance"
    d = zarr_to_lance.convert_zarr_episode(path, tmp_path / "out", META)
    commit.stage(d, commit.join(staging, "zarr", d.name))
    report = commit_staged(str(staging), "zarr", str(root), tmp_path / "work")
    assert report["rejected"] == {}

    rid = recording_id("zarr", HASH)
    assert d.name == HASH and report["batches"][0]["added"] == [rid]
    (rec,) = lance.dataset(root / "recordings.lance").to_table().to_pylist()
    assert (rec["lab"], rec["operator"], rec["scene"]) == ("rl2", "op", "3")
    assert (
        rec["end_effector"] == END_EFFECTOR.HUMAN_HAND.value and rec["num_frames"] == N
    )
    assert dict(rec["video_path"]).keys() == {"front_1", "front_1_48"}
    intr = dict(rec["intrinsics"])
    assert intr["front_1"][0] == pytest.approx(25.0)
    assert intr["front_1_48"][0] == pytest.approx(50.0)
    (task,) = lance.dataset(root / "tasks.lance").to_table().to_pylist()
    assert task["task_name"] == "fold_clothes" and task["objects"] == []
    (alias,) = lance.dataset(root / "aliases.lance").to_table().to_pylist()
    assert alias["episode_hash"] == HASH

    steps = lance.dataset(
        root / steps_path("rl2", "human_bimanual", END_EFFECTOR.HUMAN_HAND.value)
    ).to_table()
    np.testing.assert_array_equal(
        steps.column("timestamp_ns").to_numpy(), num["obs_rgb_timestamps_ns"]
    )
    right_valid = np.ones(N, bool)
    right_valid[4] = False
    assert steps.column("left_valid").to_pylist() == [True] * N
    assert steps.column("right_valid").to_pylist() == right_valid.tolist()
    for side in ("left", "right"):
        col = steps.column(side).combine_chunks()
        v = right_valid if side == "right" else np.ones(N, bool)
        for f, zk in (
            ("ee_pose", "obs_ee_pose"),
            ("keypoints", "obs_keypoints"),
            ("aria_keypoints", "obs_aria_keypoints"),
        ):
            got = np.stack(col.field(f).to_numpy(zero_copy_only=False))
            np.testing.assert_array_equal(
                got[v], num[f"{side}.{zk}"][v].astype(np.float32)
            )
    gaze = np.stack(steps.column("eye_gaze").to_numpy(zero_copy_only=False))
    assert not gaze[5].any()

    for name, shape in (("front_1", (24, 32)), ("front_1_48", (H, W))):
        with av.open(str(root / dict(rec["video_path"])[name])) as c:
            frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        assert len(frames) == N and frames[0].shape[:2] == shape


def test_one_arm_embodiment_flags_the_missing_hand(tmp_path):
    path, _ = _human_zarr(tmp_path, embodiment="human_right_arm")
    d = zarr_to_lance.convert_zarr_episode(path, tmp_path / "out", META)
    meta = json.loads((d / "meta.json").read_text())
    assert meta["valid_frac"]["left"] == 0 and meta["embodiment"] == "human_right_arm"


@pytest.mark.parametrize("missing", ["lab", "operator"])
def test_required_metadata_is_never_waived(tmp_path, missing):
    path, _ = _human_zarr(tmp_path)
    with pytest.raises(ValueError, match=missing):
        zarr_to_lance.convert_zarr_episode(
            path, tmp_path / "out", {k: v for k, v in META.items() if k != missing}
        )


@pytest.mark.parametrize(
    "attrs",
    [
        # microagi v1 carries an episode_id but neither v2's source_mcap nor mecka's user_id.
        {"episode_id": "abc"},
        # An aria zarr (no vendor attrs) migrates from raw.
        {},
    ],
)
def test_unknown_or_raw_migrated_zarr_fails_clearly(tmp_path, attrs):
    path, _ = _human_zarr(tmp_path, attrs=attrs)
    with pytest.raises(NotImplementedError, match="no Lance mapping"):
        zarr_to_lance.convert_zarr_episode(path, tmp_path / "out", META)


# One synthetic episode per vendor/robot format: its hash, embodiment, attrs that identify
# it, steps schema, extra JPEG keys and whether it carries per-frame times.
VENDOR = {
    "mecka": dict(
        hash="6930855e76acbe36f0806bee",
        embodiment="human_bimanual",
        attrs={"episode_id": "6930855e76acbe36f0806bee", "user_id": "u"},
        start_ns=0x6930855E * 1_000_000_000,
    ),
    "scale": dict(
        hash="2026-05-02-16-17-35-693957",
        embodiment="human_right_arm",
        attrs={"scale_task_id": "t", "scene": "unknown", "objects": ""},
        start_ns=int(datetime(2026, 5, 2, 16, 17, 35, tzinfo=timezone.utc).timestamp())
        * 10**9
        + 693_957_000,
    ),
    "microagi_v2": dict(
        hash="2026-06-18-23-59-55-338686",
        embodiment="human_bimanual",
        attrs={"episode_id": "e", "source_mcap": "m.mcap", "objects": ["iron"]},
        timestamps=True,
    ),
    "yam": dict(
        hash="37757ec4-c681-43cc-bfe4-7be3fc520f8a",
        embodiment="yam_bimanual",
        attrs={},
        start_ns=0,
        schema=EVA_STEPS,
        cams=("front_1", "left_wrist", "right_wrist"),
    ),
    "dexmate": dict(
        hash="2026-05-29-03-16-12-000000",
        embodiment="dexmate_bimanual",
        attrs={"rig_name": "POC22018", "sharpa_tactile_units": {"fx": "N"}},
        timestamps=True,
        schema=DEXMATE_STEPS,
        cams=("front_1", "head_right"),
        extra={"left.obs_tactile_image": "left_tactile"},
    ),
}
HUMAN_KEYS = {
    f"{s}.{k}": d
    for s in ("left", "right")
    for k, d in (
        ("obs_ee_pose", 7),
        ("obs_wrist_pose", 7),
        ("obs_keypoints", 63),
    )
}


def _poses(rng, n, d):
    x = rng.normal(size=(n, d))
    if d == 7:
        x[:, 3:] /= np.linalg.norm(x[:, 3:], axis=1, keepdims=True)
    return x


def _vendor_zarr(tmp_path, fmt: str, n: int = N, annotations=None):
    spec = VENDOR[fmt]
    rng = np.random.default_rng(1)
    schema = spec.get("schema")
    if schema is None:
        num = {k: _poses(rng, n, d) for k, d in HUMAN_KEYS.items()}
        num["obs_head_pose"] = _poses(rng, n, 7)
        if spec["embodiment"] == "human_right_arm":
            # Scale writes the absent hand as identity poses and zero keypoints.
            num["left.obs_keypoints"][:] = 0
            num["left.obs_ee_pose"][:] = num["left.obs_wrist_pose"][:] = (
                zarr_to_lance.IDENTITY_POSE
            )
    else:
        num = {}
        for zk in zarr_columns(schema):
            parent, _, name = zk.rpartition(".")
            typ = (
                schema.field(parent).type.field(name) if parent else schema.field(zk)
            ).type
            if pa.types.is_integer(typ):
                num[zk] = np.arange(n, dtype=np.int64) // 3
            else:
                num[zk] = _poses(rng, n, getattr(typ, "list_size", 1))
    if spec.get("timestamps"):
        num["obs_rgb_timestamps_ns"] = (
            1_000_000_000 + np.arange(n) * 33_333_333
        ).astype(np.int64)
    imgs = {}
    for cam in spec.get("cams", ("front_1",)):
        imgs[f"images.{cam}"] = np.zeros((n, H, W, 3), np.uint8)
    for key in spec.get("extra", {}):
        imgs[key] = np.full((n, 24, 120, 3), 128, np.uint8)
    for a in imgs.values():
        for i in range(n):
            a[i, :, : 4 * (i + 1)] = 255
    k = np.array([[50.0, 0, 32, 0], [0, 50.0, 24, 0], [0, 0, 1, 0]])
    path = tmp_path / "zarr" / f"{spec['hash']}.zarr"
    ZarrWriter.create_and_write(
        path,
        numeric_data=num,
        image_data=imgs,
        embodiment=spec["embodiment"],
        task_name="pick",
        operator="op",
        lab="vendorlab",
        scene=MissingMetadata(),
        objects=MissingMetadata(),
        vendor_recording_id=spec["hash"],
        chunk_timesteps=5,
        intrinsics={"front_1": k},
        extrinsics={"front_1": np.eye(4)} if fmt == "yam" else None,
        annotations=annotations,
    )
    # As the vendor zarrs were written: their own attrs, no lab or operator, and mecka's
    # task_name is "debug".
    z = zarr.open_group(str(path), mode="r+")
    attrs = {k: v for k, v in z.attrs.items() if k not in ("lab", "operator")}
    z.attrs.put({**attrs, "task_name": "debug", **spec["attrs"]})
    return path, num


def _commit(tmp_path, d):
    staging, root = tmp_path / "staging", tmp_path / "lance"
    commit.stage(d, commit.join(staging, "zarr", d.name))
    report = commit_staged(str(staging), "zarr", str(root), tmp_path / "work")
    assert report["rejected"] == {}
    return root


def _identity_key_map(keys) -> dict:
    return {k: {"key_type": "proprio_keys", "zarr_key": k} for k in keys}


VENDOR_META = {"lab": "vendorlab", "operator": "op", "task": "pick_place"}


@pytest.mark.parametrize("fmt", sorted(VENDOR))
def test_vendor_zarr_round_trips_through_the_committer(tmp_path, fmt):
    spec = VENDOR[fmt]
    path, num = _vendor_zarr(tmp_path, fmt)
    d = zarr_to_lance.convert_zarr_episode(path, tmp_path / "out", VENDOR_META)
    root = _commit(tmp_path, d)

    rid = recording_id("zarr", spec["hash"])
    assert d.name == spec["hash"]
    (rec,) = lance.dataset(root / "recordings.lance").to_table().to_pylist()
    assert rec["recording_id"] == rid and rec["embodiment"] == spec["embodiment"]
    assert (rec["lab"], rec["operator"]) == ("vendorlab", "op")
    (task,) = lance.dataset(root / "tasks.lance").to_table().to_pylist()
    assert task["task_name"] == "pick_place"
    if fmt == "scale":
        assert rec["scene"] is None and task["objects"] == []
    if fmt == "microagi_v2":
        assert task["objects"] == ["iron"]
    videos = set(dict(rec["video_path"]))
    assert set(spec.get("cams", ("front_1",))) <= videos
    assert set(spec.get("extra", {}).values()) <= videos
    meta = json.loads((d / "meta.json").read_text())
    assert meta["format"] == fmt
    assert meta["provenance"]["attrs"].keys() >= set(spec["attrs"]) - {
        "objects",
        "scene",
    }
    if fmt == "yam":
        assert rec["captured_at"] is None
        assert dict(rec["extrinsics"]).keys() == {"front_1"}
    if fmt == "dexmate":
        assert rec["rig_name"] == "POC22018" and dict(rec["extrinsics"]) == {}

    ee = END_EFFECTOR.HUMAN_HAND if spec.get("schema") is None else None
    ee = ee or (
        END_EFFECTOR.DEX_HAND if fmt == "dexmate" else END_EFFECTOR.PARALLEL_JAW_GRIPPER
    )
    steps = lance.dataset(
        root / steps_path("vendorlab", spec["embodiment"], ee.value)
    ).to_table()
    ts = steps.column("timestamp_ns").to_numpy()
    expect = (
        num["obs_rgb_timestamps_ns"]
        if spec.get("timestamps")
        else spec["start_ns"] + np.round(np.arange(N) * 1e9 / 30).astype(np.int64)
    )
    np.testing.assert_array_equal(ts, expect)
    left_valid = fmt != "scale"
    assert steps.column("left_valid").to_pylist() == [left_valid] * N
    assert steps.column("right_valid").to_pylist() == [True] * N

    keys = (
        [k for k in HUMAN_KEYS if fmt != "scale" or k.startswith("right.")]
        + ["obs_head_pose"]
        if spec.get("schema") is None
        else [k for k in zarr_columns(spec["schema"]) if k != "sharpa.subtask_index"]
    )
    key_map = _identity_key_map(keys)
    leaf = LanceEpisodeResolver(root, key_map=key_map, cameras={}).resolve()[rid]
    twin = ZarrDataset(path, key_map=key_map)
    assert len(leaf) == len(twin) == N
    for i in range(N):
        a, b = leaf[i], twin[leaf.anchor(i)]
        for k in keys:
            torch.testing.assert_close(a[k], b[k].to(a[k].dtype), msg=k)


def test_robot_zero_pose_frames_are_invalid_and_forward_filled(tmp_path):
    path, num = _vendor_zarr(tmp_path, "yam")
    z = zarr.open_group(str(path), mode="r+")
    for k in ("left.obs_ee_pose", "left.cmd_ee_pose", "left.obs_joints"):
        a = z[k][:]
        a[4] = 0
        z[k][:] = a
    d = zarr_to_lance.convert_zarr_episode(path, tmp_path / "out", VENDOR_META)
    steps = pq.read_table(d / "steps.parquet")
    left_valid = np.ones(N, bool)
    left_valid[4] = False
    assert steps.column("left_valid").to_pylist() == left_valid.tolist()
    assert steps.column("right_valid").to_pylist() == [True] * N
    left = steps.column("left").combine_chunks()
    for f in ("obs_ee_pose", "obs_joints", "cmd_gripper"):
        got = np.asarray(left.field(f).to_pylist(), np.float32)
        np.testing.assert_array_equal(got[4], got[3], err_msg=f)


def test_untracked_vendor_hand_frames_are_invalid(tmp_path):
    path, _ = _vendor_zarr(tmp_path, "mecka")
    z = zarr.open_group(str(path), mode="r+")
    a = z["right.obs_keypoints"][:]
    a[2] = 0
    z["right.obs_keypoints"][:] = a
    d = zarr_to_lance.convert_zarr_episode(path, tmp_path / "out", VENDOR_META)
    right = pq.read_table(d / "steps.parquet").column("right_valid").to_pylist()
    assert right == [i != 2 for i in range(N)]


def test_zarr_annotations_commit_on_the_recording_clock(tmp_path):
    human, num = _human_zarr(
        tmp_path, annotations=[("reach", 0, 4), ("fold", 4, N + 5), ("bad", N, N + 5)]
    )
    # No obs_rgb_timestamps_ns: spans map through the frame index alone.
    yam, _ = _vendor_zarr(tmp_path, "yam", annotations=[("grasp", 2, 5)])
    staging, root = tmp_path / "staging", tmp_path / "lance"
    for path, meta in ((human, META), (yam, VENDOR_META)):
        d = zarr_to_lance.convert_zarr_episode(path, tmp_path / "out", meta)
        commit.stage(d, commit.join(staging, "zarr", d.name))
    dropped = json.loads((tmp_path / "out" / HASH / "meta.json").read_text())
    assert len(dropped["annotations_dropped"]) == 1
    report = commit_staged(str(staging), "zarr", str(root), tmp_path / "work")
    assert report["rejected"] == {}
    (ann,) = report["annotations"]
    assert ann["rows"] == 3 and ann["errors"] == [] and ann["tag"]
    assert ann["deleted"] == 0

    rows = {
        r["text"]: r
        for r in lance.dataset(root / "annotations.lance").to_table().to_pylist()
    }
    ts = num["obs_rgb_timestamps_ns"]
    period = round(1e9 / 30)
    assert (rows["reach"]["t_start_ns"], rows["reach"]["t_end_ns"]) == (ts[0], ts[4])
    assert (rows["fold"]["t_start_ns"], rows["fold"]["t_end_ns"]) == (
        ts[4],
        ts[-1] + period,
    )
    assert rows["fold"]["recording_id"] == recording_id("zarr", HASH)
    assert rows["fold"]["annotation_vendor"] == "rl2"
    grasp = rows["grasp"]
    assert grasp["recording_id"] == recording_id("zarr", VENDOR["yam"]["hash"])
    assert (grasp["t_start_ns"], grasp["t_end_ns"]) == (
        round(2 * 1e9 / 30),
        round(5 * 1e9 / 30),
    )
    assert grasp["annotation_vendor"] == "vendorlab"
    sources = [
        c["source"]
        for c in lance.dataset(root / "changes.lance").to_table().to_pylist()
    ]
    assert sorted(sources) == ["annotations", "zarr"]


def test_replacing_a_recording_drops_its_stale_annotations(tmp_path):
    human, _ = _human_zarr(tmp_path, annotations=[("reach", 0, 4)])
    yam, _ = _vendor_zarr(tmp_path, "yam", annotations=[("grasp", 2, 5)])
    staging, root = tmp_path / "staging", tmp_path / "lance"

    def stage(path, meta, out):
        d = zarr_to_lance.convert_zarr_episode(path, tmp_path / out, meta)
        commit.stage(d, commit.join(staging, "zarr", d.name))

    stage(human, META, "out1")
    stage(yam, VENDOR_META, "out1")
    commit_staged(str(staging), "zarr", str(root), tmp_path / "work")
    z = zarr.open_group(str(yam), mode="r+")
    z["annotations"][:] = [
        json.dumps({"text": "place", "start_idx": 1, "end_idx": 3}).encode()
    ]
    stage(yam, VENDOR_META, "out2")
    report = commit_staged(
        str(staging), "zarr", str(root), tmp_path / "work", replace=True
    )
    (ann,) = report["annotations"]
    assert (ann["rows"], ann["deleted"]) == (1, 1)
    rows = lance.dataset(root / "annotations.lance").to_table().to_pylist()
    assert sorted(r["text"] for r in rows) == ["place", "reach"]


def test_vendor_prefixed_embodiment_maps_to_human(tmp_path):
    path, _ = _vendor_zarr(tmp_path, "mecka")
    z = zarr.open_group(str(path), mode="r+")
    z.attrs.put({**z.attrs, "embodiment": "MECKA_BIMANUAL"})
    d = zarr_to_lance.convert_zarr_episode(path, tmp_path / "out", VENDOR_META)
    meta = json.loads((d / "meta.json").read_text())
    assert (meta["format"], meta["embodiment"]) == ("mecka", "human_bimanual")


def test_blank_manifest_cells_fall_back_to_zarr_attrs(tmp_path):
    path, _ = _vendor_zarr(tmp_path, "scale")
    z = zarr.open_group(str(path), mode="r+")
    z.attrs.put({**z.attrs, "task_name": "pick", "task_description": "pick it"})
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "episode_hash,lab,operator,task,task_description\n"
        f"{VENDOR['scale']['hash']},vendorlab,op,,\n"
    )
    row = read_manifest(manifest)[VENDOR["scale"]["hash"]]
    d = zarr_to_lance.convert_zarr_episode(path, tmp_path / "out", row)
    meta = json.loads((d / "meta.json").read_text())
    assert (meta["task_name"], meta["task_description"]) == ("pick", "pick it")
    with pytest.raises(ValueError, match="operator"):
        zarr_to_lance.convert_zarr_episode(
            path, tmp_path / "out", {**row, "operator": "n/a"}
        )
