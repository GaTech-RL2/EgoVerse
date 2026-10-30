"""yam_bimanual and dexmate_bimanual steps tables: committed as delivered, read like zarr."""

import numpy as np
import pyarrow as pa
import pytest
import torch

lance = pytest.importorskip("lance")

from egomimic.rldb.lance import commit  # noqa: E402
from egomimic.rldb.lance.dataset import LanceEpisodeResolver  # noqa: E402
from egomimic.rldb.lance.schema import (  # noqa: E402
    DEXMATE_STEPS,
    EVA_STEPS,
    RECORDINGS,
    recording_id,
    robot_steps_table,
    steps_path,
    steps_schema,
    zarr_columns,
)
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset  # noqa: E402
from egomimic.rldb.zarr.zarr_writer import ZarrWriter  # noqa: E402

PERIOD_NS = 33_333_333
H = 4
TASK = "fold and stack the skirts"
STEPS_KEYS = {
    "yam": ("abc", "yam_bimanual", "parallel_jaw_gripper"),
    "dexmate": ("sharpa_dex", "dexmate_bimanual", "dex_hand"),
}


def _side_keys(side_keys: dict, sides=("left", "right")) -> dict:
    return {
        f"{s}.{k}": {**spec, "zarr_key": f"{s}.{spec['zarr_key']}"}
        for s in sides
        for k, spec in side_keys.items()
    }


ANNOTATIONS = {
    "annotations": {"key_type": "annotation_keys", "zarr_key": "annotations"}
}
# The numeric keys of Yam._get_keymap("cartesian") (aniketh/arc), horizon shortened.
YAM_KEY_MAP = {
    **_side_keys(
        {
            "obs_ee_pose": {"key_type": "proprio_keys", "zarr_key": "obs_ee_pose"},
            "obs_gripper": {"key_type": "proprio_keys", "zarr_key": "obs_gripper"},
            "cmd_gripper": {
                "key_type": "action_keys",
                "zarr_key": "cmd_gripper",
                "horizon": H,
            },
            "cmd_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "cmd_ee_pose",
                "horizon": H,
            },
        }
    ),
    **ANNOTATIONS,
}
# The numeric keys of Dexmate._get_keymap("keypoints") (jmni/sharpa-ingest-tooling).
DEXMATE_KEY_MAP = {
    **_side_keys(
        {
            "action_keypoints": {
                "key_type": "action_keys",
                "zarr_key": "obs_hand_keypoints",
                "horizon": H,
            },
            "action_wrist_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "obs_ee_pose",
                "horizon": H,
            },
            "obs_keypoints": {
                "key_type": "proprio_keys",
                "zarr_key": "obs_hand_keypoints",
            },
            "obs_wrist_pose": {"key_type": "proprio_keys", "zarr_key": "obs_ee_pose"},
        }
    ),
    "obs_head_link_pose": {
        "key_type": "proprio_keys",
        "zarr_key": "obs_head_link_pose",
    },
    **ANNOTATIONS,
}


def _every_column(schema) -> dict:
    """A key map reading every steps column, by its zarr key."""
    # Zarr stores this one (T,), so it samples as () where Lance gives (1,), like the grippers.
    skip = {"sharpa.subtask_index"}
    return {
        zk: {"key_type": "proprio_keys", "zarr_key": zk}
        for zk in zarr_columns(schema)
        if zk not in skip
    } | ANNOTATIONS


def _arrays(schema, n: int, seed: int) -> dict[str, np.ndarray]:
    """{zarr key: (n, d)} of `schema`'s columns: unit-quaternion poses, distinct ramps elsewhere."""
    rng = np.random.default_rng(seed)
    out = {}
    for i, (zk, col) in enumerate(zarr_columns(schema).items()):
        parent, _, name = col.rpartition(".")
        typ = (
            schema.field(parent).type.field(name) if parent else schema.field(name)
        ).type
        if pa.types.is_integer(typ):
            out[zk] = np.arange(n, dtype=np.int64) // 3
            continue
        d = typ.list_size if pa.types.is_fixed_size_list(typ) else 1
        x = rng.normal(0, 0.2, (n, d)) + np.arange(n)[:, None] * 0.01 + i
        if zk.endswith("pose") and d == 7:
            x[:, 3:] = [1.0, 0.0, 0.0, 0.0]
        out[zk] = x.astype(np.float32)
    return out


def _delivery(tmp, kind: str, i: int, n: int, valid=None):
    lab, embodiment, end_effector = STEPS_KEYS[kind]
    schema = steps_schema(embodiment, end_effector)
    vid = f"{kind}-{i}"
    rid = recording_id(kind, vid)
    arrays = _arrays(schema, n, seed=i)
    flags = valid or {s: np.ones(n, bool) for s in ("left", "right")}
    steps = robot_steps_table(
        schema, rid, 10**12 * i + np.arange(n) * PERIOD_NS, arrays, flags
    )
    video = tmp / "src" / vid / "front_1.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"mp4")
    h = f"2026-01-01-00-00-{i:02d}-000000"
    t0, t1 = 10**12 * i, 10**12 * i + n * PERIOD_NS
    d = commit.Delivery(
        recording=dict(
            recording_id=rid,
            vendor_recording_id=vid,
            lab=lab,
            embodiment=embodiment,
            end_effector=end_effector,
            operator="op",
            fps=30.0,
            num_frames=n,
            intrinsics=[("front_1", np.eye(3).ravel().tolist())],
            extrinsics=[],
        ),
        task=dict(
            task_instance_id=f"t-{vid}",
            recording_id=rid,
            t_start_ns=t0,
            t_end_ns=t1,
            task_name="fold",
            task_description=TASK,
            objects=[],
        ),
        alias=dict(episode_hash=h, recording_id=rid, t_start_ns=t0, t_end_ns=t1),
        steps=steps,
        videos={"front_1": video},
    )
    return d, arrays


def test_registered_steps_schemas():
    assert steps_schema("yam_bimanual", "parallel_jaw_gripper") is EVA_STEPS
    assert steps_schema("dexmate_bimanual", "dex_hand") is DEXMATE_STEPS
    # "human_hand" is a tracked human hand: its steps tables hold HUMAN_STEPS.
    for emb, ee in [("dexmate_bimanual", "human_hand"), ("yam_bimanual", "human_hand")]:
        with pytest.raises(ValueError, match="no steps schema"):
            steps_schema(emb, ee)
    assert steps_path(*STEPS_KEYS["dexmate"]) == (
        "steps/sharpa_dex/dexmate_bimanual/dex_hand.lance"
    )


def test_commit_round_trips_yam_and_dexmate_steps(tmp_path):
    root = tmp_path / "lance"
    ds = [
        _delivery(tmp_path, "yam", 0, 10)[0],
        _delivery(tmp_path, "yam", 1, 7)[0],
        _delivery(tmp_path, "dexmate", 2, 9)[0],
    ]
    assert commit.append(root, ds) == [d.recording_id for d in ds]
    for kind in STEPS_KEYS:
        sent = pa.concat_tables(
            [d.steps for d in ds if d.recording["vendor_recording_id"].startswith(kind)]
        )
        got = lance.dataset(root / steps_path(*STEPS_KEYS[kind])).to_table()
        assert got.schema.equals(sent.schema)
        assert got.equals(sent)
    rec = lance.dataset(root / "recordings.lance").to_table()
    assert rec.schema.equals(RECORDINGS)
    assert sorted(rec.column("end_effector").to_pylist()) == [
        "dex_hand",
        "parallel_jaw_gripper",
        "parallel_jaw_gripper",
    ]


def test_committer_rejects_steps_not_in_the_key_schema(tmp_path):
    root = tmp_path / "lance"
    dex, _ = _delivery(tmp_path, "dexmate", 0, 5)
    yam, _ = _delivery(tmp_path, "yam", 1, 5)
    wrong = commit.Delivery(yam.recording, yam.task, yam.alias, dex.steps, yam.videos)
    with pytest.raises(ValueError, match="steps schema"):
        commit.append(root, [wrong])
    as_hand = dict(dex.recording, end_effector="human_hand")
    with pytest.raises(ValueError, match="no steps schema"):
        commit.steps_key(as_hand)
    assert commit.committed_ids(root) == set()


def _zarr_twin(root, d, arrays, key_map) -> ZarrDataset:
    n = d.recording["num_frames"]
    path = ZarrWriter.create_and_write(
        root / f"{d.recording_id}.zarr",
        numeric_data={k: v.astype(np.float64) for k, v in arrays.items()},
        embodiment=d.recording["embodiment"],
        fps=30,
        annotations=[(TASK, 0, n)],
        intrinsics={"front_1": np.c_[np.eye(3), np.zeros(3)]},
        task_name=TASK,
        operator="op",
        lab=d.recording["lab"],
        scene="test_scene",
        objects=["cloth"],
        vendor_recording_id=d.recording["vendor_recording_id"],
    )
    return ZarrDataset(path, key_map=key_map)


@pytest.mark.parametrize(
    "kind,key_map",
    [
        ("yam", YAM_KEY_MAP),
        ("yam", _every_column(EVA_STEPS)),
        ("dexmate", DEXMATE_KEY_MAP),
        ("dexmate", _every_column(DEXMATE_STEPS)),
    ],
)
def test_lance_leaves_match_the_zarr_episode(tmp_path, kind, key_map):
    root = tmp_path / "lance"
    made = [_delivery(tmp_path, kind, i, n) for i, n in ((0, 9), (1, 6))]
    commit.append(root, [d for d, _ in made])
    leaves = LanceEpisodeResolver(root, key_map=key_map, cameras={}).resolve()
    assert set(leaves) == {d.recording_id for d, _ in made}
    for d, arrays in made:
        leaf = leaves[d.recording_id]
        twin = _zarr_twin(tmp_path, d, arrays, key_map)
        assert len(leaf) == len(twin) == d.recording["num_frames"]
        for i in range(len(leaf)):
            a, b = leaf[i], twin[leaf.anchor(i)]
            # Only Lance carries arm masks; the every-column maps have no action keys.
            assert a.pop("action_hand_mask", torch.ones(2)).tolist() == [1, 1]
            assert a.pop("proprio_hand_valid").tolist() == [1, 1]
            assert a.keys() == b.keys()
            for k in a:
                if isinstance(a[k], torch.Tensor):
                    torch.testing.assert_close(a[k], b[k].to(a[k].dtype), msg=k)
                else:
                    assert a[k] == b[k], k


def test_an_arm_dropout_masks_the_dex_hand(tmp_path):
    root = tmp_path / "lance"
    n = 8
    right = np.ones(n, bool)
    right[5] = False
    d, _ = _delivery(
        tmp_path, "dexmate", 0, n, valid={"left": np.ones(n, bool), "right": right}
    )
    commit.append(root, [d])
    leaf = LanceEpisodeResolver(
        root, key_map=DEXMATE_KEY_MAP, cameras={}, require_hands_in_frame="either"
    ).resolve()[d.recording_id]
    assert leaf[3]["action_hand_mask"].tolist() == [1, 0]
    assert leaf[0]["action_hand_mask"].tolist() == [1, 1]
