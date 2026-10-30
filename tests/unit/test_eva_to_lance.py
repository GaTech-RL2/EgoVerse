import json
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

h5py = pytest.importorskip("h5py")

from egomimic.rldb.lance.schema import EVA_STEPS  # noqa: E402
from egomimic.scripts.eva_process.eva_to_lance import (  # noqa: E402
    convert,
    episode_metadata,
    numeric_columns,
    required_fields,
    steps_table,
)
from egomimic.scripts.eva_process.eva_to_zarr import (  # noqa: E402
    _separate_numeric_and_image,
    _split_per_arm,
)
from egomimic.scripts.eva_process.eva_utils import EvaHD5Extractor  # noqa: E402

N = 10


def _demo(tmp_path, name="1766709615098"):
    """A bimanual demo whose right arm reads all zeros at frames 3-4 in every array."""
    rng = np.random.default_rng(0)
    path = tmp_path / f"{name}.hdf5"
    with h5py.File(path, "w") as f:
        for group, key in (
            ("observations", "eepose"),
            ("observations", "joints"),
            ("actions", "eepose"),
            ("actions", "joints"),
        ):
            a = rng.uniform(0.1, 0.5, (N, 14)).astype(np.float32)
            a[:, 6] = a[:, 13] = np.linspace(0.002, 0.09, N)
            a[3:5, 7:13] = 0
            f[f"{group}/{key}"] = a
        f["observations/images/front_img_1"] = np.zeros((N, 4, 4, 3), np.uint8)
    return path


def test_steps_match_the_zarr_arrays_and_flag_dropouts(tmp_path):
    path = _demo(tmp_path)
    cols, dropout = numeric_columns(path, "both")
    want = _split_per_arm(
        _separate_numeric_and_image(EvaHD5Extractor.process_episode(path, "both"))[0],
        "both",
    )
    assert set(want) <= set(cols)
    for k in want:
        np.testing.assert_array_equal(cols[k], want[k])
    assert (
        dropout[:, 0].sum() == 0 and dropout[3:5, 1].all() and dropout[:, 1].sum() == 2
    )
    # Gripper rescale at ingest: a pre-cutoff demo spans [0, 1].
    assert cols["left.obs_gripper"].min() == 0 and cols["left.obs_gripper"].max() == 1

    valid = {"left": ~dropout[:, 0], "right": ~dropout[:, 1]}
    tbl = steps_table("r", np.arange(N, dtype=np.int64), cols, valid)
    assert tbl.schema == EVA_STEPS
    assert tbl.column("right_valid").to_pylist() == [i not in (3, 4) for i in range(N)]
    right = tbl.column("right").combine_chunks()
    np.testing.assert_allclose(
        np.stack(right.field("cmd_ee_pose").to_numpy(zero_copy_only=False)),
        want["right.cmd_ee_pose"],
    )
    np.testing.assert_allclose(
        right.field("obs_gripper").to_numpy(), want["right.obs_gripper"][:, 0]
    )


def test_av1_encoder_keeps_a_fractional_frame_rate(tmp_path, monkeypatch):
    from egomimic.utils import video_utils

    cmds = []
    monkeypatch.setattr(
        video_utils.subprocess, "Popen", lambda cmd, **kw: cmds.append(cmd)
    )
    video_utils.open_av1_encoder(tmp_path / "v.mp4", 48, 64, 29.97, "ffmpeg")
    (cmd,) = cmds
    assert cmd[cmd.index("-r") + 1] == "29.97"


def test_right_arm_robot_reads_the_right_slot_and_zeros_the_left(tmp_path):
    path = _demo(tmp_path)
    cols, dropout = numeric_columns(path, "right")
    both, _ = numeric_columns(path, "both")
    assert {k.split(".")[0] for k in cols} == {"right"}
    np.testing.assert_array_equal(cols["right.obs_ee_pose"], both["right.obs_ee_pose"])
    valid = {"left": np.zeros(N, bool), "right": ~dropout[:, 1]}
    tbl = steps_table("r", np.arange(N, dtype=np.int64), cols, valid)
    pq.write_table(tbl, tmp_path / "steps.parquet")
    tbl = pq.read_table(tmp_path / "steps.parquet")
    left = tbl.column("left").combine_chunks()
    assert not np.stack(left.field("obs_ee_pose").to_numpy(zero_copy_only=False)).any()
    assert tbl.column("right_valid").to_pylist() == list(valid["right"])


GOOD_MD = {"lab": "rl2", "task": "fold_clothes", "operator": "op_7"}


def test_required_fields_pass_without_scene_or_objects():
    assert required_fields(GOOD_MD) == {
        "lab": "rl2",
        "task_name": "fold_clothes",
        "operator": "op_7",
    }


@pytest.mark.parametrize("key", ["lab", "task", "operator"])
@pytest.mark.parametrize("bad", [None, "", "  ", "unknown", "N/A", "placeholder"])
def test_lab_task_and_operator_are_never_waivable(key, bad):
    md = {**GOOD_MD, key: bad}
    if bad is None:
        del md[key]
    field = "task_name" if key == "task" else key
    with pytest.raises(ValueError, match=field):
        required_fields(md)


def test_convert_fails_the_episode_before_writing_without_an_operator(tmp_path):
    vid = "1766709615098"
    _demo(tmp_path, vid)
    md = {"robot_name": "eva_bimanual", "lab": "rl2", "task": "fold_clothes"}
    (tmp_path / f"{vid}_metadata.json").write_text(json.dumps(md))
    out = tmp_path / "out"
    args = SimpleNamespace(
        raw_dir=tmp_path, out_dir=out, vendor_recording_id=vid, fps=30.0, ffmpeg="x"
    )
    with pytest.raises(ValueError, match="operator"):
        convert(args)
    assert not out.exists()


def test_the_sql_row_replaces_the_capture_metadata():
    md = {"robot_name": "eva_bimanual", "lab": "old", "task": "t", "scene": "s"}
    sql = {"lab": "rl2", "operator": "op_7", "task": "fold_clothes", "scene": np.nan}
    got = episode_metadata(md, sql)
    assert required_fields(got) == {
        "lab": "rl2",
        "task_name": "fold_clothes",
        "operator": "op_7",
    }
    assert got["robot_name"] == "eva_bimanual" and got["scene"] is None
    assert episode_metadata(md, None) is md


def test_a_null_sql_operator_is_not_filled_from_the_capture():
    md = {**GOOD_MD, "robot_name": "eva_bimanual"}
    with pytest.raises(ValueError, match="operator"):
        required_fields(episode_metadata(md, {**GOOD_MD, "operator": None}))
