"""Unit tests for serving/input_recorder.py — no GPU, no server."""
import json
import logging
import os
import queue
import time
from pathlib import Path

import numpy as np
import pytest

from egomimic.serving.input_recorder import (
    InputRecorder, load_session, make_session_dir, _stack_records)


def _obs(i, n=1024, ch=3):
    return {"front_pcd_1": (np.ones((n, ch), np.float32) * i),
            "robot0_joint_pos": np.arange(22, dtype=np.float32) + i,
            "task_id": np.zeros(64, np.float32)}


def _act(i):
    return {"actions": np.full((1, 32, 49), float(i), np.float32),
            "server_timing": {"infer_ms": 1.0}}


def test_roundtrip_preserves_order_shapes_dtypes(tmp_path):
    rec = InputRecorder(tmp_path / "s", {"checkpoint": "x/RBY1_a/b/checkpoints/e.ckpt"},
                        chunk_size=40, flush_s=0.2)
    for i in range(250):
        rec.record(_obs(i), _act(i), conn_id=1 + (i >= 100), infer_ms=0.5 * i)
    rec.close()
    d = load_session(tmp_path / "s")
    assert d["idx"].tolist() == list(range(250))
    assert d["obs/front_pcd_1"].shape == (250, 1024, 3)
    assert d["obs/front_pcd_1"].dtype == np.float32
    assert np.all(d["obs/front_pcd_1"][:, 0, 0] == np.arange(250))
    assert d["act/actions"].shape == (250, 1, 32, 49)
    assert d["conn_id"][99] == 1 and d["conn_id"][100] == 2
    assert d["infer_ms"][10] == 5.0
    assert "server_timing" not in "".join(d.keys())
    assert d["meta"]["records_written"] == 250 and d["meta"]["records_dropped"] == 0
    assert len(sorted((tmp_path / "s").glob("chunk_*.npz"))) == 7  # 6x40 + 10
    assert not list((tmp_path / "s").glob("*.tmp"))


def test_ragged_and_mixed_keys_become_object_arrays(tmp_path):
    rec = InputRecorder(tmp_path / "s", chunk_size=10, flush_s=0.2)
    rec.record(_obs(0, n=1024), _act(0))
    rec.record(_obs(1, n=512), _act(1))                 # different N
    rec.record({"only_here": np.zeros(3)}, _act(2))      # missing keys
    rec.close()
    d = load_session(tmp_path / "s")
    assert d["obs/front_pcd_1"].dtype == object
    assert d["obs/front_pcd_1"][1].shape == (512, 3)
    assert d["obs/front_pcd_1"][2] is None
    assert d["obs/only_here"][2].shape == (3,)


def test_batch_list_records_each_element(tmp_path):
    rec = InputRecorder(tmp_path / "s", chunk_size=10, flush_s=0.2)
    rec.record([_obs(0), _obs(1), _obs(2)], [_act(0), _act(1), _act(2)], conn_id=7)
    rec.close()
    d = load_session(tmp_path / "s")
    assert len(d["idx"]) == 3 and np.all(d["conn_id"] == 7)


def test_every_n_and_max_gb(tmp_path):
    rec = InputRecorder(tmp_path / "s", chunk_size=5, flush_s=0.1, every_n=3,
                        max_gb=1e-6)   # ~1 KB cap -> first chunk trips it
    for i in range(60):
        rec.record(_obs(i, n=64), _act(i))
    rec.close()
    d = load_session(tmp_path / "s")
    assert d["idx"][:3].tolist() == [0, 3, 6]          # thinned
    st = rec.status()
    assert st["written"] == 5                           # exactly one chunk
    assert st["dropped"] > 0 and st["errors"] == 0      # rest discarded, no error
    assert st["written"] + st["dropped"] == 20          # 60 / every_n=3


def test_queue_full_drops_without_raising(tmp_path):
    rec = InputRecorder(tmp_path / "s", chunk_size=1, flush_s=0.05, queue_max=5)
    # Simulate a stalled disk: the writer blocks inside _flush, so the queue
    # (cap 5) fills and record() must drop, never block, never raise.
    real_flush = rec._flush

    def slow_flush(buf):
        time.sleep(1.5)
        real_flush(buf)
    rec._flush = slow_flush
    time.sleep(0.1)
    for i in range(30):
        rec.record(_obs(i, n=16), _act(i))
    assert rec.n_dropped >= 20                        # 30 - (1 in writer + 5 queued) = 24
    assert rec.n_recorded + rec.n_dropped == 30
    rec.close(timeout=5.0)
    assert rec.status()["errors"] == 0


def test_unwritable_dir_keeps_serving(tmp_path, caplog):
    d = tmp_path / "ro"
    d.mkdir()
    rec = InputRecorder(d, chunk_size=2, flush_s=0.1)
    os.chmod(d, 0o500)  # read+exec only -> chunk writes fail
    try:
        with caplog.at_level(logging.WARNING):
            for i in range(6):
                rec.record(_obs(i, n=8), _act(i))  # must not raise
            time.sleep(0.5)
            rec.close()
        assert rec.status()["errors"] >= 1
        assert rec.status()["dropped"] >= 1
    finally:
        os.chmod(d, 0o700)


def test_record_never_raises_on_garbage(tmp_path):
    rec = InputRecorder(tmp_path / "s", chunk_size=2, flush_s=0.1)
    rec.record("not a dict", object())       # swallowed + counted
    rec.record(None, None)
    rec.close()
    assert rec.status()["errors"] >= 0       # no exception escaped


def test_connections_log_and_session_dir(tmp_path):
    rec = InputRecorder(tmp_path / "s", chunk_size=2, flush_s=0.1)
    rec.on_connection(1, "open", ("10.0.0.5", 1234))
    rec.on_connection(1, "close", ("10.0.0.5", 1234))
    rec.close()
    lines = [json.loads(l) for l in (tmp_path / "s" / "connections.jsonl").read_text().splitlines()]
    assert [l["event"] for l in lines] == ["open", "close"]
    p = make_session_dir(tmp_path, "checkpoints/RBY1_dp3c_dual/dp3c_dual_2k/checkpoints/epoch_epoch=1999.ckpt", 8010)
    assert p.name.startswith("dp3c_dual_8010_")


def test_stack_records_scalars_and_arrays():
    out = _stack_records([{"a": np.zeros(3), "s": 1.5}, {"a": np.ones(3), "s": 2.5}])
    assert out["a"].shape == (2, 3) and out["s"].tolist() == [1.5, 2.5]
