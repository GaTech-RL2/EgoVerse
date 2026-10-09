"""Protect native-action pass-through and honest qualification accounting."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astra_reversal.complex_manipulation.xiaomi_eval import RecordedClient, summarize
from astra_reversal.complex_manipulation.xiaomi_stage import validate_release_file


def test_release_validation_catches_same_length_weight_and_code_corruption(tmp_path):
    p = tmp_path / "model"
    p.write_bytes(b"abc")
    sha = hashlib.sha256(b"abc").hexdigest()
    git = hashlib.sha1(b"blob 3\0abc").hexdigest()
    for metadata in ({"size": 3, "lfs": {"sha256": sha}}, {"size": 3, "blobId": git}):
        validate_release_file(p, metadata)
        p.write_bytes(b"abd")
        with pytest.raises(ValueError, match="mismatch"):
            validate_release_file(p, metadata)
        p.write_bytes(b"abc")


def test_instrumentation_returns_exact_native_action_array_without_edit(tmp_path):
    import time
    actions = np.arange(32 * 12, dtype=np.float32).reshape(32, 12)
    record = SimpleNamespace(directory=tmp_path, queries=0, policy_seconds=0., steps=0,
                             started=time.perf_counter())
    native = SimpleNamespace(infer=lambda *args: actions)
    client = RecordedClient(native, record)
    returned = client.infer(None, None, "original instruction")
    assert returned is actions
    assert record.queries == 1
    native.infer = lambda *args: np.full((32, 12), np.nan)
    with pytest.raises(ValueError, match="contract"):
        client.infer(None, None, "original instruction")
    assert record.queries == 1


def test_partial_batch_does_not_claim_completed_sr(tmp_path):
    rows = [{"cohort": "reference", "task": "CloseFridge", "seed": 57,
             "completed": True, "success": True, "steps": 160, "policy_queries": 10}]
    s = summarize(rows, 5)
    assert s["completed_episode_sr"] == 1.0
    assert s["intended_batch_sr"] is None
    assert s["recorded_episodes"] == 1 and s["reset_free_action_chunks"] == 10
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(rows + rows, 5)


def test_registered_reference_seeds_preserve_original_task_indices():
    root = Path(__file__).resolve().parents[3]
    protocol = json.loads((root / "astra_reversal/complex_manipulation/xiaomi_protocol.json").read_text())
    assert sum(len(g["episodes"]) for g in protocol["groups"]) == 21
    for g in protocol["groups"]:
        derived = [g["base_seed"] + g["task_index"] * g["native_num_trials"] + e for e in g["episodes"]]
        assert derived == g["expected_seeds"]
        if g["cohort"] == "reference_first5":
            assert g["episodes"] == list(range(5))
    assert protocol["criteria"]["retries_per_reset"] == 0
