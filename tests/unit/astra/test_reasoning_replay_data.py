import copy
import json

import numpy as np
import pytest

from astra_reversal.reasoning_learning.evidence import training_windows
from astra_reversal.reasoning_learning.replay_data import load
from astra_reversal.records import digest, file_sha256


def make_replay(tmp_path):
    directory = tmp_path / "rollout_0"
    directory.mkdir()
    raw = {
        "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((2, 2, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": "test task",
    }
    oid = digest(raw)
    np.savez_compressed(directory / (oid + ".npz"), **raw)
    labels = [
        {
            "episode_id": "e",
            "step": i,
            "observation_id": oid,
            "action": [0.0] * 7,
            "executed": True,
            "policy_version": 0,
            "batch_id": f"b{i // 5}",
            "event_id": "correction" if i < 5 else None,
            "preference": "win" if i < 5 else None,
            "stage": "correction" if i < 5 else "continuation",
            "evidence": "observed_useful",
            "environment_success": False,
            "terminated": False,
            "after_observation_sha256": "after",
        }
        for i in range(10)
    ]
    actual = copy.deepcopy(labels)
    for row in actual:
        row["evidence"] = "ambiguous"
    (directory / "executed_steps.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in actual)
    )
    windows = training_windows(labels, horizon=10)
    (directory / "admission.jsonl").write_text(
        json.dumps({"steps": labels, "windows": windows}) + "\n"
    )
    (directory / "result.json").write_text(
        json.dumps(
            {
                "episode_id": "e",
                "actions_executed": 10,
                "admitted_windows": 1,
                "initialization_steps": 10,
                "total_control_steps": 20,
            }
        )
    )
    manifest = {
        "schema": "reasoning-executed-replay-1",
        "source_protocol": {
            "teacher": {"frs_action_steering": False},
            "checkpoint": {"runtime_action_horizon": 10},
        },
        "instruction": "test task",
        "episodes": [{"directory": "rollout_0", "episode_id": "e"}],
        "admitted_windows": 1,
        "source_collection_control_steps": 20,
    }
    refresh_manifest(tmp_path, manifest)
    return manifest, oid


def refresh_manifest(directory, manifest):
    manifest["files"] = {
        str(p.relative_to(directory)): file_sha256(p)
        for p in directory.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))


def test_replay_recovers_actual_commands_and_own_pre_action_observation(tmp_path):
    _, oid = make_replay(tmp_path)
    windows, observations, manifest = load(
        tmp_path, file_sha256(tmp_path / "manifest.json")
    )
    assert len(windows) == len(observations) == 1
    assert digest(observations[oid]) == oid
    assert windows[0]["stages"] == ["continuation", "correction"]
    assert manifest["source_collection_control_steps"] == 20


@pytest.mark.parametrize("target", ["actions", "window", "observation"])
def test_replay_cannot_invent_a_tail_or_change_observation_even_after_rehash(
    tmp_path, target
):
    manifest, oid = make_replay(tmp_path)
    path = tmp_path / "rollout_0/admission.jsonl"
    admission = json.loads(path.read_text())
    if target == "actions":
        admission["steps"][-1]["action"][2] = 0.5
    elif target == "window":
        admission["windows"][0]["actions"][-1][2] = 0.5
    else:
        file = tmp_path / "rollout_0" / (oid + ".npz")
        with np.load(file, allow_pickle=False) as data:
            raw = {k: data[k] for k in data.files}
        raw["observation/state"][0] = 1
        np.savez_compressed(file, **raw)
    path.write_text(json.dumps(admission) + "\n")
    refresh_manifest(tmp_path, manifest)
    with pytest.raises(
        ValueError, match="execution evidence|recorded admission|observation identity"
    ):
        load(tmp_path)


def test_replay_requires_pinned_source_bytes_and_excludes_frs(tmp_path):
    manifest, _ = make_replay(tmp_path)
    original = file_sha256(tmp_path / "manifest.json")
    manifest["source_protocol"]["teacher"]["frs_action_steering"] = True
    refresh_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="manifest checksum"):
        load(tmp_path, original)
    with pytest.raises(ValueError, match="FRS"):
        load(tmp_path)


def test_masked_replay_keeps_real_tail_but_does_not_supervise_failed_steps(tmp_path):
    manifest, oid = make_replay(tmp_path)
    directory = tmp_path / "rollout_0"
    path = directory / "admission.jsonl"
    admission = json.loads(path.read_text())
    for step in admission["steps"][5:]:
        step["evidence"] = "failed"
    admission["windows"] = training_windows(admission["steps"], horizon=10)
    assert admission["windows"] == []
    path.write_text(json.dumps(admission))
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    result["admitted_windows"] = 0
    result_path.write_text(json.dumps(result))
    manifest["admission_rule"] = "mask_loss_by_evidence"
    refresh_manifest(tmp_path, manifest)
    windows, observations, _ = load(tmp_path)
    assert windows[0]["step_loss_mask"] == [True] * 5 + [False] * 5
    assert windows[0]["evidence"] == "observed_useful_masked"
    assert windows[0]["stages"] == ["correction"]
    assert len(windows[0]["actions"]) == 10
    assert digest(observations[oid]) == oid
    # The new rule still refuses changes to a real command, even if unsupervised.
    admission["steps"][-1]["action"][0] = 0.2
    path.write_text(json.dumps(admission))
    refresh_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="execution evidence"):
        load(tmp_path)


def test_masking_cannot_admit_unknown_tails_or_unapproved_corrections(tmp_path):
    make_replay(tmp_path)
    steps = json.loads((tmp_path / "rollout_0/admission.jsonl").read_text())["steps"]
    assert training_windows(steps[:5], horizon=10, mask_loss_by_evidence=True) == []
    steps[-1]["executed"] = False
    assert training_windows(steps, horizon=10, mask_loss_by_evidence=True) == []
    steps[-1]["executed"] = True
    for step in steps[:5]:
        step["preference"] = "uncertain"
    windows = training_windows(steps, horizon=10, mask_loss_by_evidence=True)
    assert windows[0]["step_loss_mask"] == [False] * 5 + [True] * 5
    assert windows[0]["stages"] == ["continuation"]
