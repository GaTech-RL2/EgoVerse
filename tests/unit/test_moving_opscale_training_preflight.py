"""Strict raw tracking, fresh norms, and quantitative real-update smoke gates."""

import copy
import json
import math
import random
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.fold_clothes_moving_opscale import run_preflight as p

COMMIT = "a" * 40
JOB = "123"
EPISODE = "selected-episode"


def stats_payload():
    return {
        "norm_run_metadata": {
            "training_tracking_policy": p.TRAINING_POLICY,
            "sampling_seed": 42026,
            "tracking_coverage": {
                EPISODE: {"eligible_starts": 3, "valid_starts": 2, "rejected_starts": 1}
            },
        }
    }


def evidence(cell="b24_op16", steps=2, start=0):
    return {
        "status": "passed",
        "training_tracking_policy": p.TRAINING_POLICY,
        "commit": COMMIT,
        "job_id": JOB,
        "cell": cell,
        "seed": 42026,
        "norm_stats_sha256": "d" * 64,
        "start_step": start,
        "expected_steps": steps,
        "batch_size": 64,
        "verified_samples": steps * 64,
        "loss_ceiling": 1000.0,
        "updates": [
            {
                "step": start + i + 1,
                "batch_size": 64,
                "loss": 1.25,
                "gradient_norm": 2.5,
            }
            for i in range(steps)
        ],
        "forced_resampling": {
            "episode": EPISODE,
            "requested_frame": 0,
            "actual_frame": 1,
            "requested_raw_valid": False,
            "actual_raw_valid": True,
            "same_episode": True,
        },
    }


def verify(value, steps=2, start=0):
    return p.verify_training_evidence(
        value,
        commit=COMMIT,
        job_id=JOB,
        cell="b24_op16",
        start_step=start,
        steps=steps,
        norm_sha256="d" * 64,
    )


def test_norm_metadata_requires_current_policy_seed_and_exact_training_coverage():
    payload = stats_payload()
    proof = p.training_norm_summary(payload, [EPISODE])
    report = {
        "status": "passed",
        "commit": COMMIT,
        "training_tracking_policy": p.TRAINING_POLICY,
        "training_validity": proof,
    }
    assert p.validate_training_norm(report, payload, [EPISODE], COMMIT) == proof
    for key, bad in [
        ("training_tracking_policy", None),
        ("sampling_seed", 42),
        ("tracking_coverage", {}),
        (
            "tracking_coverage",
            {EPISODE: {"eligible_starts": 3, "valid_starts": 0, "rejected_starts": 3}},
        ),
    ]:
        altered = copy.deepcopy(payload)
        altered["norm_run_metadata"][key] = bad
        with pytest.raises(RuntimeError):
            p.training_norm_summary(altered, [EPISODE])
    for altered in [
        {key: value for key, value in report.items() if key != "status"},
        {**report, "status": "running"},
        {**report, "status": "failed"},
        {**report, "commit": "b" * 40},
        {**report, "training_validity": {}},
        {**report, "training_tracking_policy": None},
    ]:
        with pytest.raises(RuntimeError):
            p.validate_training_norm(altered, payload, [EPISODE], COMMIT)
    with pytest.raises(RuntimeError):
        p.training_norm_summary(payload, [EPISODE, "missing"])


def test_full_update_evidence_checks_every_loss_gradient_batch_and_identity():
    assert verify(evidence())["samples"] == 128
    for key, value in [
        ("status", "running"),
        ("seed", 42),
        ("commit", "b" * 40),
        ("job_id", "124"),
        ("verified_samples", 127),
        ("batch_size", 63),
        ("loss_ceiling", math.inf),
        ("training_tracking_policy", None),
        ("norm_stats_sha256", "e" * 64),
        ("updates", []),
    ]:
        changed = evidence()
        changed[key] = value
        with pytest.raises(RuntimeError):
            verify(changed)
    for key, value in [
        ("loss", math.nan),
        ("loss", 1000.1),
        ("loss", -1),
        ("gradient_norm", math.inf),
        ("gradient_norm", -1),
        ("step", 9),
        ("batch_size", 63),
    ]:
        changed = evidence()
        changed["updates"][1][key] = value
        with pytest.raises(RuntimeError):
            verify(changed)
    changed = evidence()
    changed["forced_resampling"]["same_episode"] = False
    with pytest.raises(RuntimeError, match="same episode"):
        verify(changed)


class Leaf:
    _training_tracking_valid_mask = np.array([False, True, True])
    _training_tracking_valid_indices = np.array([1, 2])
    sample_indices = np.array([0, 1, 2])

    def __len__(self):
        return 3

    def evaluation_tracking_validity(self, frame):
        return bool(self._training_tracking_valid_mask[frame])

    def __getitem__(self, frame):
        random.random()
        np.random.rand()
        torch.rand(1)
        return {
            "episode_hash": EPISODE,
            "frame_idx": torch.tensor(1 if frame == 0 else frame),
        }


@pytest.fixture
def audit(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "lightning.pytorch", SimpleNamespace(Callback=object)
    )
    monkeypatch.setenv("PL_GLOBAL_SEED", "42026")
    stats = tmp_path / "norm_stats.json"
    stats.write_text(json.dumps(stats_payload()))
    dataset = SimpleNamespace(
        training_tracking_policy=p.TRAINING_POLICY,
        training_tracking_coverage=stats_payload()["norm_run_metadata"][
            "tracking_coverage"
        ],
        datasets={EPISODE: Leaf()},
    )
    trainer = SimpleNamespace(
        global_step=0,
        accumulate_grad_batches=1,
        datamodule=SimpleNamespace(train_datasets={"human_bimanual": dataset}),
    )
    module = SimpleNamespace(enable_grad_norm=True, grad_norm_history=deque([2.5]))
    callback = p.training_audit_callback(
        str(tmp_path / "evidence.json"),
        "b24_op16",
        COMMIT,
        JOB,
        str(stats),
        p.sha(stats),
        0,
        2,
    )
    return callback, trainer, module, dataset


def batch(frame=1, size=64):
    return {
        "human_bimanual": {
            "episode_hash": [EPISODE] * size,
            "frame_idx": torch.full((size,), frame, dtype=torch.long),
            "actions_keypoints": torch.zeros(size, 1, 1),
        }
    }


def test_callback_exercises_real_invalid_start_preserves_rng_and_proves_full_batches(
    audit,
):
    callback, trainer, module, _ = audit
    py, np_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state(),
    )
    callback.on_train_start(trainer, module)
    assert py == random.getstate()
    assert np.array_equal(np_state[1], np.random.get_state()[1])
    assert torch.equal(torch_state, torch.random.get_rng_state())
    for index in range(2):
        callback.on_train_batch_start(trainer, module, batch(), index)
        trainer.global_step += 1
        callback.on_train_batch_end(
            trainer, module, {"loss": torch.tensor(1.25)}, batch(), index
        )
    callback.on_train_end(trainer, module)
    saved = json.loads(callback.path.read_text())
    assert saved["status"] == "passed" and saved["verified_samples"] == 128
    assert saved["forced_resampling"]["requested_frame"] == 0
    assert saved["forced_resampling"]["actual_frame"] == 1
    assert saved["updates"] == evidence()["updates"]


@pytest.mark.parametrize("value", [batch(0), batch(1, 63), batch(5)])
def test_callback_rejects_untracked_or_incomplete_actual_batch(audit, value):
    callback, trainer, module, _ = audit
    callback.on_train_start(trainer, module)
    with pytest.raises(RuntimeError):
        callback.on_train_batch_start(trainer, module, value, 0)


def test_callback_records_large_loss_before_failing_and_never_passes(audit):
    callback, trainer, module, _ = audit
    callback.on_train_start(trainer, module)
    callback.on_train_batch_start(trainer, module, batch(), 0)
    trainer.global_step = 1
    with pytest.raises(RuntimeError, match="ceiling"):
        callback.on_train_batch_end(
            trainer, module, {"loss": torch.tensor(1e10)}, batch(), 0
        )
    callback.on_exception(trainer, module, RuntimeError())
    saved = json.loads(callback.path.read_text())
    assert saved["status"] == "failed" and saved["updates"][0]["loss"] > 1000


def test_smoke_gate_reopens_all_four_receipts_and_rejects_old_two_step_proof(tmp_path):
    report = {
        "status": "passed",
        "commit": COMMIT,
        "job_id": JOB,
        "training_tracking_policy": p.TRAINING_POLICY,
        "run_root": str(tmp_path),
        "runs": [],
        "training_diagnostic": {},
    }
    for cell, start, steps, epoch in [
        ("b24_op16", 0, 512, 1),
        ("b24_op4", 0, 64, 1),
        ("b24_op4", 64, 64, 2),
        ("b24_op128", 0, 64, 1),
    ]:
        normdir = tmp_path / "norm" / cell
        stats = normdir / "norm_stats/norm_stats.json"
        stats.parent.mkdir(parents=True, exist_ok=True)
        stats.write_text(json.dumps(stats_payload()))
        normproof = p.training_norm_summary(stats_payload(), [EPISODE])
        p.write(
            normdir / "norm_complete.json",
            {
                "status": "passed",
                "commit": COMMIT,
                "training_tracking_policy": p.TRAINING_POLICY,
                "training_validity": normproof,
                "stats_path": str(stats),
                "stats_sha256": p.sha(stats),
            },
        )
        p.write(
            tmp_path / "private/splits" / f"fold_moving_{cell}.json",
            {"train": [EPISODE]},
        )
        path = (
            tmp_path
            / "smoke"
            / COMMIT[:12]
            / f"attempt_{JOB}"
            / cell
            / f"training_evidence_{start}_{start + steps}.json"
        )
        value = evidence(cell, steps, start)
        value["norm_stats_sha256"] = p.sha(stats)
        p.write(path, value)
        summary = p.verify_training_evidence(
            value,
            commit=COMMIT,
            job_id=JOB,
            cell=cell,
            start_step=start,
            steps=steps,
            norm_sha256=p.sha(stats),
        )
        phase = {
            "operator_count": int(cell.split("op")[1]),
            "epochs": epoch,
            "checkpoint": {"step": start + steps},
            "normalization": normproof,
            "training_evidence": {"path": str(path), "sha256": p.sha(path), **summary},
        }
        if cell == "b24_op16":
            report["training_diagnostic"] = phase
        else:
            report["runs"].append(phase)
    assert len(p.validate_smoke_training(report, tmp_path, COMMIT)) == 4
    altered = copy.deepcopy(report)
    altered["training_diagnostic"] = {}
    with pytest.raises(RuntimeError, match="op16"):
        p.validate_smoke_training(altered, tmp_path, COMMIT)
    path = Path(report["runs"][0]["training_evidence"]["path"])
    value = json.loads(path.read_text())
    value["updates"] = value["updates"][:2]
    p.write(path, value)
    report["runs"][0]["training_evidence"]["sha256"] = p.sha(path)
    with pytest.raises(RuntimeError, match="incomplete"):
        p.validate_smoke_training(report, tmp_path, COMMIT)
