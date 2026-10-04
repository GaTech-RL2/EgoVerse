"""Single execution decisions and conservative, observation-aligned training data."""

import copy
import json
import os
from pathlib import Path

import numpy as np

from astra_reversal.records import digest


def append_record(path, record):
    """Durable append; this study uses one process per worker ledger."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class CandidateBatch:
    """Freeze policy/observation/rule/reference, then consume at most once.

    No environment is exposed to candidate generation or judging. A persisted
    claim is written BEFORE execution: an interrupted claim cannot be retried.
    """

    def __init__(self, *, observation_id, policy_version, rule, reference):
        self._binding = copy.deepcopy(
            {
                "observation_id": observation_id,
                "policy_version": policy_version,
                "rule": rule,
            }
        )
        self._candidates = {"native": self._array(reference)}
        self._binding["reference_sha256"] = digest(self._candidates["native"])
        self.batch_id = digest(self._binding)
        self._verdicts = {}
        self._claimed = False

    @staticmethod
    def _array(value):
        result = np.array(value, dtype=np.float32, copy=True)
        if result.ndim != 2 or result.shape[1] != 7 or not np.isfinite(result).all():
            raise ValueError("Candidate must be finite [H,7] controller commands")
        result.flags.writeable = False
        return result

    @property
    def binding(self):
        return copy.deepcopy(self._binding)

    def add(self, candidate_id, actions):
        if self._claimed or candidate_id in self._candidates:
            raise ValueError("Candidate ID already used or batch consumed")
        value = self._array(actions)
        if value.shape != self._candidates["native"].shape:
            raise ValueError("Candidate horizon differs from fixed native reference")
        self._candidates[candidate_id] = value

    def proposals(self):
        return {key: value.copy() for key, value in self._candidates.items()}

    def judge(self, candidate_id, *, binding, preference, rationale):
        if self._claimed or binding != self._binding or candidate_id == "native":
            raise ValueError("Judgment must use the unchanged reference and rule")
        if candidate_id not in self._candidates or candidate_id in self._verdicts:
            raise ValueError("Unknown or already judged candidate")
        if preference not in ("win", "tie", "loss", "uncertain"):
            raise ValueError("Unknown preference")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("Concise evidence/uncertainty rationale required")
        self._verdicts[candidate_id] = {
            "preference": preference,
            "rationale": rationale,
        }

    def claim(self, candidate_id, ledger_directory):
        if self._claimed or candidate_id not in self._candidates:
            raise ValueError("Batch consumed or candidate unknown")
        if (
            candidate_id != "native"
            and self._verdicts.get(candidate_id, {}).get("preference") != "win"
        ):
            raise ValueError(
                "Only a predicted clear win may replace the native proposal"
            )
        directory = Path(ledger_directory)
        directory.mkdir(parents=True, exist_ok=True)
        record = {
            "batch_id": self.batch_id,
            "binding": self.binding,
            "selected": candidate_id,
            "judgments": copy.deepcopy(self._verdicts),
            "candidate_hashes": {k: digest(v) for k, v in self._candidates.items()},
            "status": "claimed_before_execution",
            "physical_success": None,
        }
        with (directory / (self.batch_id + ".json")).open("x") as stream:
            json.dump(record, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self._claimed = True
        return self._candidates[candidate_id].copy(), record


def training_windows(steps, *, horizon, stride=5, mask_loss_by_evidence=False):
    """Complete actual action windows only. Unknown tails are never zero-filled.

    Each step supplies its OWN pre-action observation ID. Admission requires
    local observed usefulness; correction steps also require the pre-execution
    relative-improvement gate. Episode success does not admit every step.
    """
    if type(horizon) is not int or horizon < 1 or type(stride) is not int or stride < 1:
        raise ValueError("Positive integer horizon and stride required")
    result = []
    for start in range(0, len(steps) - horizon + 1, stride):
        window = steps[start : start + horizon]
        first = window[0]
        if any(
            row["episode_id"] != first["episode_id"] or row["step"] != first["step"] + i
            for i, row in enumerate(window)
        ):
            continue
        if not all(row.get("executed") for row in window):
            continue
        useful = [
            row.get("evidence") == "observed_useful"
            and row.get("stage") in ("setup", "correction", "continuation")
            and (row["stage"] != "correction" or row.get("preference") == "win")
            for row in window
        ]
        if not (any(useful) if mask_loss_by_evidence else all(useful)):
            continue
        actions = np.asarray([row["action"] for row in window], dtype=np.float32)
        if actions.shape != (horizon, 7) or not np.isfinite(actions).all():
            raise ValueError("Malformed executed commands")
        admitted = [row for row, good in zip(window, useful, strict=True) if good]
        record = {
            "episode_id": first["episode_id"],
            "observation_id": first["observation_id"],
            "start_step": first["step"],
            "end_step_exclusive": first["step"] + horizon,
            "event_ids": sorted(
                {row["event_id"] for row in admitted if row.get("event_id")}
            ),
            "stages": sorted({row["stage"] for row in admitted}),
            "evidence": "observed_useful_masked"
            if mask_loss_by_evidence
            else "observed_useful",
            "source": "executed_commands",
            "actions": actions.tolist(),
            "policy_versions": sorted({row["policy_version"] for row in window}),
        }
        if mask_loss_by_evidence:
            # All actions are real and known. Unknown/failed usefulness masks
            # the supervised loss; it never fills an unexecuted future tail.
            record["step_loss_mask"] = useful
        record["window_id"] = digest(record)
        result.append(record)
    return result


def balanced_window_indices(windows, count, rng):
    """Sample event/stage groups before windows; a long event cannot dominate."""
    if not windows or type(count) is not int or count < 1:
        raise ValueError("Nonempty windows and positive update count required")
    groups = {}
    for index, row in enumerate(windows):
        key = (row["episode_id"], tuple(row["event_ids"]), tuple(row["stages"]))
        groups.setdefault(key, []).append(index)
    keys = list(groups)
    weights = np.array([2.0 if "correction" in key[2] else 1.0 for key in keys])
    weights /= weights.sum()
    return [
        int(rng.choice(groups[keys[int(rng.choice(len(keys), p=weights))]]))
        for _ in range(count)
    ]
