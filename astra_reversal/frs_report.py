"""Aggregate recorded FRS tasks and independent audits without model calls.

Inputs are downloaded worker directories, or explicit task_N directories whose
parent retains the worker metadata. Explicit task selection permits whole-task
recovery across workflows; duplicate tasks are rejected. Audit arguments are
task receipts, not substitute outcome records. Partial inputs retain known
physical costs and suppress new efficacy claims.

python -m astra_reversal.frs_report --inputs WORKER... --audits TASK_AUDIT...
    --phase evaluation --require-complete --output NEW_DIRECTORY
"""

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

from .frs_agent import ROLES, summarize_calls
from .frs_html_report import (
    SCHEMA_VERSION,
    TASKS,
    TOKEN_FIELDS,
    _sha,
    _sum_usage,
    _usage,
    derived_results,
    render_report,
    require,
    validate_report,
)
from .records import digest, file_sha256

PRODUCER_VERSION = "frs-aggregate-1.0"
AUDIT_VERSION = "frs-artifact-audit-1.0"
METHODS = {
    "native_euler10": (
        "Native Euler 10",
        "Frozen pi05 with independent full Gaussian noise, raw policy images and no Astra calls.",
    ),
    "native_repeated_noise": (
        "Native repeated noise",
        "Frozen pi05 with a repeated seven-dimensional control-noise vector and fresh Gaussian padding; matched temporal correlation control.",
    ),
    "astra_direction_direct": (
        "Astra direction, direct execution",
        "Astra coarse directions are converted to actions and executed directly; fine or rejected calls use the current native prediction.",
    ),
    "astra_frs": (
        "Astra flow reversal steering",
        "Astra coarse references are reversed and denoised with Euler 10; the finite-step flow and refreshed padding are deliberate.",
    ),
    "critique_frs_no_learning": (
        "Critique and FRS, no learning",
        "Three same-reset action-edit rounds retain only Astra-promoted rules; frozen native weights and no auxiliary fitting. Evaluation uses the final retained rules.",
    ),
    "learned_noise": (
        "Learned auxiliary noise policy",
        "The frozen VLA is driven by the saved task-specific noise actor without online Astra edits. Each round's checkpoint is evaluated on separate resets.",
    ),
    "critique_frs_learning": (
        "Critique and FRS with auxiliary learning",
        "Three same-reset action-edit rounds use the preceding auxiliary checkpoint. Only Astra-better rollouts supply executed repeated-noise labels for auxiliary fitting.",
    ),
}
REQUIRED_VALIDATIONS = {
    "complete_physical_coverage",
    "all_array_hashes",
    "native_weight_bytes_unchanged",
    "separated_reset_streams",
    "exact_request_provider_bindings",
    "calibrated_direction_and_action_transforms",
    "executed_loop_noise_repeated",
    "training_labels_only_Astra_better_adaptation",
    "no_heldout_feedback_or_labels",
    "checkpoint_optimizer_replay_bindings",
}


def _json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _json_line(text):
    def pairs(items):
        row = {}
        for key, value in items:
            require(key not in row, "Duplicate JSON key in recording")
            row[key] = value
        return row

    def nonfinite(_):
        raise ValueError("Nonfinite JSON value in recording")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)


def _compact(value):
    if isinstance(value, dict):
        if set(value) == {"array", "shape", "dtype", "sha256"}:
            return {key: item for key, item in value.items() if key != "array"}
        return {key: _compact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


def _compact_usage(records):
    recorded = summarize_calls(records)
    value = {
        "calls": recorded["provider_calls"],
        "accepted_calls": recorded["accepted_proposals"],
        "failed_calls": recorded["failed_calls"],
        "preflight_failures": recorded["preflight_failures"],
        "tokens": {
            field: {
                key: recorded["tokens"][field][key] for key in ("sum", "missing_calls")
            }
            for field in TOKEN_FIELDS
        },
    }
    _usage(value)
    return value


def _same(actual, expected, label):
    require(actual == expected, label)


def _assignment(phase, worker):
    require(type(worker) is int, "Worker index must be an integer")
    if phase == "development" and worker in range(3):
        return {
            "suite": "libero_goal_ood" if worker == 0 else "libero_spatial_ood",
            "task_ids": [[6], [2], [8]][worker],
        }
    if phase == "evaluation" and worker in range(8):
        return {
            "suite": ("libero_goal_ood", "libero_spatial_ood")[worker // 4],
            "task_ids": list(range(worker % 4, 10, 4)),
        }
    raise ValueError("Unexpected worker assignment")


def _expected_tasks(phase, protocol):
    if phase == "evaluation":
        return set(TASKS)
    return {
        f"{suite}:{task}"
        for suite, tasks in protocol["development_cases"].items()
        for task in tasks
    }


def _episode(task, seed, state):
    suite, task_id = task.split(":")
    return f"{suite}:seed{seed}:task{task_id}:state{state}"


class _Files:
    def __init__(self, root):
        self.root = Path(root)
        self.hashes = {}

    def read(self, name, *, lines=False, optional=False):
        path = self.root / name
        if optional and not path.exists():
            return None
        require(
            path.is_file() and not path.is_symlink(),
            f"Missing or symlinked recording: {name}",
        )
        raw = path.read_bytes()
        self.hashes[name] = hashlib.sha256(raw).hexdigest()
        if lines:
            return [_json_line(row) for row in raw.decode().splitlines() if row.strip()]
        return _json_line(raw.decode())

    def stable(self):
        for name, checksum in self.hashes.items():
            require(
                file_sha256(self.root / name) == checksum,
                "Inputs changed during report construction; use a stable snapshot",
            )


def _receipts(values):
    result = {}
    iterable = values.values() if isinstance(values, dict) else (values or [])
    for item in iterable:
        if isinstance(item, (str, Path)):
            path = Path(item)
            raw = path.read_bytes()
            value = _json_line(
                gzip.decompress(raw).decode() if path.suffix == ".gz" else raw.decode()
            )
            source_hash = hashlib.sha256(raw).hexdigest()
            encoding = "gzip" if path.suffix == ".gz" else "json"
        else:
            value = copy.deepcopy(item)
            source_hash = hashlib.sha256(_json_bytes(value)).hexdigest()
            encoding = "canonical_json_from_mapping"
        require(
            value.get("schema_version") == AUDIT_VERSION
            and value.get("status") == "passed",
            "Supply only passed FRS task audit receipts",
        )
        key = f"{value['suite']}:{value['task_id']}"
        require(key not in result, "Duplicate task audit receipt")
        require(
            REQUIRED_VALIDATIONS <= value.get("validations", {}).keys()
            and all(
                value["validations"][name] is True for name in REQUIRED_VALIDATIONS
            ),
            "Audit receipt lacks a required passed validation",
        )
        result[key] = (
            value,
            {
                "sha256": source_hash,
                "encoding": encoding,
                "content_sha256": hashlib.sha256(_json_bytes(value)).hexdigest(),
            },
        )
    return result


def _worker(path, phase):
    files = _Files(path)
    names = (
        "runtime.json",
        "protocol.json",
        "frozen_plan.json",
        "reset_manifest.json",
        "checkpoint.json",
        "prompts.json",
        "frozen_weights_before.json",
    )
    values = {name: files.read(name) for name in names}
    values["frozen_weights_after.json"] = files.read(
        "frozen_weights_after.json", optional=True
    )
    runtime, protocol = values["runtime.json"], values["protocol.json"]
    require(
        runtime["phase"] == phase
        and runtime["tf32"] is False
        and "L40S" in runtime["gpu"],
        "Worker phase/runtime differs from frozen experiment",
    )
    target = _assignment(phase, runtime["worker"])
    _same(
        runtime["assignment"], target, "Worker assignment differs from fixed partition"
    )
    require(
        protocol["schema_version"] == "frs-policy-improvement-1.0"
        and protocol["rounds"] == 3
        and protocol["evaluation_states"] == list(range(1, 11))
        and protocol["adaptation_state"] == 0
        and protocol["execute_steps"] == 10
        and protocol["action_budget"] == 300,
        "Unsupported FRS task/reset/round protocol",
    )
    require(
        protocol["seed"]
        == (protocol["development_seed"] if phase == "development" else 43),
        "Unexpected phase seed",
    )
    require(
        protocol["development_seed"] == 19
        and protocol["development_cases"]
        == {"libero_goal_ood": [6], "libero_spatial_ood": [2, 8]},
        "Development task/reset cohort differs from the frozen protocol",
    )
    require(
        set(protocol["evaluation_methods"]) == set(METHODS) - {"critique_frs_learning"}
        and protocol["adaptation_methods"]
        == ["critique_frs_no_learning", "critique_frs_learning"],
        "Frozen method inventory differs",
    )
    _same(
        protocol["solver"],
        {"solver": "euler", "steps": 10, "time_power": 1.0},
        "Frozen sampler differs",
    )
    plan, manifest = values["frozen_plan.json"], values["reset_manifest.json"]
    _same(plan["runtime"], runtime, "Frozen plan runtime differs")
    require(
        plan["protocol_sha256"] == digest(protocol), "Frozen protocol digest differs"
    )
    require(
        manifest["sha256"]
        == plan["manifest_sha256"]
        == digest({key: value for key, value in manifest.items() if key != "sha256"}),
        "Full reset manifest digest differs",
    )
    expected = {
        _episode(f"{target['suite']}:{task}", protocol["seed"], state)
        for task in target["task_ids"]
        for state in range(11)
    }
    entries = manifest["episodes"]
    require(
        len(entries) == len(expected)
        and {row["episode_id"] for row in entries} == expected,
        "Worker manifest reset coverage differs",
    )
    _same(
        plan["assigned_episodes"],
        [row["episode_id"] for row in entries],
        "Frozen assigned episode order differs",
    )
    _same(plan["assigned_tasks"], target["task_ids"], "Frozen assigned task IDs differ")
    before, after = (
        values["frozen_weights_before.json"],
        values["frozen_weights_after.json"],
    )
    require(
        before["tensors"] and before["sha256"] == digest(before["tensors"]),
        "Initial native tensor receipt is invalid",
    )
    if after is not None:
        _same(after, before, "Native tensor bytes changed during worker execution")
    prompts = values["prompts.json"]
    require(
        set(prompts["roles"]) == set(ROLES),
        "Exact prompt manifest omits or adds a role",
    )
    for row in prompts["roles"].values():
        require(
            isinstance(row["system_prompt"], str)
            and hashlib.sha256(row["system_prompt"].encode()).hexdigest()
            == row["sha256"],
            "Exact prompt text/hash differs",
        )
    return {
        "files": files,
        "values": values,
        "runtime": runtime,
        "protocol": protocol,
        "target": target,
        "entries": entries,
        "native_sha256": before["sha256"],
    }


def _schedule(protocol, development):
    expected = {("native_repeated_noise", 0, 0)}
    expected.update(
        (method, 0, index)
        for method in protocol["adaptation_methods"]
        for index in range(1, 4)
    )
    states = (
        protocol["evaluation_states"][:1]
        if development
        else protocol["evaluation_states"]
    )
    expected.update(
        (method, state, index)
        for method in protocol["evaluation_methods"]
        for state in states
        for index in (range(1, 4) if method == "learned_noise" else [3])
    )
    return expected


def _checkpoint_metadata(checkpoint):
    require(
        checkpoint["schema_version"] == "frs-noise-checkpoint-1.0",
        "Unknown auxiliary checkpoint schema",
    )
    _sha(checkpoint["manifest_sha256"], "Auxiliary manifest SHA")
    _sha(checkpoint["files"]["state.pt"]["sha256"], "Auxiliary state file SHA")
    value = checkpoint["policy"]
    require(
        value["state_id"]
        == digest({key: item for key, item in value.items() if key != "state_id"}),
        "Auxiliary state identity differs",
    )
    for key in (
        "parameter_sha256",
        "optimizer_sha256",
        "replay_sha256",
        "history_sha256",
    ):
        _sha(value[key], f"Auxiliary {key}")
    require(
        value["synthetic_cpu_test_only"] is False
        and value["test_only_cpu_steps"] is None,
        "A synthetic auxiliary checkpoint cannot be reported as a native experiment",
    )
    return value


class _Task:
    def __init__(self, directory, worker, receipt):
        self.files, self.worker = _Files(directory), worker
        self.summary = self.files.read("summary.json")
        self.events = self.files.read("events.jsonl", lines=True)
        self.providers = (
            self.files.read("provider.jsonl", lines=True, optional=True) or []
        )
        self.protocol = worker["protocol"]
        self.key = f"{self.summary['suite']}:{self.summary['task_id']}"
        self.complete = self.summary["status"] == "complete"
        self.receipt, self.receipt_source = receipt if receipt else (None, None)
        self.used_calls, self.overheads, self.learning = set(), [], []
        self.physical, self.starts, self.ends = {}, {}, {}
        self.rounds, self.checkpoints = {}, {}
        require(
            self.summary["schema_version"] == "frs-task-1.0"
            and self.summary["status"] in ("running", "complete"),
            "Unknown task recording schema/status",
        )
        require(
            self.key in TASKS and self.summary["instruction"] == TASKS[self.key],
            "Task instruction differs from the released inventory",
        )
        require(
            self.summary["development"]
            == (worker["runtime"]["phase"] == "development"),
            "Task phase differs from worker",
        )
        require(
            self.summary["suite"] == worker["target"]["suite"]
            and self.summary["task_id"] in worker["target"]["task_ids"],
            "Task was not assigned to this worker",
        )
        require(
            [row["sequence"] for row in self.events] == list(range(len(self.events))),
            "Task event sequence is incomplete or duplicated",
        )
        tasks = [row for row in self.events if row["kind"] == "task"]
        require(
            len(tasks) == 1 and tasks[0]["sequence"] == 0,
            "Missing unique initial task event",
        )
        task = tasks[0]
        self.entries = {row["episode_id"]: row for row in task["entries"]}
        prescribed_entries = [
            row
            for row in worker["entries"]
            if row["task_id"] == self.summary["task_id"]
        ]
        _same(
            task["entries"],
            prescribed_entries,
            "Task reset entries differ from the full frozen manifest",
        )
        _same(task["protocol"], self.protocol, "Task/worker protocol differs")
        _same(
            task["checkpoint"],
            worker["values"]["checkpoint.json"],
            "Task/worker native checkpoint differs",
        )
        _same(
            self.summary["checkpoint"],
            task["checkpoint"],
            "Summary native checkpoint differs",
        )
        require(
            self.summary["entries_sha256"] == digest(task["entries"])
            and self.summary["protocol_sha256"] == digest(self.protocol),
            "Task input digest differs",
        )
        self._events()
        self._physical()
        self._adaptation()
        self._ledger_tail()
        self._costs()
        self._seal()
        self._receipt()

    def _events(self):
        allowed = {
            "task",
            "noise_policy_initial",
            "rollout_start",
            "rollout_end",
            "generation",
            "native_parity",
            "guide_probe",
            "astra_decision",
            "adaptation_round",
        }
        require(
            all(row["kind"] in allowed for row in self.events),
            "Unknown task event kind",
        )
        decisions = [row for row in self.events if row["kind"] == "astra_decision"]
        self.decisions = {}
        for decision in decisions:
            index = decision["provider_record_index"]
            require(
                type(index) is int
                and 0 <= index < len(self.providers)
                and index not in self.decisions,
                "Invalid or duplicate decision/provider index",
            )
            provider = self.providers[index]
            require(
                decision["request_index"] == provider["request_index"] == index + 1
                and decision["role"] == provider["role"]
                and decision["request_fingerprint"] == provider["request_fingerprint"],
                "Decision/provider identity differs",
            )
            require(
                (decision["response"] is not None) == provider["accepted"],
                "Decision/provider acceptance differs",
            )
            self.decisions[index] = decision
        require(
            [row["request_index"] for row in self.providers]
            == list(range(1, len(self.providers) + 1)),
            "Provider ledger request order differs",
        )
        for provider in self.providers:
            require(
                provider["episode_id"] in self.entries and provider["role"] in ROLES,
                "Provider task/role differs",
            )
            require(
                provider["requested_model"] == self.protocol["astra"]["model"]
                and provider["prompt_template_version"]
                == self.worker["values"]["prompts.json"]["prompt_template_version"],
                "Provider model/prompt version differs",
            )
            if provider["provider_call"]:
                require(
                    provider["system_prompt_sha256"]
                    == self.worker["values"]["prompts.json"]["roles"][provider["role"]][
                        "sha256"
                    ],
                    "Provider exact system prompt hash differs",
                )
            if provider["accepted"]:
                require(
                    provider["response"]["model"] == self.protocol["astra"]["model"],
                    "Accepted call returned another model",
                )
        if self.complete:
            require(
                set(self.decisions) == set(range(len(self.providers))),
                "Complete ledger contains an unrecorded decision",
            )
        self.provider_usage = summarize_calls(self.providers)
        initial = [row for row in self.events if row["kind"] == "noise_policy_initial"]
        require(len(initial) <= 1, "Duplicate initial auxiliary checkpoint")
        if initial:
            checkpoint = initial[0]["checkpoint"]
            _same(
                checkpoint,
                self.summary.get("initial_noise_policy_checkpoint"),
                "Initial auxiliary checkpoint summary/event differs",
            )
            self.checkpoints[0] = self._checkpoint("noise_policy_initial", checkpoint)
        if self.complete:
            require(
                0 in self.checkpoints,
                "Completed task lacks its initial auxiliary checkpoint",
            )
        for event in self.events:
            if event["kind"] == "rollout_start":
                require(
                    event["attempt_id"] not in self.starts,
                    "Duplicate physical rollout start",
                )
                self.starts[event["attempt_id"]] = event
            elif event["kind"] == "rollout_end":
                name = event["result"]["attempt_id"]
                require(name not in self.ends, "Duplicate physical rollout end")
                self.ends[name] = event
            elif event["kind"] == "adaptation_round":
                key = (event["method"], event["round_index"])
                require(key not in self.rounds, "Duplicate adaptation round")
                self.rounds[key] = event
                if event["method"] == "critique_frs_learning":
                    self.checkpoints[event["round_index"]] = self._checkpoint(
                        f"noise_policy_round{event['round_index']}", event["checkpoint"]
                    )

    def _checkpoint(self, name, checkpoint):
        metadata = _checkpoint_metadata(checkpoint)
        require(
            metadata["task_id"] == self.key,
            "Auxiliary checkpoint belongs to another task",
        )
        path = f"{name}/manifest.json"
        if (self.files.root / path).exists():
            actual = self.files.read(path)
            require(
                self.files.hashes[path] == checkpoint["manifest_sha256"],
                "Auxiliary manifest file hash differs",
            )
            _same(
                actual,
                {
                    key: value
                    for key, value in checkpoint.items()
                    if key != "manifest_sha256"
                },
                "Auxiliary manifest metadata differs",
            )
        elif self.receipt is not None:
            require(
                self.receipt["input_file_sha256"].get(path)
                == checkpoint["manifest_sha256"],
                "Audited auxiliary manifest binding is missing",
            )
        return checkpoint

    def _physical(self):
        expected = _schedule(self.protocol, self.summary["development"])
        observed, resets = set(), {}
        summary_rows = {
            row["attempt_id"]: row for row in self.summary["physical_rollouts"]
        }
        require(
            len(summary_rows) == len(self.summary["physical_rollouts"]),
            "Summary repeats a physical run",
        )
        for name, end in self.ends.items():
            require(name in self.starts, "Physical rollout ended without a start")
            start, result = self.starts[name], end["result"]
            require(
                start["sequence"] < end["sequence"],
                "Physical rollout event order differs",
            )
            entry = self.entries[result["episode_id"]]
            key = (result["method"], entry["initial_state_id"], result["round_index"])
            require(
                key in expected and key not in observed,
                "Unexpected or duplicate physical method/reset/round",
            )
            observed.add(key)
            require(
                name == f"{key[0]}_state{key[1]}_round{key[2]}"
                and result["evaluation"] == (key[1] != 0),
                "Physical attempt identity differs",
            )
            for field in (
                "attempt_id",
                "method",
                "episode_id",
                "round_index",
                "evaluation",
            ):
                _same(
                    start[field], result[field], f"Physical start/end {field} differs"
                )
            require(
                start["reset_entry_sha256"] == digest(entry),
                "Rollout reset entry binding differs",
            )
            compact = {
                key: value
                for key, value in result.items()
                if key not in ("snapshots", "executed_actions", "video_path")
            }
            compact["physical_run_id"] = f"{entry['episode_id']}:{name}"
            if name in summary_rows:
                _same(
                    compact,
                    summary_rows[name],
                    "Physical summary differs from rollout_end evidence",
                )
            else:
                require(not self.complete, "Completed summary omitted a physical run")
            reset = result["reset_audit"]
            require(
                reset["sha256"]
                == digest(
                    {key: value for key, value in reset.items() if key != "sha256"}
                ),
                "Rollout reset audit digest differs",
            )
            for field in (
                "episode_id",
                "seed",
                "reset_state_sha256",
                "reset_model_sha256",
                "bddl_sha256",
            ):
                _same(
                    reset[field],
                    entry[field],
                    f"Reset {field} differs from frozen entry",
                )
            if result["episode_id"] in resets:
                _same(
                    reset,
                    resets[result["episode_id"]],
                    "Paired methods/rounds used different reset observations or model states",
                )
            resets[result["episode_id"]] = reset
            require(
                type(result["success"]) is bool
                and type(result["initial_success"]) is bool
                and type(result["actions_executed"]) is int
                and 0 <= result["actions_executed"] <= 300,
                "Invalid physical outcome",
            )
            require(
                result["initial_success"] == reset["initial_success"]
                and result["zero_action_success"]
                == (result["success"] and result["actions_executed"] == 0),
                "Initial/zero-action success flag differs",
            )
            require(
                result["execute_steps"] == 10 and result["action_budget"] == 300,
                "Physical action budget differs",
            )
            indexes = result["provider_record_indexes"]
            require(
                all(type(i) is int and 0 <= i < len(self.providers) for i in indexes)
                and indexes == sorted(set(indexes))
                and not self.used_calls.intersection(indexes),
                "Duplicate or invalid rollout provider attribution",
            )
            for index in indexes:
                row = self.providers[index]
                require(
                    row["attempt_id"] == name
                    and row["episode_id"] == result["episode_id"]
                    and row["role"] in ("paper_direction", "action_edit"),
                    "Rollout provider scope differs",
                )
            _same(
                result["provider_usage"],
                summarize_calls([self.providers[i] for i in indexes]),
                "Rollout provider usage differs from original ledger",
            )
            self.used_calls.update(indexes)
            self.physical[name] = compact
        require(
            set(summary_rows) <= self.ends.keys(),
            "Summary physical run has no rollout_end event",
        )
        if self.complete:
            require(
                observed == expected and self.starts.keys() == self.ends.keys(),
                "Completed task lacks the full physical rollout schedule",
            )
        evaluation = self.summary["evaluation"]
        require(
            len({row["physical_run_id"] for row in evaluation}) == len(evaluation),
            "Duplicate evaluation reference",
        )
        for row in evaluation:
            require(
                row["attempt_id"] in self.physical,
                "Evaluation index names no completed physical run",
            )
            actual = self.physical[row["attempt_id"]]
            require(
                all(actual[key] == value for key, value in row.items()),
                "Evaluation reference differs from physical outcome",
            )
        if self.complete:
            require(
                {row["attempt_id"] for row in evaluation}
                == {name for name, row in self.physical.items() if row["evaluation"]},
                "Evaluation index is incomplete",
            )

    def _overhead(self, label, indexes, *, method=None, index=None, update=None):
        require(
            not self.used_calls.intersection(indexes),
            "Provider overhead was counted twice",
        )
        self.used_calls.update(indexes)
        records = [self.providers[i] for i in indexes]
        update = update or {}
        row = {
            "id": f"{self.key}:{label}",
            "label": f"{self.key}: {label}",
            "provider_usage": _compact_usage(records),
            "velocity_evaluations": 0,
            "training_steps": update.get("optimizer_updates", 0),
            "wall_seconds": sum(r["latency_seconds"] for r in records)
            + update.get("wall_seconds", 0.0),
            "source_sha256": self.files.hashes["events.jsonl"],
            "provider_record_indexes": indexes,
        }
        if method is not None:
            row["attribution"] = {
                "cohort_id": "adaptation",
                "episode_id": _episode(self.key, self.protocol["seed"], 0),
                "method_id": method,
                "round_index": index,
            }
        self.overheads.append(row)

    def _adaptation(self):
        baseline_id = "native_repeated_noise_state0_round0"
        for method in self.protocol["adaptation_methods"]:
            arm = self.summary["adaptation"].get(method)
            if arm is None:
                require(not self.complete, "Completed task omitted an adaptation arm")
                continue
            require(
                arm["baseline_attempt_id"] == baseline_id
                and baseline_id in self.physical,
                "Adaptation shared baseline differs",
            )
            reported = {row["round_index"]: row for row in arm["rounds"]}
            require(
                len(reported) == len(arm["rounds"]),
                "Duplicate summarized adaptation round",
            )
            indexes = sorted(index for name, index in self.rounds if name == method)
            require(
                indexes == list(range(1, len(indexes) + 1)),
                "Adaptation rounds are missing or reordered",
            )
            require(
                set(reported) <= set(indexes),
                "Summarized adaptation round has no event",
            )
            if self.complete:
                require(
                    indexes == [1, 2, 3] and set(reported) == set(indexes),
                    "Completed task omits an adaptation revision",
                )
            latest = incumbent = baseline_id
            first_success = [self.physical[baseline_id]["success"]]
            for index in indexes:
                event = self.rounds[(method, index)]
                compact = _compact(
                    {
                        key: value
                        for key, value in event.items()
                        if key not in ("kind", "sequence", "timestamp", "method")
                    }
                )
                if index in reported:
                    _same(
                        compact,
                        reported[index],
                        "Adaptation event differs from summary",
                    )
                name = f"{method}_state0_round{index}"
                require(
                    event["candidate_attempt_id"] == name
                    and event["incumbent_attempt_id"] == incumbent
                    and name in self.physical,
                    "Adaptation candidate/incumbent differs",
                )
                start, end = self.starts[name], self.ends[name]
                critique = [
                    i
                    for i, d in self.decisions.items()
                    if i not in self.used_calls
                    and d["role"] == "critique"
                    and self.ends[latest]["sequence"]
                    < d["sequence"]
                    < start["sequence"]
                ]
                judge = [
                    i
                    for i, d in self.decisions.items()
                    if i not in self.used_calls
                    and d["role"] == "judge"
                    and end["sequence"] < d["sequence"] < event["sequence"]
                ]
                require(
                    len(critique) == len(judge) == 1,
                    "Missing or ambiguous per-round critique/judge overhead",
                )
                _same(
                    event["critique"],
                    self.decisions[critique[0]]["response"],
                    "Critique response differs from accepted decision",
                )
                _same(
                    event["judge"],
                    self.decisions[judge[0]]["response"],
                    "Judge response differs from accepted decision",
                )
                promoted = (
                    event["critique"] is not None
                    and event["judge"] is not None
                    and event["judge"]["verdict"] == "better"
                )
                require(
                    event["promoted"] == promoted,
                    "Replay promotion differs from Astra better-only rule",
                )
                update = event["update"]
                if update is not None:
                    require(
                        method == "critique_frs_learning" and promoted,
                        "An unpromoted/no-learning round has a training receipt",
                    )
                    require(
                        update["status"] in ("updated", "not_updated"),
                        "Unknown training update status",
                    )
                    if update["status"] == "updated":
                        require(
                            update["optimizer_updates"] == 1000
                            and update["judge_sha256"] == digest(event["judge"])
                            and update["rollout_id"] == name,
                            "Training update is not bound to the accepted candidate",
                        )
                elif promoted and method == "critique_frs_learning":
                    require(
                        not self.complete,
                        "Promoted learning round has no update receipt",
                    )
                for role, call_indexes in (("critique", critique), ("judge", judge)):
                    self._overhead(
                        f"{method}:round{index}:{role}",
                        call_indexes,
                        method=method,
                        index=index,
                    )
                if update is not None:
                    self._overhead(
                        f"{method}:round{index}:training",
                        [],
                        method=method,
                        index=index,
                        update=update,
                    )
                if promoted:
                    incumbent = name
                latest = name
                first_success.append(self.physical[name]["success"])
                self.learning.append(
                    {
                        "task_key": self.key,
                        "method_id": method,
                        "round_index": index,
                        "promoted": promoted,
                        "judge_verdict": event["judge"]["verdict"]
                        if event["judge"]
                        else None,
                        "critique_accepted": event["critique"] is not None,
                        "candidate_attempt_id": name,
                        "incumbent_attempt_id": event["incumbent_attempt_id"],
                        "candidate_recorded_success": self.physical[name]["success"],
                        "candidate_credited_success": self.credited(
                            self.physical[name]
                        ),
                        "rules": event["rules"],
                        "checkpoint": copy.deepcopy(event["checkpoint"]),
                        "training": _compact(update),
                        "critique_provider_index": critique[0],
                        "judge_provider_index": judge[0],
                    }
                )
            if self.complete:
                first = next(
                    (i for i, success in enumerate(first_success) if success), None
                )
                _same(
                    arm["first_success"],
                    {
                        "first_success_round": first,
                        "censored": first is None,
                        "observed_rounds": 3,
                        "success_by_round": [
                            any(first_success[: i + 1]) for i in range(4)
                        ],
                    },
                    "Runtime first-success index differs from recorded outcomes",
                )

    def _ledger_tail(self):
        missing = sorted(set(range(len(self.providers))) - self.used_calls)
        if self.complete:
            require(not missing, "Complete task has unattributed provider cost")
        elif missing:
            self._overhead("unassigned_partial_provider_calls", missing)

    def _costs(self):
        updates = [
            row["training"]
            for row in self.learning
            if row["training"] and row["training"]["status"] == "updated"
        ]
        self.cost = {
            "rollouts": len(self.physical),
            "actions": sum(row["actions_executed"] for row in self.physical.values()),
            "velocity_evaluations": sum(
                row["counters"]["velocity_evaluations"]
                for row in self.physical.values()
            ),
            "rollout_wall_seconds": sum(
                row["wall_seconds"] for row in self.physical.values()
            ),
            "provider_calls": self.provider_usage["provider_calls"],
            "client_attempts": len(self.providers),
            "accepted_policy_updates": len(updates),
            "optimizer_steps": sum(row["optimizer_updates"] for row in updates),
            "training_seconds": sum(row["wall_seconds"] for row in updates),
            "auxiliary_inferences": sum(
                row["counters"]["auxiliary_inferences"]
                for row in self.physical.values()
            ),
            "auxiliary_request_seconds": sum(
                row["counters"]["auxiliary_request_seconds"]
                for row in self.physical.values()
            ),
        }
        if self.complete:
            _same(
                self.summary["provider_usage"],
                self.provider_usage,
                "Summary all-call provider ledger differs",
            )
            for key, value in self.cost.items():
                actual = self.summary["physical_cost"][key]
                require(
                    math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-9)
                    if isinstance(value, float)
                    else actual == value,
                    f"Physical summary cost differs: {key}",
                )

    def _seal(self):
        self.seal = self.files.read("completion_receipt.json", optional=True)
        if self.seal is None:
            return
        require(
            self.complete and self.seal["schema_version"] == "frs-completed-task-1.0",
            "Task seal does not identify a completed task",
        )
        runtime = self.worker["runtime"]
        require(
            self.seal["task_id"] == self.summary["task_id"]
            and self.seal["workflow"] == runtime["workflow"]
            and self.seal["worker"] == runtime["worker"],
            "Task seal belongs to another workflow/task",
        )
        after = self.files.read("frozen_weights_after.json")
        _same(
            after,
            self.worker["values"]["frozen_weights_before.json"],
            "Native tensor bytes changed during sealed task",
        )
        require(
            self.seal["native_tensor_sha256"] == self.worker["native_sha256"],
            "Task seal native tensor digest differs",
        )
        required_task = {"summary.json", "events.jsonl", "frozen_weights_after.json"}
        if "provider.jsonl" in self.files.hashes:
            required_task.add("provider.jsonl")
        require(
            set(self.seal["task_files_sha256"]) == required_task,
            "Task seal omits or adds a recording file",
        )
        for name, checksum in self.seal["task_files_sha256"].items():
            require(
                self.files.hashes.get(name) == checksum,
                "Task seal recording bytes differ",
            )
        required_worker = {
            "runtime.json",
            "checkpoint.json",
            "protocol.json",
            "frozen_plan.json",
            "reset_manifest.json",
            "prompts.json",
            "frozen_weights_before.json",
        }
        require(
            set(self.seal["worker_metadata_sha256"]) == required_worker,
            "Task seal omits or adds immutable worker metadata",
        )
        for name, checksum in self.seal["worker_metadata_sha256"].items():
            require(
                self.worker["files"].hashes.get(name) == checksum,
                "Task seal immutable worker metadata differs",
            )

    def _receipt(self):
        if self.receipt is None:
            return
        receipt = self.receipt
        require(
            self.complete
            and (
                self.worker["values"]["frozen_weights_after.json"] is not None
                or self.seal is not None
            ),
            "Passed task audit requires complete task and worker-final or task-sealed native weights",
        )
        require(
            receipt["suite"] == self.summary["suite"]
            and receipt["task_id"] == self.summary["task_id"]
            and receipt["development"] == self.summary["development"],
            "Audit task identity differs",
        )
        for field, name in (
            ("summary_sha256", "summary.json"),
            ("events_sha256", "events.jsonl"),
            ("provider_sha256", "provider.jsonl"),
        ):
            require(
                receipt[field] == self.files.hashes.get(name),
                "Audit receipt refers to different source bytes",
            )
        for name, checksum in self.files.hashes.items():
            require(
                receipt["input_file_sha256"].get(name) == checksum,
                "Audit small-file inventory differs",
            )
        audited_worker = receipt["worker"]
        _same(
            audited_worker["runtime"],
            self.worker["runtime"],
            "Audit worker runtime differs",
        )
        for name, checksum in audited_worker["input_file_sha256"].items():
            require(
                self.worker["files"].hashes.get(name) == checksum,
                "Audit frozen worker input bytes differ",
            )
        if self.seal is not None:
            require(
                audited_worker.get("completion_receipt_sha256")
                == self.files.hashes["completion_receipt.json"]
                and audited_worker.get("task_frozen_weights_after_sha256")
                == self.files.hashes["frozen_weights_after.json"],
                "Audit completed-task seal binding differs",
            )
        if self.worker["values"]["frozen_weights_after.json"] is None:
            require(
                audited_worker.get("weight_check_scope") == "completed_task",
                "Audit does not prove native weights for the completed task",
            )
        require(
            audited_worker["native_tensor_sha256"] == self.worker["native_sha256"]
            and audited_worker["reset_manifest_sha256"]
            == self.worker["values"]["reset_manifest.json"]["sha256"]
            and audited_worker["checkpoint_metadata_sha256"]
            == digest(self.summary["checkpoint"]),
            "Audit native/reset/checkpoint identity differs",
        )
        _same(
            receipt["provider_usage"],
            self.provider_usage,
            "Audit provider totals differ",
        )
        _same(
            receipt["physical_cost"],
            self.summary["physical_cost"],
            "Audit physical totals differ",
        )
        expected = [
            {
                "episode_id": row["episode_id"],
                "method": row["method"],
                "round_index": row["round_index"],
                "physical_run_id": row["physical_run_id"],
                "attempt_id": row["attempt_id"],
                "evaluation": row["evaluation"],
                "success": row["success"],
                "actions": row["actions_executed"],
                "velocity_evaluations": row["counters"]["velocity_evaluations"],
                "reset_sha256": row["reset_audit"]["sha256"],
            }
            for row in self.physical.values()
        ]
        _same(receipt["physical_runs"], expected, "Audit physical run bindings differ")
        for key, expected_value in (
            ("physical_rollouts", self.cost["rollouts"]),
            ("actions", self.cost["actions"]),
            ("velocity_evaluations", self.cost["velocity_evaluations"]),
            ("provider_bindings", len(self.providers)),
            ("accepted_policy_updates", self.cost["accepted_policy_updates"]),
            ("optimizer_steps", self.cost["optimizer_steps"]),
        ):
            require(
                receipt["counts"].get(key, 0) == expected_value,
                f"Audit counter differs: {key}",
            )
        initial = self.checkpoints[0]
        _same(
            receipt["initial_checkpoint"],
            {
                "manifest_sha256": initial["manifest_sha256"],
                "state_file_sha256": initial["files"]["state.pt"]["sha256"],
                "state_id": initial["policy"]["state_id"],
                "parameter_sha256": initial["policy"]["parameter_sha256"],
            },
            "Audit initial auxiliary checkpoint binding differs",
        )
        require(
            [row["round_index"] for row in receipt["checkpoints"]] == [1, 2, 3],
            "Audit lacks three auxiliary checkpoint bindings",
        )
        for row in receipt["checkpoints"]:
            checkpoint = self.checkpoints[row["round_index"]]
            require(
                row["manifest_sha256"] == checkpoint["manifest_sha256"]
                and row["state_file_sha256"]
                == checkpoint["files"]["state.pt"]["sha256"]
                and row["state_id"] == checkpoint["policy"]["state_id"]
                and row["parameter_sha256"] == checkpoint["policy"]["parameter_sha256"],
                "Audited auxiliary checkpoint identity differs",
            )

    @staticmethod
    def credited(row):
        return (
            row["success"]
            and row["actions_executed"] > 0
            and not row["initial_success"]
        )

    def row(self, name, method=None, *, reused=False):
        row = self.physical[name]
        actor = None
        if row["method"] in ("critique_frs_learning", "learned_noise"):
            index = row["round_index"] - int(row["method"] == "critique_frs_learning")
            require(
                index in self.checkpoints,
                "Completed rollout lacks the corresponding auxiliary checkpoint",
            )
            actor = self.checkpoints[index]["policy"]["state_id"]
        return {
            "episode_id": row["episode_id"],
            "method_id": method or row["method"],
            "physical_run_id": row["physical_run_id"],
            "reused": reused,
            "success": self.credited(row),
            "recorded_success": row["success"],
            "initial_success": row["initial_success"],
            "zero_action_success": row["zero_action_success"],
            "error": None,
            "actions": row["actions_executed"],
            "velocity_evaluations": row["counters"]["velocity_evaluations"],
            "wall_seconds": row["wall_seconds"],
            "policy_sha256": digest(
                {
                    "native_tensor_sha256": self.worker["native_sha256"],
                    "auxiliary_state_id": actor,
                }
            ),
            "source_sha256": self.files.hashes["events.jsonl"],
            "audit": {"status": "passed", "sha256": self.receipt_source["sha256"]}
            if self.receipt
            else {"status": "pending"},
            "provider_usage": _compact_usage(
                [self.providers[i] for i in row["provider_record_indexes"]]
            ),
            "recorded_method": row["method"],
            "attempt_id": name,
            "reset_audit_sha256": row["reset_audit"]["sha256"],
            "native_tensor_sha256": self.worker["native_sha256"],
            "auxiliary_state_id": actor,
            "recorded_counters": copy.deepcopy(row["counters"]),
        }

    def evidence(self):
        return {
            "task_key": self.key,
            "status": self.summary["status"],
            "audited": self.receipt is not None,
            "workflow": self.worker["runtime"]["workflow"],
            "worker": self.worker["runtime"]["worker"],
            "input_file_sha256": dict(self.files.hashes),
            "audit_receipt": self.receipt_source,
            "task_completion_seal_sha256": self.files.hashes.get(
                "completion_receipt.json"
            ),
            "native_weight_scope": "completed_task_seal"
            if self.seal is not None
            else "complete_worker"
            if self.worker["values"]["frozen_weights_after.json"] is not None
            else "initial_only_pending_final_proof",
            "audit_postprocessor_sources": self.receipt["postprocessor_sources"]
            if self.receipt
            else None,
            "audit_array_inventory": self.receipt.get("arrays")
            if self.receipt
            else None,
            "audit_native_parity": self.receipt.get("native_parity")
            if self.receipt
            else None,
            "known_generating_noise_for_Astra_reference": None,
            "post_transform_action_deviation": {
                key: self.receipt["post_transform_action_deviation"][key]
                for key in ("pairs", "max_abs", "interpretation")
            }
            if self.receipt and "post_transform_action_deviation" in self.receipt
            else None,
            "native_tensor_sha256": self.worker["native_sha256"],
            "checkpoint_metadata_sha256": digest(self.summary["checkpoint"]),
            "reset_manifest_sha256": self.worker["values"]["reset_manifest.json"][
                "sha256"
            ],
            "physical_cost": self.cost,
            "provider_usage": _compact_usage(self.providers),
            "provider_by_role": {
                role: _compact_usage(
                    row for row in self.providers if row["role"] == role
                )
                for role in ROLES
            },
            "provider_records": [
                {
                    "index": i,
                    "role": row["role"],
                    "attempt_id": row["attempt_id"],
                    "episode_id": row["episode_id"],
                    "provider_call": row["provider_call"],
                    "accepted": row["accepted"],
                    "request_fingerprint": row["request_fingerprint"],
                    "token_usage": row["token_usage"],
                    "latency_seconds": row["latency_seconds"],
                    "error_kind": row.get("error_kind"),
                    "record_digest": digest(row),
                }
                for i, row in enumerate(self.providers)
            ],
            "initial_noise_policy_checkpoint": copy.deepcopy(self.checkpoints.get(0)),
            "learning_rounds": self.learning,
            "initial_success_runs": [
                row["physical_run_id"]
                for row in self.physical.values()
                if row["initial_success"]
            ],
            "zero_action_success_runs": [
                row["physical_run_id"]
                for row in self.physical.values()
                if row["zero_action_success"]
            ],
            "unfinished_rollout_ids": sorted(self.starts.keys() - self.ends.keys()),
            "audit_limitations": self.receipt["limitations"]
            if self.receipt
            else ["Full artifact audit is pending."],
        }


def build_report(inputs, *, phase, audit_receipts=None):
    """Return renderer input from stable worker snapshots and passed task audits.

    Missing tasks/audits produce a partial report. Contradictory identities,
    changed bytes, duplicate tasks or inconsistent completed records raise.
    No receipt is manufactured for an unaudited task.
    """
    require(
        phase in ("development", "evaluation") and bool(inputs),
        "Specify a phase and at least one worker input directory",
    )
    receipts = _receipts(audit_receipts)
    workers, tasks, common, expected_tasks = [], {}, None, None
    selections, manifests_by_worker = {}, {}
    for path in inputs:
        path = Path(path)
        if (path / "runtime.json").is_file():
            root, selected = path, None
        else:
            require(
                path.name.startswith("task_")
                and path.name[5:].isdigit()
                and (path / "summary.json").is_file()
                and (path.parent / "runtime.json").is_file(),
                "Input must be a worker or recorded task directory with its worker metadata",
            )
            root, selected = path.parent, {int(path.name[5:])}
        key = root.resolve()
        if key in selections:
            previous = selections[key][1]
            require(
                previous is not None
                and selected is not None
                and not previous.intersection(selected),
                "Duplicate logical worker or task input",
            )
            previous.update(selected)
        else:
            selections[key] = (root, selected)
    for path, selected in selections.values():
        worker = _worker(path, phase)
        if selected is not None:
            require(
                selected <= set(worker["target"]["task_ids"]),
                "Selected task is outside its frozen worker assignment",
            )
        worker_index = worker["runtime"]["worker"]
        reset_manifest = worker["values"]["reset_manifest.json"]["sha256"]
        if worker_index in manifests_by_worker:
            require(
                reset_manifest == manifests_by_worker[worker_index],
                "Recovery changed the logical worker's frozen reset manifest",
            )
        manifests_by_worker[worker_index] = reset_manifest
        identity = {
            "protocol": worker["protocol"],
            "checkpoint": worker["values"]["checkpoint.json"],
            "prompts": worker["values"]["prompts.json"],
            "native_tensor_sha256": worker["native_sha256"],
            "runtime": {
                key: worker["runtime"][key]
                for key in (
                    "phase",
                    "gpu",
                    "tf32",
                    "python",
                    "payload_sha256",
                    "source_revision",
                    "packages",
                )
            },
        }
        if common is None:
            common = identity
            expected_tasks = _expected_tasks(phase, worker["protocol"])
        else:
            _same(
                identity,
                common,
                "Workers differ in frozen protocol/checkpoint/prompts/runtime identity",
            )
        workers.append(worker)
        for task_id in worker["target"]["task_ids"]:
            if selected is not None and task_id not in selected:
                continue
            directory = Path(path) / f"task_{task_id}"
            if not (directory / "summary.json").exists():
                continue
            key = f"{worker['target']['suite']}:{task_id}"
            require(
                key not in tasks,
                "Duplicate task input; repairs must select an entire authoritative task externally",
            )
            tasks[key] = _Task(directory, worker, receipts.get(key))
    require(
        receipts.keys() <= tasks.keys(),
        "An audit receipt has no supplied task recording",
    )
    require(
        tasks.keys() <= expected_tasks, "A supplied task is outside the phase cohort"
    )
    complete = set(tasks) == expected_tasks and all(
        task.complete and task.receipt is not None for task in tasks.values()
    )
    status = "complete" if complete else "partial"
    protocol = common["protocol"]
    task_order = [key for key in TASKS if key in expected_tasks]
    states = (
        protocol["evaluation_states"][:1]
        if phase == "development"
        else protocol["evaluation_states"]
    )
    adaptation_methods = protocol["adaptation_methods"]
    adaptation_rounds, evaluation_rounds = [], []
    for index in range(4):
        rows = []
        for method_number, method in enumerate(adaptation_methods):
            for key in task_order:
                task = tasks.get(key)
                name = (
                    "native_repeated_noise_state0_round0"
                    if index == 0
                    else f"{method}_state0_round{index}"
                )
                if task is not None and name in task.physical:
                    rows.append(
                        task.row(name, method, reused=index == 0 and method_number > 0)
                    )
        adaptation_rounds.append(
            {
                "index": index,
                "label": "Shared baseline" if index == 0 else f"Up to revision {index}",
                "method_ids": adaptation_methods,
                "episodes": rows,
            }
        )
    for index in range(1, 4):
        methods = protocol["evaluation_methods"] if index == 3 else ["learned_noise"]
        rows = []
        for method in methods:
            for key in task_order:
                task = tasks.get(key)
                for state in states:
                    name = f"{method}_state{state}_round{index}"
                    if task is not None and name in task.physical:
                        rows.append(task.row(name))
        evaluation_rounds.append(
            {
                "index": index,
                "label": f"Checkpoint after round {index}",
                "method_ids": methods,
                "episodes": rows,
            }
        )
    cohorts = [
        {
            "id": "adaptation",
            "label": "Same-reset adaptation: cumulative simulator success",
            "status": status,
            "curve_kind": "best_of_attempts",
            "expected_episode_ids": [
                _episode(key, protocol["seed"], 0) for key in task_order
            ],
            "method_ids": adaptation_methods,
            "rounds": adaptation_rounds,
        },
        {
            "id": "evaluation",
            "label": "Separate-reset policy evaluation"
            + (" (development only)" if phase == "development" else ""),
            "status": status,
            "curve_kind": "checkpoint_evaluation",
            "expected_episode_ids": [
                _episode(key, protocol["seed"], state)
                for key in task_order
                for state in states
            ],
            "method_ids": protocol["evaluation_methods"],
            "rounds": evaluation_rounds,
        },
    ]
    overheads = [
        row for key in task_order if key in tasks for row in tasks[key].overheads
    ]
    prompt_manifest = common["prompts"]
    initial_count = sum(
        sum(row["initial_success"] for row in task.physical.values())
        for task in tasks.values()
    )
    evidence = {
        "schema_version": PRODUCER_VERSION,
        "phase": phase,
        "status": status,
        "expected_task_keys": task_order,
        "missing_task_keys": [key for key in task_order if key not in tasks],
        "unaudited_task_keys": [
            key for key in task_order if key in tasks and tasks[key].receipt is None
        ],
        "unfinished_task_keys": [
            key for key in task_order if key in tasks and not tasks[key].complete
        ],
        "protocol_sha256": digest(protocol),
        "producer_sha256": file_sha256(__file__),
        "workers": [
            {
                "runtime": worker["runtime"],
                "input_file_sha256": dict(worker["files"].hashes),
                "native_tensor_sha256": worker["native_sha256"],
                "reset_manifest_sha256": worker["values"]["reset_manifest.json"][
                    "sha256"
                ],
            }
            for worker in workers
        ],
        "tasks": [tasks[key].evidence() for key in task_order if key in tasks],
        "physical_provider_usage": _compact_usage(
            row for key in task_order if key in tasks for row in tasks[key].providers
        ),
        "initial_success_physical_runs": initial_count,
        "initial_success_convention": "Retain every prescribed reset; credited success requires recorded success, positive executed actions, and initial_success=false. Recorded flags remain available separately.",
        "cost_scope": "All recorded physical provider calls, including failures and preflights, are partitioned once. Online-call latency and auxiliary inference are already inside rollout wall time. Critique/judge latency and training time are additional. Worker bootstrap, checkpoint I/O and unrecorded/in-flight work are not inferred.",
        "array_replay_performed_by_reporter": False,
        "partial_recording_limit": None
        if complete
        else "This is a stable partial recording, not a final experiment cost or efficacy claim. Unfinished rollout actions/velocity evaluations and calls not yet written to the ledger are unknown.",
    }
    value = {
        "schema_version": SCHEMA_VERSION,
        "title": f"Astra flow reversal and auxiliary policy improvement — {phase}",
        "status": status,
        "phase": phase,
        "tasks": [
            {"task_key": key, "instruction": instruction}
            for key, instruction in TASKS.items()
        ],
        "protocol": protocol,
        "methods": [
            {"id": key, "label": label, "description": description}
            for key, (label, description) in METHODS.items()
        ],
        "prompts": [
            {
                "id": role,
                "text": prompt_manifest["roles"][role]["system_prompt"],
                "sha256": prompt_manifest["roles"][role]["sha256"],
                "scope": f"Recorded {prompt_manifest['prompt_template_version']} / {role}; {prompt_manifest['adaptation']}",
            }
            for role in ROLES
        ],
        "notes": [
            "All tasks are known published OOD compositions. Separate reset indexes are not held-out tasks; checkpoint training overlap is unknown.",
            "Development and evaluation are separate. Development evaluates only reset1 of its three selected tasks; evaluation uses resets1–10 of all20 tasks.",
            "Adaptation always runs three revisions on reset0. Cumulative retry success does not measure a deployed learned policy. Checkpoint evaluations do not return feedback or labels to adaptation.",
            "Astra judgments, simulator predicates and auxiliary update acceptance are separate. Judges receive observable trajectories, not success labels; early environment stopping can reveal trajectory length.",
            "Only the auxiliary noise actor is trained. Native VLA tensor bytes are bound before and after each fully audited worker or sealed completed task. Online steering methods and the learned-noise evaluation have different provider/computation costs.",
            evidence["initial_success_convention"]
            + f" Recorded initial-success physical runs: {initial_count}.",
            "Shared adaptation baseline recordings count once in physical totals. Static evaluation controls are executed once at the final round, not duplicated physically at earlier checkpoints.",
            "Physical costs cover the supplied authoritative whole-task records. Aborted or replaced task attempts require separate cost reports and are not silently pooled into this cohort.",
            "Tokens through first-success revision include that completed revision's critique, online actions and judge, including failed calls. Later fixed revisions remain in the full physical cost.",
            evidence["cost_scope"],
            "No dollar price or superiority over matched RL efficiency is asserted. The renderer and producer do not rerun hidden model computation or simulator physics.",
        ],
        "sources": [
            {
                "label": "Original LIBERO-OOD paper",
                "href": "https://arxiv.org/html/2505.03500v5",
            },
            {
                "label": "Flow reversal steering paper",
                "href": "https://arxiv.org/html/2606.13675v2",
            },
            {
                "label": "Pinned OOD release",
                "href": "https://github.com/QuanyiLi/pi0-text-latent/tree/587a6cbf64f16c7b87fa5805dc0ed934192239a4",
            },
        ],
        "cohorts": cohorts,
        "overheads": overheads,
        "producer_evidence": evidence,
    }
    checked = validate_report(value)
    _same(
        checked["physical_cost"]["provider_usage"],
        evidence["physical_provider_usage"],
        "Logical report cost does not conserve the physical provider ledger",
    )
    require(
        checked["physical_cost"]["unique_rollouts"]
        == sum(task.cost["rollouts"] for task in tasks.values())
        and checked["physical_cost"]["actions"]
        == sum(task.cost["actions"] for task in tasks.values())
        and checked["physical_cost"]["velocity_evaluations"]
        == sum(task.cost["velocity_evaluations"] for task in tasks.values()),
        "Logical report duplicates or drops physical rollout work",
    )
    value["supporting_results"] = _supporting_results(value)
    for worker in workers:
        worker["files"].stable()
    for task in tasks.values():
        task.files.stable()
    return value


def _supporting_results(report):
    """Portable per-suite and per-reset numbers, computed only after full audits."""
    if report["status"] != "complete":
        return {
            "status": "withheld_until_complete",
            "per_suite": [],
            "adaptation_attempts": [],
            "paired_evaluation": [],
        }
    suite_rows, attempts, paired = [], [], []
    for cohort in report["cohorts"]:
        for method in cohort["method_ids"]:
            by_episode = {}
            for round_ in cohort["rounds"]:
                if method not in round_["method_ids"]:
                    continue
                if cohort["curve_kind"] == "checkpoint_evaluation":
                    by_episode = {}
                for row in round_["episodes"]:
                    if row["method_id"] == method:
                        by_episode[row["episode_id"]] = (
                            by_episode.get(row["episode_id"], False) or row["success"]
                        )
                for suite in ("libero_goal_ood", "libero_spatial_ood"):
                    selected = [
                        value
                        for episode, value in by_episode.items()
                        if episode.startswith(suite + ":")
                    ]
                    suite_rows.append(
                        {
                            "cohort_id": cohort["id"],
                            "method_id": method,
                            "round_index": round_["index"],
                            "suite": suite,
                            "successes": sum(selected),
                            "episodes": len(selected),
                            "estimand": cohort["curve_kind"],
                        }
                    )
        if cohort["curve_kind"] == "best_of_attempts":
            for method in cohort["method_ids"]:
                for episode in cohort["expected_episode_ids"]:
                    rows = [
                        next(
                            row
                            for row in round_["episodes"]
                            if row["method_id"] == method
                            and row["episode_id"] == episode
                        )
                        for round_ in cohort["rounds"]
                    ]
                    first = next(
                        (i for i, row in enumerate(rows) if row["success"]), None
                    )
                    prefix = first if first is not None else len(rows) - 1
                    overhead = [
                        row
                        for row in report["overheads"]
                        if (row.get("attribution") or {}).get("cohort_id")
                        == cohort["id"]
                        and row["attribution"]["episode_id"] == episode
                        and row["attribution"]["method_id"] == method
                        and row["attribution"]["round_index"] <= prefix
                    ]
                    usage = _sum_usage(
                        row["provider_usage"]
                        for row in [*rows[: prefix + 1], *overhead]
                    )
                    attempts.append(
                        {
                            "episode_id": episode,
                            "method_id": method,
                            "baseline_success": rows[0]["success"],
                            "first_success_revision": first,
                            "rescued": not rows[0]["success"] and first is not None,
                            "censored": first is None,
                            "provider_through_completed_first_success_revision_or_cap": usage,
                            "success_by_revision": [
                                any(row["success"] for row in rows[: i + 1])
                                for i in range(len(rows))
                            ],
                            "physical_run_ids": [
                                row["physical_run_id"] for row in rows
                            ],
                        }
                    )
        else:
            final = cohort["rounds"][-1]
            references = {
                method: {
                    row["episode_id"]: row["success"]
                    for row in final["episodes"]
                    if row["method_id"] == method
                }
                for method in ("native_euler10", "native_repeated_noise")
            }
            for round_ in cohort["rounds"]:
                for method in round_["method_ids"]:
                    observed = {
                        row["episode_id"]: row["success"]
                        for row in round_["episodes"]
                        if row["method_id"] == method
                    }
                    for reference, control in references.items():
                        if reference == method:
                            continue
                        require(
                            observed.keys() == control.keys(),
                            "Paired held-out comparison reset IDs differ",
                        )
                        buckets = {
                            "both_succeed": [],
                            "method_only": [],
                            "native_only": [],
                            "both_fail": [],
                        }
                        for episode in cohort["expected_episode_ids"]:
                            key = (
                                "both_succeed"
                                if observed[episode] and control[episode]
                                else "method_only"
                                if observed[episode]
                                else "native_only"
                                if control[episode]
                                else "both_fail"
                            )
                            buckets[key].append(episode)
                        paired.append(
                            {
                                "method_id": method,
                                "checkpoint_round": round_["index"],
                                "reference_method_id": reference,
                                "reference_recorded_round": final["index"],
                                "episodes": len(observed),
                                "counts": {
                                    key: len(values) for key, values in buckets.items()
                                },
                                "method_only_episode_ids": buckets["method_only"],
                                "native_only_episode_ids": buckets["native_only"],
                                "reference_is_reused_measurement": round_["index"]
                                != final["index"],
                            }
                        )
    return {
        "status": "complete",
        "per_suite": suite_rows,
        "adaptation_attempts": attempts,
        "paired_evaluation": paired,
    }


def _format_tokens(usage, field="total_tokens"):
    row = usage["tokens"][field]
    return f"{row['sum']:,}" + (
        f" known ({row['missing_calls']} missing)" if row["missing_calls"] else ""
    )


def _markdown(report, checked, derived):
    evidence, cost = report["producer_evidence"], checked["physical_cost"]
    lines = [
        f"# {report['title']}",
        "",
        f"Status: **{report['status']}**. Recorded tasks: {len(evidence['tasks'])}/{len(evidence['expected_task_keys'])}; passed full task audits: {sum(row['audited'] for row in evidence['tasks'])}.",
        "",
        evidence["initial_success_convention"],
        f"Initial-success physical recordings: **{evidence['initial_success_physical_runs']}**.",
        "",
    ]
    if report["status"] != "complete":
        lines += [
            "Efficacy is withheld until the full declared task/reset/method grid and all independent task audits are complete.",
            "",
            evidence["partial_recording_limit"],
            "",
        ]
    for cohort in report["cohorts"]:
        if cohort["id"] not in derived:
            continue
        lines += [
            f"## {cohort['label']}",
            "",
            "| Method | Round / revision | Successes / prescribed episodes |",
            "|---|---:|---:|",
        ]
        for method, detail in derived[cohort["id"]].items():
            for point in detail["points"]:
                lines.append(
                    f"| {METHODS[method][0]} | {point['round']} | {point['successes']}/{point['episodes']} |"
                )
        if cohort["curve_kind"] == "best_of_attempts":
            lines += [
                "",
                "| Method | Rescues / baseline failures | Median revision among rescues | Median tokens among complete-usage rescues | Rescues with missing usage | Censored failures |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for method, detail in derived[cohort["id"]].items():
                row = detail["rescue"]
                lines.append(
                    f"| {METHODS[method][0]} | {row['rescued']}/{row['baseline_failed']} | {row['median_revision_among_rescues']} | {row['median_tokens_among_rescues_with_complete_usage']} | {row['rescues_missing_usage']} | {row['censored']} |"
                )
        else:
            lines += [
                "",
                "Checkpoint success is not accumulated across rounds. Static comparators have one final-round measurement; no unexecuted earlier measurement is invented.",
            ]
        lines += [
            "",
            "| Final-round task | "
            + " | ".join(METHODS[name][0] for name in cohort["method_ids"])
            + " |",
            "|---|" + "---:|" * len(cohort["method_ids"]),
        ]
        for task in evidence["expected_task_keys"]:
            values = [
                derived[cohort["id"]][method]["per_task"][task]
                for method in cohort["method_ids"]
            ]
            lines.append(
                f"| {task}: {TASKS[task]} | "
                + " | ".join(f"{row['successes']}/{row['episodes']}" for row in values)
                + " |"
            )
        lines.append("")
        lines += [
            "| Suite | Method | Round / revision | Successes / episodes |",
            "|---|---|---:|---:|",
        ]
        for row in report["supporting_results"]["per_suite"]:
            if row["cohort_id"] == cohort["id"]:
                lines.append(
                    f"| {row['suite']} | {METHODS[row['method_id']][0]} | {row['round_index']} | {row['successes']}/{row['episodes']} |"
                )
        lines.append("")
    usage = cost["provider_usage"]
    lines += [
        "## Recorded physical cost",
        "",
        f"{cost['unique_rollouts']:,} unique rollouts; {cost['actions']:,} actions; {cost['velocity_evaluations']:,} velocity evaluations; {cost['training_steps']:,} auxiliary optimizer steps.",
        "",
        f"{usage['calls']:,} physical provider calls: {usage['accepted_calls']:,} accepted, {usage['failed_calls']:,} failed/rejected; {usage['preflight_failures']:,} additional no-network preflight failures.",
        "",
        f"Input tokens: {_format_tokens(usage, 'input_tokens')}; output: {_format_tokens(usage, 'output_tokens')}; total: **{_format_tokens(usage)}**; reasoning subset: {_format_tokens(usage, 'reasoning_tokens')}.",
        "",
        evidence["cost_scope"],
        "",
        "## Recorded auxiliary updates",
        "",
        "| Task | Round | Astra promoted | Optimizer steps | Replay samples | Actor state |",
        "|---|---:|---|---:|---:|---|",
    ]
    for task in evidence["tasks"]:
        for row in task["learning_rounds"]:
            if row["method_id"] == "critique_frs_learning":
                checkpoint = row["checkpoint"]["policy"]
                lines.append(
                    f"| {task['task_key']} | {row['round_index']} | {row['promoted']} | {(row['training'] or {}).get('optimizer_updates', 0)} | {checkpoint['replay_samples']} | `{checkpoint['state_id']}` |"
                )
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    lines.extend(note + "\n" for note in report["notes"])
    lines += [
        "Exact system prompts, hash-bound source inventories, per-call token availability and per-round training/checkpoint details are retained in report.json. No new experiment, model, provider or simulator call was made to generate this report.",
        "",
    ]
    return "\n".join(lines)


def write_report(report, output):
    """Write a new aggregate plus self-contained HTML; preserve source artifacts."""
    output = Path(output)
    require(not output.exists(), "Refusing to overwrite a report directory")
    checked, derived = validate_report(report), derived_results(report)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_bytes(_json_bytes(report))
    (output / "derived.json").write_bytes(
        _json_bytes({"validation": checked, "cohorts": derived})
    )
    (output / "supporting_results.json").write_bytes(
        _json_bytes(report["supporting_results"])
    )
    (output / "report.md").write_text(_markdown(report, checked, derived))
    fields = [
        "cohort",
        "round",
        "episode_id",
        "method_id",
        "physical_run_id",
        "reused",
        "success",
        "recorded_success",
        "initial_success",
        "actions",
        "velocity_evaluations",
        "policy_sha256",
        "audit_status",
        "calls",
        *TOKEN_FIELDS,
        "total_tokens_missing_calls",
    ]
    with (output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cohort in report["cohorts"]:
            for round_ in cohort["rounds"]:
                for row in round_["episodes"]:
                    record = {key: row[key] for key in fields if key in row}
                    record.update(
                        cohort=cohort["id"],
                        round=round_["index"],
                        audit_status=row["audit"]["status"],
                        calls=row["provider_usage"]["calls"],
                        total_tokens_missing_calls=row["provider_usage"]["tokens"][
                            "total_tokens"
                        ]["missing_calls"],
                    )
                    record.update(
                        {
                            field: row["provider_usage"]["tokens"][field]["sum"]
                            for field in TOKEN_FIELDS
                        }
                    )
                    writer.writerow(record)
    render_report(output / "report.json", output / "html")
    manifest = {
        "schema_version": PRODUCER_VERSION,
        "status": report["status"],
        "phase": report["phase"],
        "producer_sha256": file_sha256(__file__),
        "report_sha256": file_sha256(output / "report.json"),
        "files": {
            str(path.relative_to(output)): {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    (output / "manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "evaluation"), required=True)
    parser.add_argument("--audits", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    value = build_report(args.inputs, phase=args.phase, audit_receipts=args.audits)
    require(
        not args.require_complete or value["status"] == "complete",
        "Complete publication requires every task and passed audit",
    )
    manifest = write_report(value, args.output)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "phase": manifest["phase"],
                "report_sha256": manifest["report_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
