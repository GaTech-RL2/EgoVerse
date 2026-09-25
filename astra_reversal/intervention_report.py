"""Audit complete iterative-intervention workers and report attempts and tokens.

Run ``python -m astra_reversal.intervention_report --phase development
--inputs worker_0 worker_1 --output NEW_DIR`` (eight workers for evaluation).
Inputs are read-only. Events and provider ledgers are required, but lossless
array archives need not be downloaded. This is a recording/denominator audit;
it does not independently recompute model inversions or rerun the simulator.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .config import BenchmarkConfig
from .intervention_agent import _validate_request, parse_proposal, summarize_calls
from .intervention_rollout import load_reset_manifest
from .intervention_search import aggregate_reports, arm_summary, feedback_row
from .interventions import ARMS, noise_basis, random_noise_proposal
from .records import digest, file_sha256

SUITES = ("libero_goal_ood", "libero_spatial_ood")
TOKEN_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")
CONTROLS = ("known_noise", "policy_fresh")
SUPPORTED_PROMPTS = {"astra-intervention-http-1", "astra-intervention-http-2"}
RANDOM_COEFFICIENT_ULPS = 4
LIMITATIONS = [
    "Online adaptation with simulator reset access; not zero-shot evaluation. "
    "The OOD tasks and seed7 outcomes were previously observed. Checkpoint "
    "training overlap is unknown; seed19 is a follow-up reset condition.",
    "A rejected proposal consumes an attempt, even when no rollout executes. "
    "Unsuccessful searches remain right-censored after attempt 5; success-only "
    "iteration medians do not summarize the failures.",
    "Reasoning tokens are a subset of output tokens and must not be added twice. "
    "Token sums with missing records are partial observed costs. No USD price "
    "or missing usage is imputed.",
    "One identity baseline is physically executed per case and attributed to "
    "each arm for a standalone comparison. Standalone arm costs must not be "
    "summed as physical experiment costs. Development v1 attempts one revision "
    "per arm after baseline success; v2 continues rejected proposals until one "
    "revision executes, within the same five-attempt cap.",
    "Initialization uses RK4/100/power3; rollout execution uses Euler/10/power1. "
    "Initialization velocity evaluations are separate from rollout evaluations. "
    "The baseline's policy and wall times include initialization, recording and "
    "archive synchronization; case wall time includes all search work. Parallel "
    "worker wall times are retained separately, not pooled into a latency p95.",
    "Hashes bind the downloaded reports, events, provider ledgers and complete "
    "reset manifests. The audit verifies recorded initialization errors and "
    "paired reset hashes, not independent numerical reconstruction or native "
    "simulator replay. Raw array artifacts remain separate.",
    "Noise-only Astra proposals versus random noise share the same basis and "
    "bounds. Comparisons of language/vision arms against random noise change "
    "the intervention operator as well as the proposal source. The 20-case "
    "OOD follow-up is exploratory and uses known tasks with new resets.",
    "Regenerated random directions allow at most four float64 ULPs of "
    "coefficient normalization roundoff across CPU libraries; basis identities, "
    "coefficient shapes, kinds and scales remain exact. Observed differences "
    "are logged, and recorded proposals are never altered.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, *, integer=False):
    require(
        type(value) in ((int,) if integer else (int, float))
        and math.isfinite(value)
        and value >= 0,
        f"Invalid nonnegative metric: {name}",
    )
    return value


def _sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, f"Duplicate JSON key: {key}")
            value[key] = item
        return value

    def constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


class Inputs:
    """Hash exact source bytes, stream events, and detect changing downloads."""

    def __init__(self):
        self.sources = {}
        self.absent = set()

    def _record(self, path, checksum, size):
        key = str(Path(path).resolve())
        source = {"path": key, "sha256": checksum, "bytes": size}
        require(
            key not in self.sources or self.sources[key] == source,
            f"Input changed while reading: {path}",
        )
        self.sources[key] = source

    def read(self, path):
        raw = Path(path).read_bytes()
        self._record(path, hashlib.sha256(raw).hexdigest(), len(raw))
        return _json(raw)

    def lines(self, path):
        checksum, size = hashlib.sha256(), 0
        with Path(path).open("rb") as stream:
            for line in stream:
                checksum.update(line)
                size += len(line)
                if line.strip():
                    yield _json(line)
        self._record(path, checksum.hexdigest(), size)

    def source(self, path):
        return self.sources[str(Path(path).resolve())]

    def finish(self):
        for path, source in self.sources.items():
            require(
                file_sha256(path) == source["sha256"],
                f"Input changed during audit: {path}",
            )
        require(
            all(not Path(path).exists() for path in self.absent),
            "A provider ledger appeared during the audit",
        )


def _assignment(phase, worker):
    if phase == "development":
        return {"suite": "libero_10", "case_shard": worker, "case_shards": 2}
    return {
        "suite": SUITES[worker // 4],
        "case_shard": worker % 4,
        "case_shards": 4,
    }


def _protocol(value):
    require(
        value["seed"] == 19
        and value["arms"] == list(ARMS)
        and value["attempt_budget"] == 5
        and value["execute_steps"] == 5
        and value["development_cases"] == [[0, 1], [1, 1]]
        and value["evaluation_cases"] == [[task, 0] for task in range(10)]
        and value["evaluation_suites"] == list(SUITES)
        and value["development_force_one_intervention"] is True
        and value["inversion_solver"]
        == {"solver": "rk4", "steps": 100, "time_power": 3.0}
        and value["execution_solver"]
        == {"solver": "euler", "steps": 10, "time_power": 1.0}
        and value["numerical_gate"] == {"noise_max_abs": 0.1, "action_max_abs": 0.02}
        and value["noise"]["rank"] == 8
        and value["noise"]["maximum_rms_delta"] == 0.5
        and value["language"]
        == {"operator": "pooled_text_residual", "maximum_relative_norm": 0.25}
        and value["vision"]
        == {
            "operator": "static_magenta_annotations",
            "feedback": "unmodified raw frames",
            "tracking": False,
        }
        and value["astra"]
        == {
            "model": "azure/openai/gpt-6-astra",
            "reasoning_effort": "low",
            "max_completion_tokens": 4096,
            "timeout_seconds": 170,
        },
        "Unknown intervention protocol or changed fixed cases/solver",
    )
    fields = {"development_stopping", "development_embedding_probe"} & set(value)
    require(
        not fields or fields == {"development_stopping", "development_embedding_probe"},
        "Incomplete v2 development protocol",
    )
    if fields:
        require(
            value["development_embedding_probe"]
            == {
                "alpha": 0.5,
                "guidance_suffix": "\nGuidance: Focus on the target object and its destination.",
                "executes_actions": False,
            },
            "Unknown fixed development probe protocol",
        )


def _reset(audit, entry, benchmark):
    require(
        audit["sha256"] == digest({k: v for k, v in audit.items() if k != "sha256"}),
        "Paired reset audit hash mismatch",
    )
    for name in (
        "episode_id",
        "seed",
        "reset_state_sha256",
        "reset_model_sha256",
        "bddl_sha256",
    ):
        require(audit[name] == entry[name], f"Paired reset differs: {name}")
    require(
        all(
            _sha(audit[f"post_stabilization_{part}_sha256"])
            for part in ("state", "model", "observation")
        )
        and audit["stabilization_steps"] == benchmark.stabilization_steps
        and audit["initial_success"] is False
        and audit["initial_terminated"] is False,
        "Invalid stabilized initial scene",
    )


def _attempt(row, entry, benchmark, expected_reset):
    require(type(row["success"]) is bool, "Nonboolean success")
    require(type(row["rollout_executed"]) is bool, "Nonboolean rollout flag")
    for key in ("actions_executed", "velocity_evaluations", "iteration"):
        _number(row[key], key, integer=True)
    _number(row["wall_seconds"], "wall_seconds")
    _number(row.get("proposal_seconds", 0), "proposal_seconds")
    if not row["rollout_executed"]:
        require(
            row["status"] == "proposal_error"
            and row["success"] is False
            and row["actions_executed"] == row["velocity_evaluations"] == 0
            and row["wall_seconds"] == 0
            and isinstance(row.get("error"), str)
            and bool(row["error"]),
            "Invalid unexecuted proposal attempt",
        )
        return
    require(
        row["episode_id"] == entry["episode_id"]
        and row["initial_success"] is False
        and row["zero_action_success"] is False
        and row["captured_initial_success"] == entry["initially_successful"]
        and type(row["terminated"]) is bool
        and row["execute_steps"] == 5
        and row["action_budget"] == benchmark.task_action_budget
        and 0 < row["actions_executed"] <= row["action_budget"],
        "Invalid executed rollout or initially successful scene",
    )
    _reset(row["reset_audit"], entry, benchmark)
    require(
        row["reset_audit"] == expected_reset,
        "Full paired reset differs across attempts",
    )
    _number(row["policy_replans"], "policy_replans", integer=True)
    require(
        row["policy_replans"] == math.ceil(row["actions_executed"] / 5)
        and row["velocity_evaluations"] == 10 * row["policy_replans"],
        "Rollout replans or Euler velocity count mismatch",
    )
    for key in ("reset_seconds", "policy_seconds", "environment_seconds"):
        _number(row[key], key)
        require(
            row[key] <= row["wall_seconds"] + 1e-6, f"{key} exceeds rollout wall time"
        )
    _number(row["clipped_values"], "clipped_values", integer=True)
    _number(row["condition_preparation_seconds"], "condition_preparation_seconds")
    expected_status = (
        "success"
        if row["success"]
        else "terminated"
        if row["terminated"]
        else "budget_exhausted"
    )
    require(row["status"] == expected_status, "Rollout termination label mismatch")
    require(
        row["success"]
        or row["terminated"]
        or row["actions_executed"] == row["action_budget"],
        "Unsuccessful rollout stopped before its action budget",
    )


def _cost(rows):
    return {
        "rollouts": sum(row["rollout_executed"] for row in rows),
        "simulated_actions": sum(row["actions_executed"] for row in rows),
        "execution_velocity_evaluations": sum(
            row["velocity_evaluations"] for row in rows
        ),
        "rollout_wall_seconds": sum(row["wall_seconds"] for row in rows),
        "proposal_seconds": sum(row.get("proposal_seconds", 0) for row in rows),
        "policy_replans": sum(row.get("policy_replans", 0) for row in rows),
        "clipped_values": sum(row.get("clipped_values", 0) for row in rows),
    }


def _sum_cost(rows):
    return {key: sum(row[key] for row in rows) for key in rows[0]} if rows else {}


def _event_attempt_key(row):
    proposal = row.get("proposal")
    mode = (
        row["candidate_id"] if proposal is None else proposal.get("arm", "random_noise")
    )
    return mode, row["iteration"]


def _requested_nonzero(proposal):
    proposal = proposal or {}
    noise, language = proposal.get("noise"), proposal.get("language")
    return {
        "noise": bool(
            noise and noise["perturbation_scale"] > 0 and any(noise["coefficients"])
        ),
        "language": bool(language and language["scale"] > 0),
        "vision": any(mark["gain"] > 0 for mark in proposal.get("vision", [])),
    }


def _core_attempt(row):
    omitted = {"cumulative_token_usage"}
    if row["rollout_executed"]:
        omitted.add("proposal_seconds")
    return {key: value for key, value in row.items() if key not in omitted}


def _provider_records(inputs, directory, report, protocol):
    records = {}
    unexpected = {p.name for p in directory.glob("*_provider.jsonl")} - {
        f"{arm}_provider.jsonl" for arm in ARMS
    }
    require(not unexpected, "Undeclared provider ledger")
    for arm in ARMS:
        path = directory / f"{arm}_provider.jsonl"
        if path.exists():
            rows = list(inputs.lines(path))
        else:
            inputs.absent.add(str(path.resolve()))
            rows = []
        summarize_calls(rows)
        require(arm != "random_noise" or not rows, "Random search made provider calls")
        require(
            len({row["iteration"] for row in rows}) == len(rows),
            "Duplicate provider calls or hidden retries",
        )
        for row in rows:
            require(
                row["episode_id"] == report["episode_id"]
                and row["arm"] == arm
                and row["iteration"] in range(2, 6)
                and row["requested_model"] == protocol["astra"]["model"]
                and row["prompt_template_version"] in SUPPORTED_PROMPTS
                and row["prompt_template_version"]
                == (
                    "astra-intervention-http-2"
                    if "development_stopping" in protocol
                    else "astra-intervention-http-1"
                )
                and row["sampling_settings"]
                == {
                    key: protocol["astra"][key]
                    for key in ("reasoning_effort", "max_completion_tokens")
                }
                and row["cache"] == {"no-cache": True}
                and row["response_format"] == {"type": "json_object"},
                "Provider identity/settings differ from the frozen protocol",
            )
        records[arm] = rows
    return records


def _events(inputs, directory, report, entry, protocol, records, expected_basis):
    """Reconcile complete event coverage without loading referenced NPY arrays."""
    physical = [report["baseline"], *report["controls"].values()] + [
        row for arm in ARMS for row in report["arms"][arm]["attempts"][1:]
    ]
    expected = {("reversal_identity", 1): report["baseline"]}
    expected.update({(mode, 1): row for mode, row in report["controls"].items()})
    expected.update(
        {
            (arm, row["iteration"]): row
            for arm in ARMS
            for row in report["arms"][arm]["attempts"][1:]
        }
    )
    require(len(expected) == len(physical), "Duplicate physical attempt identity")
    providers = {(arm, row["iteration"]): row for arm in ARMS for row in records[arm]}
    attempts, requests, proposals, local_errors = set(), set(), set(), set()
    generations = defaultdict(list)
    counts = Counter()
    basis_id = action_spec = None
    for sequence, event in enumerate(inputs.lines(directory / "events.jsonl")):
        require(event["sequence"] == sequence, "Missing or reordered event sequence")
        kind = event["kind"]
        counts[kind] += 1
        if kind == "case":
            require(
                sequence == 0
                and event["entry"] == entry
                and event["protocol"] == protocol,
                "Case event differs from frozen inputs",
            )
            basis_id = event["basis_id"]
            require(
                basis_id == expected_basis and event["basis"]["sha256"] == basis_id,
                "Noise basis descriptor hash mismatch",
            )
        elif kind == "inversion_initialization":
            action_spec = event["action_spec"]
            require(
                all(
                    event[key] == value
                    for key, value in report["initialization"].items()
                ),
                "Initialization event differs from summary",
            )
            for name in ("known_noise", "recovered_noise"):
                require(
                    event[name]["sha256"] == report["initialization"][f"{name}_sha256"],
                    "Initialization array descriptor differs",
                )
        elif kind == "development_embedding_probe":
            require(
                event["probe"]
                == report["initialization"].get("development_embedding_probe"),
                "Development embedding probe differs from initialization",
            )
        elif kind in ("attempt", "proposal_error"):
            row = event["attempt"]
            key = _event_attempt_key(row)
            if kind == "proposal_error":
                key = event["arm"], row["iteration"]
            require(
                key in expected and key not in attempts,
                "Extra or duplicate attempt event",
            )
            require(
                _core_attempt(row) == _core_attempt(expected[key]),
                "Attempt event and summary disagree",
            )
            require(
                (kind == "attempt") == row["rollout_executed"],
                "Wrong attempt event kind",
            )
            attempts.add(key)
        elif kind == "candidate_generation":
            key = event["mode"], event["iteration"]
            require(
                key in expected and expected[key]["rollout_executed"],
                "Generation without a physical rollout",
            )
            require(
                event["candidate_id"] == expected[key]["candidate_id"]
                and event["velocity_evaluations"] == 10,
                "Generation identity or Euler evaluation count differs",
            )
            generations[key].append(event["observation_step"])
        elif kind == "intervention_request":
            request = event["request"]
            _validate_request(request)
            key = request["arm"], request["iteration"]
            require(
                key in expected and key not in requests and key[0] != "random_noise",
                "Extra or duplicate intervention request",
            )
            require(
                request["episode_id"] == entry["episode_id"]
                and request["task_instruction"] == entry["instruction"]
                and request["basis_id"] == basis_id
                and request["max_iterations"] == 5
                and request["action_spec"] == action_spec,
                "Request differs from the frozen case",
            )
            prior = report["arms"][key[0]]["attempts"][: key[1] - 1]
            require(
                request["prior_candidates"] == [feedback_row(row) for row in prior],
                "Request history differs from observed prior attempts",
            )
            if key not in providers:
                require(
                    not (directory / f"{key[0]}_provider.jsonl").exists()
                    and expected[key]["status"] == "proposal_error"
                    and expected[key].get("error")
                    == "NVIDIA_INFERENCE_API_KEY is not set",
                    "Provider request lacks its call ledger or a recognized pre-HTTP failure",
                )
                local_errors.add(key)
                requests.add(key)
                continue
            provider = providers[key]
            require(
                provider["request_fingerprint"] == request["request_fingerprint"],
                "Provider request fingerprint mismatch",
            )
            if provider["accepted"]:
                response = provider["response"]
                choices = response.get("choices", [])
                require(
                    response.get("model") == protocol["astra"]["model"]
                    and 200 <= provider["http_status"] < 300
                    and len(choices) == 1
                    and choices[0].get("finish_reason") == "stop",
                    "Accepted provider response has invalid model/finish",
                )
                message = choices[0]["message"]
                require(
                    message.get("role") == "assistant" and not message.get("refusal"),
                    "Accepted provider response is not a valid assistant proposal",
                )
                proposal = parse_proposal(message["content"], request)
                require(
                    proposal == expected[key]["proposal"]
                    and provider["candidate_id"] == proposal["candidate_id"],
                    "Accepted provider proposal differs from executed/rejected candidate",
                )
            else:
                require(
                    not expected[key]["rollout_executed"],
                    "Rejected provider response produced a rollout",
                )
            requests.add(key)
        elif kind == "intervention_proposal":
            key = event["arm"], event["iteration"]
            require(
                key in expected
                and key not in proposals
                and event["proposal"] == expected[key]["proposal"],
                "Proposal event differs from the candidate",
            )
            proposals.add(key)
        else:
            raise ValueError(f"Unknown intervention event kind: {kind}")
    require(
        counts["case"] == counts["inversion_initialization"] == 1,
        "Missing or duplicate case/initialization event",
    )
    require(
        counts["development_embedding_probe"]
        == int(report["initialization"].get("development_embedding_probe") is not None),
        "Missing or duplicate development embedding probe",
    )
    require(attempts == set(expected), "Missing physical attempt events")
    require(
        requests - local_errors == set(providers),
        "Provider ledger has an unmatched call",
    )
    for key, row in expected.items():
        require(
            generations[key]
            == (
                list(range(0, row["actions_executed"], 5))
                if row["rollout_executed"]
                else []
            ),
            "Generation event coverage differs from executed steps",
        )
        if key[0] in ARMS and row.get("proposal") is not None:
            require(key in proposals, "Candidate proposal event is missing")
        if key[0] in ARMS and key[0] != "random_noise" and row["rollout_executed"]:
            require(
                key in requests, "Executed Astra candidate lacks a provider request"
            )
    return {
        "event_counts": dict(counts),
        "pre_http_proposal_errors": [
            {
                "arm": arm,
                "iteration": iteration,
                "reason": "NVIDIA_INFERENCE_API_KEY is not set",
            }
            for arm, iteration in sorted(local_errors)
        ],
    }


def _random_noise_check(row, expected):
    """Allow only float64 normalization roundoff when regenerating directions.

    BLAS reductions can differ by a few ULPs across the audit and worker hosts.
    The recorded proposal remains authoritative for the actual noise operation;
    neither its coefficients nor any recorded events are rewritten here.
    """
    message = "Random-search candidate differs from the matched seeded basis/scales"
    proposal = row["proposal"]
    observed = proposal["noise"]
    require(
        row["rollout_executed"]
        and proposal["language"] is None
        and proposal["vision"] == []
        and isinstance(observed, dict)
        and {key: value for key, value in observed.items() if key != "coefficients"}
        == {key: value for key, value in expected.items() if key != "coefficients"},
        message,
    )
    coefficients = observed.get("coefficients")
    require(
        isinstance(coefficients, list)
        and all(
            type(value) in (int, float) and math.isfinite(value)
            for value in coefficients
        ),
        message,
    )
    actual = np.asarray(coefficients, dtype=np.float64)
    wanted = np.asarray(expected["coefficients"], dtype=np.float64)
    require(actual.shape == wanted.shape and np.isfinite(actual).all(), message)
    difference = np.abs(actual - wanted)
    spacing = np.spacing(np.abs(wanted))
    require(np.all(difference <= RANDOM_COEFFICIENT_ULPS * spacing), message)
    return {
        "iteration": row["iteration"],
        "coefficient_tolerance_ulps": RANDOM_COEFFICIENT_ULPS,
        "coefficient_max_abs_difference": float(difference.max()),
        "coefficient_max_ulp_difference": float((difference / spacing).max()),
        "coefficients_exact": bool(np.array_equal(actual, wanted)),
        "basis_kind_and_scale_exact": True,
    }


def _case(inputs, directory, entry, benchmark, protocol, phase, checkpoint):
    report = inputs.read(directory / "summary.json")
    require(
        report["status"] == "complete"
        and report["episode_id"] == entry["episode_id"]
        and report["suite"] == benchmark.suite
        and report["task_id"] == entry["task_id"]
        and report["seed"] == 19
        and report["development"] == (phase == "development")
        and report["protocol_sha256"] == digest(protocol)
        and report["reset_entry_sha256"] == digest(entry)
        and report["checkpoint"] == checkpoint,
        "Case identity/status differs from frozen inputs",
    )
    require(
        set(report["arms"]) == set(ARMS) and set(report["controls"]) == set(CONTROLS),
        "Case has missing or undeclared arms/controls",
    )
    initial = report["initialization"]
    runtime_v2 = "development_embedding_probe" in initial
    require(
        runtime_v2 == ("development_stopping" in protocol),
        "Initialization runtime revision differs from the frozen protocol",
    )
    probe = initial.get("development_embedding_probe")
    if probe is not None:
        conditioning = probe["conditioning"]
        require(
            phase == "development"
            and probe["kind"]
            == "fixed_numerical_probe_not_an_astra_proposal_or_rollout"
            and probe["velocity_evaluations"] == 10
            and conditioning["has_effect"] is True
            and 0
            < _number(conditioning["delta_frobenius"], "embedding delta")
            <= _number(conditioning["bound_frobenius"], "embedding bound")
            and 0
            < _number(conditioning["relative_rms"], "relative embedding delta")
            <= 0.25
            and 0
            < _number(probe["output_difference"]["max_abs"], "probe action difference"),
            "Fixed development embedding probe did not show a bounded effect",
        )
    require(
        not runtime_v2 or (probe is not None) == (phase == "development"),
        "Missing or unexpected development embedding probe",
    )
    errors = initial["errors"]
    for error in errors.values():
        for key in ("max_abs", "rmse"):
            _number(error[key], f"initialization {key}")
    require(
        initial["passed"] is True
        and errors["noise"]["max_abs"] <= protocol["numerical_gate"]["noise_max_abs"]
        and errors["actions"]["max_abs"] <= protocol["numerical_gate"]["action_max_abs"]
        and errors["native_parity"]["max_abs"] <= 1e-5
        and errors["zero_embedding_hook_parity"]["max_abs"] == 0
        and initial["velocity_evaluations"] == 1230 + (10 if probe is not None else 0),
        "Initialization gate/evaluation count does not pass the fixed protocol",
    )
    for key in ("condition_id", "known_noise_sha256", "recovered_noise_sha256"):
        require(_sha(initial[key]), f"Invalid initialization {key}")
    baseline = report["baseline"]
    require(
        baseline["candidate_id"] == "reversal_identity"
        and baseline["iteration"] == 1
        and baseline["proposal"] is None
        and baseline["rollout_executed"],
        "Invalid common identity baseline",
    )
    reset = baseline["reset_audit"]
    _attempt(baseline, entry, benchmark, reset)
    for mode, row in report["controls"].items():
        require(
            row["candidate_id"] == mode
            and row["iteration"] == 1
            and row["proposal"] is None
            and row["rollout_executed"],
            "Invalid control",
        )
        _attempt(row, entry, benchmark, reset)
    records = _provider_records(inputs, directory, report, protocol)
    seed_words = [19, int(digest(entry["episode_id"])[:8], 16)]
    basis, basis_id = noise_basis(
        (1, checkpoint["horizon"], checkpoint["model_action_dim"]),
        np.random.SeedSequence(seed_words + [1]),
    )
    random_rng = np.random.default_rng(np.random.SeedSequence(seed_words + [3]))
    arms, compatibility, random_checks = {}, [], []
    for arm in ARMS:
        value = report["arms"][arm]
        attempts = value["attempts"]
        require(
            value["status"] == "complete" and attempts and attempts[0] == baseline,
            "All arms must share the exact common baseline",
        )
        require(
            [row["iteration"] for row in attempts] == list(range(1, len(attempts) + 1))
            and len(attempts) <= 5,
            "Attempts must consume consecutive budget slots",
        )
        require(
            len({row["candidate_id"] for row in attempts}) == len(attempts),
            "Candidate IDs repeat within an arm",
        )
        for row in attempts:
            _attempt(row, entry, benchmark, reset)
        if arm == "random_noise":
            for row in attempts[1:]:
                random_checks.append(
                    _random_noise_check(row, random_noise_proposal(basis, random_rng))
                )
        for row in attempts[1:]:
            noise = (row.get("proposal") or {}).get("noise")
            require(
                noise is None or noise.get("basis_id") == basis_id,
                "Candidate uses a different fixed noise basis",
            )
        derived = arm_summary(attempts, 5)
        first = derived["first_success_attempt"]
        if phase == "development" and runtime_v2:
            first_revision = next(
                (row["iteration"] for row in attempts[1:] if row["rollout_executed"]), 5
            )
            last = max(first or 5, first_revision)
        else:
            last = max(first or 5, 2 if phase == "development" else 1)
        require(
            len(attempts) == last,
            "Search stopped early or continued beyond its stopping rule",
        )
        derived["standalone_velocity_evaluations_through_success_or_cap"] = (
            derived["velocity_evaluations_through_success_or_cap"]
            + initial["velocity_evaluations"]
        )
        require(
            value["summary"] == derived, "Stored arm success/cost summary disagrees"
        )
        usage = summarize_calls(records[arm])
        if value["token_usage"] is None:
            require(
                not records[arm] and not (directory / f"{arm}_provider.jsonl").exists(),
                "Null token summary cannot conceal a provider ledger",
            )
            compatibility.append(arm)
        else:
            require(
                value["token_usage"] == usage,
                "Arm token summary disagrees with provider ledger",
            )
        for row in attempts[1:]:
            prefix = [r for r in records[arm] if r["iteration"] <= row["iteration"]]
            require(
                row["cumulative_token_usage"] == summarize_calls(prefix),
                "Cumulative attempt tokens disagree with provider ledger",
            )
        first_usage = (
            summarize_calls([r for r in records[arm] if r["iteration"] <= first])
            if first is not None
            else None
        )
        require(
            value["tokens_to_first_success"] == first_usage,
            "Tokens to first success disagree with provider ledger",
        )
        through = [
            row for row in attempts if first is None or row["iteration"] <= first
        ]
        standalone = _cost(through)
        standalone["initialization_velocity_evaluations"] = initial[
            "velocity_evaluations"
        ]
        standalone["total_velocity_evaluations"] = (
            standalone["execution_velocity_evaluations"]
            + initial["velocity_evaluations"]
        )
        arms[arm] = {
            **derived,
            "token_usage_actual": usage,
            "token_usage_through_success_or_cap": first_usage
            if first is not None
            else usage,
            "tokens_to_first_success": first_usage,
            "standalone_cost_through_success_or_cap": standalone,
            "actual_attempt_cost_including_shared_baseline": _cost(attempts),
            "attempts": attempts,
        }
    event_audit = _events(inputs, directory, report, entry, protocol, records, basis_id)
    physical_rows = [baseline, *report["controls"].values()] + [
        row for arm in ARMS for row in report["arms"][arm]["attempts"][1:]
    ]
    physical = _cost(physical_rows)
    physical["initialization_velocity_evaluations"] = initial["velocity_evaluations"]
    physical["total_velocity_evaluations"] = (
        physical["execution_velocity_evaluations"] + initial["velocity_evaluations"]
    )
    physical_usage = summarize_calls([row for arm in ARMS for row in records[arm]])
    stored_cost = report["physical_cost"]
    require(
        stored_cost["rollouts"] == physical["rollouts"]
        and stored_cost["simulated_actions"] == physical["simulated_actions"]
        and stored_cost["velocity_evaluations"]
        == physical["total_velocity_evaluations"]
        and stored_cost["token_usage"] == physical_usage,
        "Physical costs disagree with unique rollouts/provider calls",
    )
    _number(report["physical_wall_seconds"], "physical_wall_seconds")
    result = {
        "episode_id": entry["episode_id"],
        "suite": entry["suite"],
        "task_id": entry["task_id"],
        "initial_state_id": entry["initial_state_id"],
        "instruction": entry["instruction"],
        "reset_entry_sha256": digest(entry),
        "reset_audit": reset,
        "random_noise_reconstruction": random_checks,
        "baseline": baseline,
        "controls": report["controls"],
        "initialization": initial,
        "runtime_semantics": (
            "evaluation_first_success_or_attempt5_v2"
            if runtime_v2
            else "evaluation_first_success_or_attempt5_v1"
        )
        if phase == "evaluation"
        else (
            "development_first_success_with_executed_revision_or_attempt5_v2"
            if runtime_v2
            else "development_first_success_force_attempt2_v1"
        ),
        "arms": arms,
        "physical_cost": {
            **physical,
            "token_usage": physical_usage,
            "case_wall_seconds": report["physical_wall_seconds"],
        },
        **event_audit,
        "legacy_null_usage_verified_zero_arms": compatibility,
        "source": inputs.source(directory / "summary.json"),
    }
    return result, report, records


def _aggregate(cases, records):
    baseline_failed = [case for case in cases if not case["baseline"]["success"]]
    arms = {}
    for arm in ARMS:
        rows = [case["arms"][arm] for case in cases]
        firsts = [
            row["first_success_attempt"]
            for row in rows
            if row["first_success_attempt"] is not None
        ]
        rescue = [
            case["episode_id"]
            for case in baseline_failed
            if case["arms"][arm]["first_success_attempt"] is not None
        ]
        pair_by_attempt = []
        for iteration in range(1, 6):
            pairs = {
                name: [] for name in ("both", "arm_only", "random_only", "neither")
            }
            for case in baseline_failed:
                a = case["arms"][arm]["success_by_attempt"][iteration - 1]
                b = case["arms"]["random_noise"]["success_by_attempt"][iteration - 1]
                key = (
                    "both"
                    if a and b
                    else "arm_only"
                    if a
                    else "random_only"
                    if b
                    else "neither"
                )
                pairs[key].append(case["episode_id"])
            pair_by_attempt.append(
                {
                    "attempt": iteration,
                    "denominator": len(baseline_failed),
                    "counts": {key: len(ids) for key, ids in pairs.items()},
                    "episode_ids": pairs,
                }
            )
        actual = [r for case_records in records for r in case_records[arm]]
        cap_records = [
            r
            for case, case_records in zip(cases, records, strict=True)
            for r in case_records[arm]
            if (
                case["arms"][arm]["first_success_attempt"] is None
                or r["iteration"] <= case["arms"][arm]["first_success_attempt"]
            )
        ]
        executed_proposals = [
            attempt["proposal"]
            for row in rows
            for attempt in row["attempts"][1:]
            if attempt["rollout_executed"]
        ]
        rescued_cases = [
            case for case in baseline_failed if case["episode_id"] in rescue
        ]
        rescue_iterations = [
            case["arms"][arm]["intervention_iterations_to_success"]
            for case in rescued_cases
        ]
        rescued_records = [
            r
            for case, case_records in zip(cases, records, strict=True)
            if case["episode_id"] in rescue
            for r in case_records[arm]
            if r["iteration"] <= case["arms"][arm]["first_success_attempt"]
        ]
        arms[arm] = {
            "cases": len(cases),
            "successes_by_attempt": [
                sum(row["success_by_attempt"][i] for row in rows) for i in range(5)
            ],
            "first_success_attempt_counts": {
                str(i): firsts.count(i) for i in range(1, 6)
            },
            "median_attempt_to_success_among_successes": statistics.median(firsts)
            if firsts
            else None,
            "median_intervention_iterations_among_rescued_cases": (
                statistics.median(rescue_iterations) if rescue_iterations else None
            ),
            "censored_cases": len(cases) - len(firsts),
            "censoring_attempt": 5,
            "conditional_rescue": {
                "baseline_failed_cases": len(baseline_failed),
                "rescued_cases": len(rescue),
                "rescue_rate": len(rescue) / len(baseline_failed)
                if baseline_failed
                else None,
                "rescued_episode_ids": rescue,
            },
            "paired_rescue_vs_random_noise_by_attempt": pair_by_attempt,
            "proposal_failures": sum(row["proposal_failures"] for row in rows),
            "executed_intervention_rollouts": sum(
                attempt["rollout_executed"]
                for row in rows
                for attempt in row["attempts"][1:]
            ),
            "cases_with_executed_interventions": sum(
                any(attempt["rollout_executed"] for attempt in row["attempts"][1:])
                for row in rows
            ),
            "executed_revisions_with_nonzero_requested_parameters": sum(
                any(_requested_nonzero(proposal).values())
                for proposal in executed_proposals
            ),
            "nonzero_requested_channels_in_executed_revisions": {
                channel: sum(
                    _requested_nonzero(proposal)[channel]
                    for proposal in executed_proposals
                )
                for channel in ("noise", "language", "vision")
            },
            "comparison_to_random_noise": (
                "matched_basis_and_operator_prior_source"
                if arm == "noise_only"
                else "random_reference"
                if arm == "random_noise"
                else "different_intervention_operator_and_proposal_source"
            ),
            "standalone_cost_through_success_or_cap": _sum_cost(
                [row["standalone_cost_through_success_or_cap"] for row in rows]
            ),
            "token_usage_actual": summarize_calls(actual),
            "token_usage_through_success_or_cap": summarize_calls(cap_records),
            "tokens_to_first_success_among_rescued_cases": {
                "cases": len(rescued_cases),
                "usage": summarize_calls(rescued_records),
                "case_usage": {
                    case["episode_id"]: case["arms"][arm]["tokens_to_first_success"]
                    for case in rescued_cases
                },
            },
            "cumulative_token_usage_by_attempt_budget": [
                {
                    "attempt_budget": i,
                    "through_success_or_cap": summarize_calls(
                        [r for r in cap_records if r["iteration"] <= i]
                    ),
                    "actual_including_development_hooks": summarize_calls(
                        [r for r in actual if r["iteration"] <= i]
                    ),
                }
                for i in range(1, 6)
            ],
        }
    physical = _sum_cost(
        [
            {
                key: value
                for key, value in case["physical_cost"].items()
                if key != "token_usage"
            }
            for case in cases
        ]
    )
    physical["token_usage"] = summarize_calls(
        [row for case_records in records for arm in ARMS for row in case_records[arm]]
    )
    return {
        "cases": len(cases),
        "baseline_successes": len(cases) - len(baseline_failed),
        "arms": arms,
        "physical_cost": physical,
        "controls": {
            mode: {
                "cases": len(cases),
                "successes": sum(case["controls"][mode]["success"] for case in cases),
                "cost": _cost([case["controls"][mode] for case in cases]),
            }
            for mode in CONTROLS
        },
    }


def build_report(directories, *, phase):
    """Require every prespecified case before producing a complete result."""
    require(
        phase in ("development", "evaluation"),
        "Explicit development/evaluation phase required",
    )
    count = 2 if phase == "development" else 8
    directories = [Path(path).resolve() for path in directories]
    require(
        len(directories) == len(set(directories)) == count,
        f"{phase} requires exactly {count} distinct worker folders",
    )
    inputs = Inputs()
    workers, cases, ledgers, seen_workers, manifests = [], [], [], set(), {}
    common_protocol = common_checkpoint = common_packages = None
    common_payload = None
    for directory in sorted(directories):
        require(
            not (directory / "failure.json").exists(),
            "Worker recorded an execution failure",
        )
        runtime = inputs.read(directory / "runtime.json")
        worker = runtime["worker"]
        require(
            type(worker) is int
            and worker in range(count)
            and worker not in seen_workers,
            "Duplicate or invalid worker assignment",
        )
        seen_workers.add(worker)
        target = _assignment(phase, worker)
        require(
            runtime["phase"] == phase
            and runtime["assignment"] == target
            and runtime["tf32"] is False
            and "L40S" in runtime["gpu"]
            and _sha(runtime["payload_sha256"])
            and bool(runtime["workflow"]),
            "Runtime differs from frozen phase/worker/GPU settings",
        )
        protocol = inputs.read(directory / "protocol.json")
        _protocol(protocol)
        checkpoint = inputs.read(directory / "checkpoint.json")
        require(
            checkpoint["frozen"] is True
            and checkpoint["training_overlap"] == "unknown"
            and checkpoint["input_profile"] == "openpi_libero",
            "Unexpected policy identity",
        )
        if common_protocol is None:
            common_protocol, common_checkpoint, common_packages = (
                protocol,
                checkpoint,
                runtime["packages"],
            )
            common_payload = runtime["payload_sha256"]
        require(
            protocol == common_protocol
            and checkpoint == common_checkpoint
            and runtime["packages"] == common_packages,
            "Workers differ in protocol/checkpoint/runtime packages",
        )
        require(
            runtime["payload_sha256"] == common_payload,
            "Workers use different frozen source payloads",
        )
        benchmark = BenchmarkConfig.preset(target["suite"])
        manifest_path = directory / "reset_manifest.json"
        raw = inputs.read(manifest_path)
        manifest = load_reset_manifest(manifest_path, benchmark)
        require(
            raw == manifest
            and manifest["seed"] == 19
            and manifest["cases"] == protocol[f"{phase}_cases"]
            and manifest["split"]
            == ("development" if phase == "development" else "followup_adaptation"),
            "Frozen reset manifest differs from the fixed phase cases",
        )
        suite = target["suite"]
        require(
            manifests.get(suite, manifest["sha256"]) == manifest["sha256"],
            "Full reset manifests differ across workers in the same suite",
        )
        manifests[suite] = manifest["sha256"]
        entries = manifest["episodes"][target["case_shard"] :: target["case_shards"]]
        plan = inputs.read(directory / "frozen_plan.json")
        require(
            plan["runtime"] == runtime
            and plan["protocol_sha256"] == digest(protocol)
            and plan["manifest_sha256"] == manifest["sha256"]
            and plan["assigned_episodes"] == [entry["episode_id"] for entry in entries],
            "Frozen plan differs from the prespecified worker assignment",
        )
        progress = inputs.read(directory / "progress.json")
        require(
            progress["status"] == "complete"
            and progress["complete_cases"] == progress["assigned_cases"] == len(entries)
            and all(progress[key] == value for key, value in target.items()),
            "Worker is incomplete or has wrong case counts",
        )
        expected_dirs = {
            f"case_{e['task_id']}_{e['initial_state_id']}" for e in entries
        }
        require(
            {p.name for p in directory.glob("case_*") if p.is_dir()} == expected_dirs,
            "Missing or extra case directories",
        )
        original, worker_cases = [], []
        for entry in entries:
            case_dir = (
                directory / f"case_{entry['task_id']}_{entry['initial_state_id']}"
            )
            case, source_report, records = _case(
                inputs, case_dir, entry, benchmark, protocol, phase, checkpoint
            )
            case["workflow"], case["worker"] = runtime["workflow"], worker
            case["manifest_sha256"] = manifest["sha256"]
            case["frozen_plan_file_sha256"] = inputs.source(
                directory / "frozen_plan.json"
            )["sha256"]
            cases.append(case)
            ledgers.append(records)
            original.append(source_report)
            worker_cases.append(case)
        require(
            inputs.read(directory / "aggregate.json")
            == aggregate_reports(original, protocol),
            "Worker aggregate disagrees with complete case reports",
        )
        if phase == "development" and "development_stopping" in protocol:
            validation = {
                case["episode_id"]: {
                    arm: any(
                        row["rollout_executed"]
                        for row in case["arms"][arm]["attempts"][1:]
                    )
                    for arm in ARMS
                }
                for case in worker_cases
            }
            require(
                inputs.read(directory / "development_validation.json") == validation
                and all(
                    value is True
                    for arms in validation.values()
                    for value in arms.values()
                ),
                "Development validation failed to execute every intervention arm",
            )
        workers.append(
            {
                "runtime": runtime,
                "assigned_episodes": plan["assigned_episodes"],
                "reset_manifest_sha256": manifest["sha256"],
                "frozen_plan_source": inputs.source(directory / "frozen_plan.json"),
                "case_wall_seconds_sum": sum(
                    case["physical_cost"]["case_wall_seconds"] for case in worker_cases
                ),
            }
        )
    require(
        len(cases)
        == len({case["episode_id"] for case in cases})
        == (2 if phase == "development" else 20),
        "Incomplete or duplicate fixed episode coverage",
    )
    ordered = sorted(
        zip(cases, ledgers, strict=True),
        key=lambda pair: (pair[0]["suite"], pair[0]["task_id"]),
    )
    cases, ledgers = map(list, zip(*ordered, strict=True))
    inputs.finish()
    module = Path(__file__).resolve()
    dependencies = (
        "config.py",
        "records.py",
        "intervention_rollout.py",
        "intervention_agent.py",
        "intervention_search.py",
        "interventions.py",
        "intervention_conditioning.py",
        "libero_runner.py",
    )
    aggregate = _aggregate(cases, ledgers)
    hooks = all(
        aggregate["arms"][arm]["cases_with_executed_interventions"] == len(cases)
        for arm in ARMS
        if arm != "random_noise"
    )
    accepted = aggregate["physical_cost"]["token_usage"]["accepted_proposals"]
    calls = aggregate["physical_cost"]["token_usage"]["provider_calls"]
    executed = sum(
        aggregate["arms"][arm]["executed_intervention_rollouts"]
        for arm in ARMS
        if arm != "random_noise"
    )
    attempted = sum(
        len(case["arms"][arm]["attempts"]) - 1
        for case in cases
        for arm in ARMS
        if arm != "random_noise"
    )
    integration = {
        "status": ("passed" if hooks else "failed")
        if phase == "development"
        else (
            "provider_and_execution_observed"
            if accepted and executed
            else "failed"
            if attempted
            else "not_exercised"
        ),
        "all_forced_development_hooks_executed": hooks
        if phase == "development"
        else None,
        "provider_calls": calls,
        "accepted_proposals": accepted,
        "executed_astra_intervention_rollouts": executed,
        "attempted_astra_revisions": attempted,
        "executed_astra_revisions_with_nonzero_requested_parameters": sum(
            aggregate["arms"][arm][
                "executed_revisions_with_nonzero_requested_parameters"
            ]
            for arm in ARMS
            if arm != "random_noise"
        ),
        "fixed_development_embedding_probe_cases": sum(
            case["initialization"].get("development_embedding_probe") is not None
            for case in cases
        ),
        "baseline_failed_cases": len(cases) - aggregate["baseline_successes"],
        "interpretation": "Shared baseline successes do not demonstrate intervention efficacy. "
        "Recording completeness is separate from genuine Astra proposal/hook execution. "
        "Neutral accepted proposals exercise the integration but do not establish a nonzero effect. "
        "The fixed development probe is separate from Astra proposals and environment rollouts.",
    }
    random_checks = [
        check for case in cases for check in case["random_noise_reconstruction"]
    ]
    report = {
        "schema_version": "intervention_report_v1",
        "status": "complete_verified_recording",
        "phase": phase,
        "expected_cases": 2 if phase == "development" else 20,
        "expected_workers": count,
        "protocol": common_protocol,
        "protocol_sha256": digest(common_protocol),
        "checkpoint": common_checkpoint,
        "verified_full_reset_manifest_sha256": manifests,
        "audit": {
            "full_fixed_case_coverage": True,
            "shared_baselines_verified": True,
            "paired_full_reset_audits_verified": True,
            "provider_event_ledger_verified": True,
            "zero_action_successes": 0,
            "legacy_null_usage_normalized_to_verified_zero": sum(
                len(case["legacy_null_usage_verified_zero_arms"]) for case in cases
            ),
            "pre_http_proposal_errors": sum(
                len(case["pre_http_proposal_errors"]) for case in cases
            ),
            "numerical_errors_independently_recomputed": False,
            "native_resets_independently_replayed": False,
            "random_noise_coefficient_reconstruction": {
                "maximum_allowed_ulps": RANDOM_COEFFICIENT_ULPS,
                "proposals_checked": len(random_checks),
                "nonexact_proposals": sum(
                    not check["coefficients_exact"] for check in random_checks
                ),
                "maximum_abs_difference": max(
                    (
                        check["coefficient_max_abs_difference"]
                        for check in random_checks
                    ),
                    default=0.0,
                ),
                "maximum_ulp_difference": max(
                    (
                        check["coefficient_max_ulp_difference"]
                        for check in random_checks
                    ),
                    default=0.0,
                ),
            },
        },
        "summary": aggregate,
        "integration": integration,
        "cases": cases,
        "workers": sorted(workers, key=lambda row: row["runtime"]["worker"]),
        "source_files": sorted(inputs.sources.values(), key=lambda row: row["path"]),
        "verified_absent_provider_ledgers": sorted(inputs.absent),
        "report_source_sha256": file_sha256(module),
        "report_dependency_sha256": {
            name: file_sha256(module.with_name(name)) for name in dependencies
        },
        "limitations": LIMITATIONS,
    }
    report["sha256"] = digest(report)
    return report


def _table_rows(report):
    result = []
    for arm, value in report["summary"]["arms"].items():
        usage = value["token_usage_through_success_or_cap"]
        actual = value["token_usage_actual"]
        paired = value["paired_rescue_vs_random_noise_by_attempt"][-1]["counts"]
        row = {
            "arm": arm,
            "cases": value["cases"],
            **{
                f"success_by_attempt_{i}": value["successes_by_attempt"][i - 1]
                for i in range(1, 6)
            },
            "median_first_success_among_successes": value[
                "median_attempt_to_success_among_successes"
            ],
            "median_intervention_iterations_among_rescued_cases": value[
                "median_intervention_iterations_among_rescued_cases"
            ],
            "censored_cases": value["censored_cases"],
            "rescued_cases": value["conditional_rescue"]["rescued_cases"],
            "baseline_failed_cases": value["conditional_rescue"][
                "baseline_failed_cases"
            ],
            "rescue_arm_only_vs_random": paired["arm_only"],
            "rescue_random_only": paired["random_only"],
            "provider_calls_through_success_or_cap": usage["provider_calls"],
            "actual_provider_calls": actual["provider_calls"],
            "actual_failed_provider_calls": actual["failed_calls"],
            "executed_intervention_rollouts": value["executed_intervention_rollouts"],
            "proposal_failures": value["proposal_failures"],
            **{
                f"standalone_{key}": metric
                for key, metric in value[
                    "standalone_cost_through_success_or_cap"
                ].items()
            },
        }
        for field in TOKEN_FIELDS:
            for prefix, source in (
                ("through_success_or_cap", usage),
                ("actual", actual),
            ):
                row[f"{prefix}_{field}"] = source["tokens"][field]["sum"]
                row[f"{prefix}_{field}_missing_calls"] = source["tokens"][field][
                    "missing_calls"
                ]
        result.append(row)
    return result


def _markdown(report):
    summary = report["summary"]
    integration = report["integration"]
    lines = [
        f"Iterative intervention {report['phase']}: {summary['cases']} verified case recordings.",
        "",
        f"Astra integration: **{integration['status']}**. "
        f"{integration['accepted_proposals']}/{integration['provider_calls']} provider proposals accepted; "
        f"{integration['executed_astra_intervention_rollouts']} Astra intervention rollouts executed. "
        "Shared baseline successes are not evidence of intervention benefit.",
        "",
        f"Common identity baseline: {summary['baseline_successes']}/{summary['cases']}. "
        f"Known-noise control: {summary['controls']['known_noise']['successes']}/{summary['cases']}. "
        f"Fresh-noise control: {summary['controls']['policy_fresh']['successes']}/{summary['cases']}.",
        "",
        "Counts are cumulative successes by attempt; the baseline is attempt 1. "
        "Rescue is conditional on baseline failure. Token costs run through first success or cap.",
        "",
        "| Arm | 1 | 2 | 3 | 4 | 5 | Median first success¹ | Median rescue revisions | Censored | Rescue | Calls | Input / output / reasoning² |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for arm, row in summary["arms"].items():
        usage = row["token_usage_through_success_or_cap"]
        tokens = []
        for field in TOKEN_FIELDS[:3]:
            value = usage["tokens"][field]
            tokens.append(
                str(value["sum"])
                + (
                    f" ({value['missing_calls']} missing)"
                    if value["missing_calls"]
                    else ""
                )
            )
        rescue = row["conditional_rescue"]
        median = row["median_attempt_to_success_among_successes"]
        rescue_median = row["median_intervention_iterations_among_rescued_cases"]
        cells = [
            arm,
            *row["successes_by_attempt"],
            median if median is not None else "—",
            rescue_median if rescue_median is not None else "—",
            row["censored_cases"],
            f"{rescue['rescued_cases']}/{rescue['baseline_failed_cases']}",
            usage["provider_calls"],
            " / ".join(tokens),
        ]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    physical = summary["physical_cost"]
    physical_tokens = []
    for name in ("input", "output", "total", "reasoning"):
        value = physical["token_usage"]["tokens"][name + "_tokens"]
        if not value["available_calls"] and value["missing_calls"]:
            description = f"unknown ({value['missing_calls']} calls missing usage)"
        elif value["missing_calls"]:
            description = f"{value['sum']:,} observed (partial; {value['missing_calls']} calls missing usage)"
        else:
            description = f"{value['sum']:,}"
        physical_tokens.append(f"{name} {description}")
    lines += [
        "",
        "¹ Among successful cases only. ² Reasoning is included in output tokens. "
        "A token sum marked missing is partial, including failed provider calls when usage exists.",
        "",
        f"Physical execution: {physical['rollouts']} rollouts, {physical['simulated_actions']} actions, "
        f"{physical['total_velocity_evaluations']} velocity evaluations "
        f"({physical['initialization_velocity_evaluations']} for initialization), "
        f"{physical['token_usage']['provider_calls']} provider calls. "
        "Per-arm standalone costs and paired rescue counts against random search are in report.json and arms.csv.",
        "Physical provider tokens: "
        + "; ".join(physical_tokens)
        + ". Reasoning is included in output tokens.",
        f"Physical provider usage is unavailable for {physical['token_usage']['usage_unavailable_calls']} "
        f"of {physical['token_usage']['provider_calls']} calls; missing usage is not zero cost. "
        "budgets.csv separates cumulative search-to-success tokens from actual development-hook tokens.",
        "",
        f"Report source SHA-256: `{report['report_source_sha256']}`. "
        f"Report content digest: `{report['sha256']}`.",
        "",
    ]
    lines.extend(f"- {note}" for note in report["limitations"])
    return "\n".join(lines) + "\n"


def write_outputs(directory, report):
    """Only create a fresh analysis directory; never modify downloaded reports."""
    directory = Path(directory).resolve()
    require(
        report["sha256"] == digest({k: v for k, v in report.items() if k != "sha256"}),
        "Report content changed before writing",
    )
    require(
        all(
            not Path(source["path"]).is_relative_to(directory)
            for source in report["source_files"]
        ),
        "Output cannot contain source inputs",
    )
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    rows = _table_rows(report)
    with (directory / "arms.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_case = [
        {
            "episode_id": case["episode_id"],
            "suite": case["suite"],
            "task_id": case["task_id"],
            "arm": arm,
            "baseline_success": case["baseline"]["success"],
            "first_success_attempt": value["first_success_attempt"],
            "intervention_iterations_to_success": value[
                "intervention_iterations_to_success"
            ],
            "censored_after_attempt": value["censored_after_attempt"],
            "provider_calls_through_success_or_cap": value[
                "token_usage_through_success_or_cap"
            ]["provider_calls"],
            "reset_audit_sha256": case["reset_audit"]["sha256"],
            "source_summary_sha256": case["source"]["sha256"],
            "workflow": case["workflow"],
            "worker": case["worker"],
        }
        for case in report["cases"]
        for arm, value in case["arms"].items()
    ]
    with (directory / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_case[0]))
        writer.writeheader()
        writer.writerows(per_case)
    budgets = []
    for arm, value in report["summary"]["arms"].items():
        for curve in value["cumulative_token_usage_by_attempt_budget"]:
            budget = curve["attempt_budget"]
            row = {
                "arm": arm,
                "attempt_budget_including_baseline": budget,
                "cases": value["cases"],
                "successes": value["successes_by_attempt"][budget - 1],
            }
            for label in (
                "through_success_or_cap",
                "actual_including_development_hooks",
            ):
                usage = curve[label]
                row[f"{label}_provider_calls"] = usage["provider_calls"]
                row[f"{label}_failed_provider_calls"] = usage["failed_calls"]
                for field in TOKEN_FIELDS:
                    row[f"{label}_{field}"] = usage["tokens"][field]["sum"]
                    row[f"{label}_{field}_missing_calls"] = usage["tokens"][field][
                        "missing_calls"
                    ]
            budgets.append(row)
    with (directory / "budgets.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(budgets[0]))
        writer.writeheader()
        writer.writerows(budgets)
    attempts = []
    for case in report["cases"]:
        for arm, value in case["arms"].items():
            previous = summarize_calls([])
            for attempt in value["attempts"]:
                cumulative = attempt.get("cumulative_token_usage", summarize_calls([]))
                row = {
                    "episode_id": case["episode_id"],
                    "arm": arm,
                    "iteration_including_baseline": attempt["iteration"],
                    "shared_baseline_attribution": attempt["iteration"] == 1,
                    "rollout_executed": attempt["rollout_executed"],
                    "status": attempt["status"],
                    "success": attempt["success"],
                    "first_success_attempt": value["first_success_attempt"],
                    "censored_after_attempt": value["censored_after_attempt"],
                    "actions_executed": attempt["actions_executed"],
                    "velocity_evaluations": attempt["velocity_evaluations"],
                    "rollout_wall_seconds": attempt["wall_seconds"],
                    "proposal_seconds": attempt.get("proposal_seconds", 0),
                    "provider_calls": cumulative["provider_calls"]
                    - previous["provider_calls"],
                    "cumulative_provider_calls": cumulative["provider_calls"],
                    "source_summary_sha256": case["source"]["sha256"],
                }
                for field in TOKEN_FIELDS:
                    for name in ("sum", "missing_calls"):
                        row[f"{field}_{name}"] = (
                            cumulative["tokens"][field][name]
                            - previous["tokens"][field][name]
                        )
                attempts.append(row)
                previous = cumulative
    with (directory / "attempts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(attempts[0]))
        writer.writeheader()
        writer.writerows(attempts)
    (directory / "report.md").write_text(_markdown(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("development", "evaluation"))
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.inputs, phase=args.phase)
    write_outputs(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "astra_integration": report["integration"]["status"],
                "phase": report["phase"],
                "cases": report["summary"]["cases"],
                "sha256": report["sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
