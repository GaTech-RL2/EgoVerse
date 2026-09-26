"""Synthetic evidence fixtures for report coverage, estimands and cost accounting."""

import copy
import hashlib
import json

import pytest

from astra_reversal.frs_html_report import (
    SCHEMA_VERSION,
    TASKS,
    TOKEN_FIELDS,
    _read_json,
    derived_results,
    render_report,
    validate_report,
)


def _usage(calls=0, *, failed=False, missing=False):
    amounts = dict(input_tokens=8, output_tokens=5, total_tokens=13, reasoning_tokens=2)
    return {
        "calls": calls,
        "accepted_calls": 0 if failed else calls,
        "failed_calls": calls if failed else 0,
        "preflight_failures": 0,
        "tokens": {
            key: {
                "sum": 0 if missing else amount * calls,
                "missing_calls": calls if missing else 0,
            }
            for key, amount in amounts.items()
        },
    }


def _row(episode, method, physical, *, success=False, reused=False, calls=0):
    return {
        "episode_id": episode,
        "method_id": method,
        "physical_run_id": physical,
        "reused": reused,
        "success": success,
        "error": None,
        "actions": 10,
        "velocity_evaluations": 10,
        "wall_seconds": 1.5,
        "policy_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "audit": {"status": "passed", "sha256": "c" * 64},
        "provider_usage": _usage(calls),
    }


def _episodes(states):
    return [
        f"{key.split(':')[0]}:seed43:task{key.split(':')[1]}:state{state}"
        for key in TASKS
        for state in states
    ]


def _base(method_ids):
    prompt = "Synthetic test fixture. Return JSON. <script>never execute</script>"
    return {
        "schema_version": SCHEMA_VERSION,
        "title": "Synthetic fixture, not measured results",
        "status": "complete",
        "phase": "evaluation",
        "tasks": [
            {"task_key": key, "instruction": value} for key, value in TASKS.items()
        ],
        "protocol": {
            "seed": 43,
            "adaptation_state": 0,
            "evaluation_states": list(range(1, 11)),
            "rounds": 3,
            "evaluation_methods": ["native_euler10", "learned_noise"],
            "adaptation_methods": ["critique_frs_no_learning", "critique_frs_learning"],
        },
        "methods": [
            {
                "id": name,
                "label": name,
                "description": "Synthetic <script>fixture</script>",
            }
            for name in method_ids
        ],
        "prompts": [
            {
                "id": "test_prompt",
                "text": prompt,
                "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "scope": "Synthetic fixture only",
            }
        ],
        "notes": ["No empirical result is represented by this fixture."],
        "sources": [
            {"label": "Primary paper", "href": "https://arxiv.org/abs/2606.13675v2"}
        ],
        "cohorts": [],
        "overheads": [],
    }


def _overhead(name="setup", *, attribution=None):
    return {
        "id": name,
        "label": "Synthetic failed provider call and training overhead",
        "provider_usage": _usage(1, failed=True),
        "velocity_evaluations": 50,
        "training_steps": 1000,
        "wall_seconds": 7.0,
        "source_sha256": "d" * 64,
        "attribution": attribution,
    }


def _checkpoint_report():
    methods = ["native_euler10", "learned_noise"]
    value = _base(methods)
    episodes = _episodes(range(1, 11))
    rounds = []
    for index in range(1, 4):
        rows = []
        for number, episode in enumerate(episodes):
            rows.append(
                _row(
                    episode,
                    methods[0],
                    f"baseline:{episode}",
                    success=number < 4,
                    reused=index > 1,
                )
            )
            row = _row(
                episode,
                methods[1],
                f"round{index}:{episode}",
                success=index == 2,
            )
            row["policy_sha256"] = str(index) * 64
            rows.append(row)
        rounds.append(
            {
                "index": index,
                "label": f"After round {index}",
                "method_ids": methods[:],
                "episodes": rows,
            }
        )
    value["cohorts"] = [
        {
            "id": "evaluation",
            "label": "Held-out resets of known tasks",
            "status": "complete",
            "curve_kind": "checkpoint_evaluation",
            "expected_episode_ids": episodes,
            "method_ids": methods,
            "rounds": rounds,
        }
    ]
    value["overheads"] = [_overhead()]
    return value


def _retry_report():
    methods = ["critique_frs_no_learning", "critique_frs_learning"]
    value = _base(methods)
    episodes = _episodes([0])
    rounds = []
    for index in range(4):
        rows = []
        for number, episode in enumerate(episodes):
            for arm, method in enumerate(methods):
                row = _row(
                    episode,
                    method,
                    f"common:{episode}"
                    if index == 0
                    else f"{method}:{index}:{episode}",
                    success=(index == 0 and number == 0)
                    or (arm == 0 and (number, index) in ((1, 1), (2, 2)))
                    or (arm == 1 and (number, index) == (1, 2)),
                    reused=index == 0 and arm == 1,
                    calls=int(index > 0),
                )
                if arm == 0 and (number, index) == (2, 1):
                    row["provider_usage"] = _usage(1, missing=True)
                rows.append(row)
        rounds.append(
            {
                "index": index,
                "label": f"Revision {index}",
                "method_ids": methods[:],
                "episodes": rows,
            }
        )
    value["cohorts"] = [
        {
            "id": "adaptation",
            "label": "Reset-assisted adaptation",
            "status": "complete",
            "curve_kind": "best_of_attempts",
            "expected_episode_ids": episodes,
            "method_ids": methods,
            "rounds": rounds,
        }
    ]
    value["overheads"] = [
        _overhead(
            "critique-call",
            attribution={
                "cohort_id": "adaptation",
                "episode_id": episodes[1],
                "method_id": methods[0],
                "round_index": 1,
            },
        )
    ]
    return value


def test_checkpoint_curve_can_fall_and_shared_native_cost_is_not_duplicated():
    value = _checkpoint_report()
    checked = validate_report(value)
    derived = derived_results(value)["evaluation"]
    assert [point["successes"] for point in derived["learned_noise"]["points"]] == [
        0,
        200,
        0,
    ]
    assert all(point["episodes"] == 200 for point in derived["learned_noise"]["points"])
    assert derived["native_euler10"]["standalone_unique_rollouts"] == 200
    assert checked["physical_cost"]["unique_rollouts"] == 800
    assert checked["physical_cost"]["velocity_evaluations"] == 8050
    assert checked["physical_cost"]["training_steps"] == 1000
    assert checked["physical_cost"]["provider_usage"] == _usage(1, failed=True)


def test_retry_rescue_conditioning_censoring_and_failed_call_tokens():
    value = _retry_report()
    checked = validate_report(value)
    detail = derived_results(value)["adaptation"]["critique_frs_no_learning"]
    assert [point["successes"] for point in detail["points"]] == [1, 2, 3, 3]
    assert detail["rescue"] == {
        "baseline_failed": 19,
        "rescued": 2,
        "censored": 17,
        "median_revision_among_rescues": 1.5,
        "median_first_success_among_successes": 1,
        "median_tokens_among_rescues_with_complete_usage": 26,
        "rescues_missing_usage": 1,
    }
    assert checked["physical_cost"]["unique_rollouts"] == 140
    assert checked["physical_cost"]["provider_usage"]["calls"] == 121
    assert checked["physical_cost"]["provider_usage"]["failed_calls"] == 1
    assert detail["provider_usage"]["calls"] == 61


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing_episode", "exact episode"),
        ("duplicate_episode", "Duplicate, unexpected"),
        ("wrong_reset", "denominator differs"),
        ("missing_method", "frozen protocol method"),
        ("missing_round", "checkpoint evaluation round"),
        ("pending_audit", "passed audits"),
        ("zero_action_success", "Zero-action"),
        ("execution_error", "passed audits"),
        ("cost_mutation_on_reuse", "changed evidence/cost"),
        ("reuse_not_marked", "reuse marker"),
    ],
)
def test_incomplete_or_inconsistent_records_cannot_be_final(mutation, message):
    value = _checkpoint_report()
    cohort = value["cohorts"][0]
    rounds = cohort["rounds"]
    if mutation == "missing_episode":
        rounds[0]["episodes"].pop()
    elif mutation == "duplicate_episode":
        rounds[0]["episodes"].append(copy.deepcopy(rounds[0]["episodes"][0]))
    elif mutation == "wrong_reset":
        cohort["expected_episode_ids"][0] = cohort["expected_episode_ids"][0].replace(
            "seed43", "seed44"
        )
    elif mutation == "missing_method":
        cohort["method_ids"] = ["native_euler10"]
    elif mutation == "missing_round":
        rounds.pop(1)
    elif mutation == "pending_audit":
        rounds[0]["episodes"][0]["audit"] = {"status": "pending"}
    elif mutation == "zero_action_success":
        rounds[0]["episodes"][0]["actions"] = 0
    elif mutation == "execution_error":
        rounds[0]["episodes"][1]["error"] = "CUDA error"
    elif mutation == "cost_mutation_on_reuse":
        rounds[1]["episodes"][0]["wall_seconds"] = 99
    elif mutation == "reuse_not_marked":
        rounds[1]["episodes"][0]["reused"] = False
    with pytest.raises(ValueError, match=message):
        validate_report(value)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("total", "input plus output"),
        ("reasoning", "subset of output"),
        ("missing_positive", "no known token sum"),
        ("unknown_calls", "sum to physical calls"),
        ("zero_calls", "no known token sum"),
        ("bool_calls", "nonnegative integer"),
    ],
)
def test_token_cost_does_not_hide_unknowns_or_double_count_reasoning(mutation, message):
    value = _checkpoint_report()
    usage = value["overheads"][0]["provider_usage"]
    if mutation == "total":
        usage["tokens"]["total_tokens"]["sum"] += 2
    elif mutation == "reasoning":
        usage["tokens"]["reasoning_tokens"]["sum"] = 6
    elif mutation == "missing_positive":
        usage["tokens"]["total_tokens"]["missing_calls"] = 1
    elif mutation == "unknown_calls":
        usage["failed_calls"] = 0
    elif mutation == "zero_calls":
        usage["calls"] = usage["failed_calls"] = 0
    elif mutation == "bool_calls":
        usage["calls"] = True
    with pytest.raises(ValueError, match=message):
        validate_report(value)


def test_unknown_usage_is_reported_as_known_sum_and_missing_count():
    value = _checkpoint_report()
    value["overheads"][0]["provider_usage"] = _usage(2, missing=True)
    usage = validate_report(value)["physical_cost"]["provider_usage"]
    assert all(
        usage["tokens"][key] == {"sum": 0, "missing_calls": 2} for key in TOKEN_FIELDS
    )


def test_complete_provider_study_requires_exact_prompt_text():
    value = _checkpoint_report()
    value["prompts"][0]["text"] += " changed"
    with pytest.raises(ValueError, match="Exact prompt text"):
        validate_report(value)
    value["prompts"] = []
    with pytest.raises(ValueError, match="exact prompt text"):
        validate_report(value)


def test_partial_report_never_emits_efficacy_even_with_a_complete_subcohort(tmp_path):
    value = _checkpoint_report()
    value["status"] = "partial"
    assert derived_results(value) == {}
    value["cohorts"][0]["status"] = "partial"
    value["cohorts"][0]["rounds"] = value["cohorts"][0]["rounds"][:1]
    value["cohorts"][0]["rounds"][0]["episodes"] = value["cohorts"][0]["rounds"][0][
        "episodes"
    ][:2]
    source = tmp_path / "input.json"
    source.write_text(json.dumps(value))
    output = tmp_path / "partial"
    render_report(source, output)
    markup = (output / "index.html").read_text()
    assert "Pending complete study/audit" in markup
    assert "Final success" not in markup
    assert "Success (%)" not in markup
    assert not list(output.glob("*_curves.svg"))


def test_portable_bundle_preserves_input_escapes_text_and_binds_every_file(tmp_path):
    source = tmp_path / "input.json"
    raw = (json.dumps(_checkpoint_report(), indent=3) + "\n").encode()
    source.write_bytes(raw)
    output = tmp_path / "complete"
    manifest = render_report(source, output)
    assert (output / "report.json").read_bytes() == raw
    assert manifest["input_sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest == json.loads((output / "manifest.json").read_text())
    markup = (output / "index.html").read_text()
    assert "<script" not in markup
    assert "&lt;script&gt;" in markup
    assert "<svg" in markup
    assert "Success (%)" in markup
    assert "Censored" not in markup  # Checkpoints are not cumulative retry searches.
    assert "physical provider calls" in markup
    assert "reasoning tokens" in markup
    for name, row in manifest["files"].items():
        data = (output / name).read_bytes()
        assert row == {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    with pytest.raises(ValueError, match="overwrite"):
        render_report(source, output)


def test_unsafe_source_url_and_changed_task_inventory_fail_before_publication(tmp_path):
    value = _checkpoint_report()
    value["sources"][0]["href"] = "javascript:alert(1)"
    with pytest.raises(ValueError, match="public HTTP"):
        validate_report(value)
    value["sources"] = []
    value["tasks"][2]["instruction"] = "put the bbq sauce on the plate"
    source = tmp_path / "bad.json"
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Task names"):
        render_report(source, tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_duplicate_and_nonfinite_json_is_rejected(tmp_path, text):
    source = tmp_path / "bad.json"
    source.write_text(text)
    with pytest.raises(ValueError):
        _read_json(source)
