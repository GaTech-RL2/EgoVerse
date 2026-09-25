"""Audit gating, paired denominators and measured costs; synthetic data only."""

import copy
import json
from pathlib import Path

import pytest

from astra_reversal.image_perturbation_agent import summarize_calls
from astra_reversal.image_perturbation_report import (
    _paired,
    build_report,
    write_outputs,
)
from astra_reversal.image_perturbation_search import arm_summary
from astra_reversal.records import digest, file_sha256

from .test_image_perturbation_search import image_search_factory  # noqa: F401


def receipt_for(directory, summary):
    """Simulated receipt, not evidence of an actual numerical audit."""
    physical = [summary["baseline"], *summary["controls"].values()] + [
        row for arm in summary["arms"].values() for row in arm["attempts"][1:]
    ]
    cost = summary["physical_cost"]
    return {
        "schema_version": "image-perturbation-audit-1.0",
        "status": "passed",
        "complete": True,
        "episode_id": summary["episode_id"],
        "protocol_sha256": summary["protocol_sha256"],
        "library_id": summary["image_library"]["library_id"],
        "summary_sha256": file_sha256(Path(directory) / "summary.json"),
        "reset_entry_sha256": summary["reset_entry_sha256"],
        "events_sha256": "b" * 64,
        "array_inventory_sha256": "c" * 64,
        "all_arrays_verified": True,
        "provider_bindings_verified": True,
        "reset_pairing_verified": True,
        "decision_lifetimes_verified": True,
        "source_sha256": {"synthetic_auditor": "d" * 64},
        "input_file_sha256": {"synthetic_input": "e" * 64},
        "counts": {
            "physical_rollouts": cost["rollouts"],
            "actions": cost["simulated_actions"],
            "velocity_evaluations": cost["velocity_evaluations"],
            "provider_calls": cost["token_usage"]["provider_calls"],
            "preflight_failures": cost["token_usage"]["preflight_failures"],
            "accepted_proposals": cost["token_usage"]["accepted_proposals"],
            "executed_decisions": sum(
                row["accepted_decisions_executed"] for row in physical
            ),
        },
        "provider": copy.deepcopy(cost["token_usage"]),
    }


def mark_short_synthetic_controls_terminated(search, summary):
    # The integration harness stops controls at ten steps for speed. Express
    # that synthetic stop truthfully before testing the stricter report gate.
    for control in summary["controls"].values():
        if (
            not control["success"]
            and control["actions_executed"] < search.protocol["action_budget"]
        ):
            control["terminated"] = True
            control["status"] = "terminated"
    save_summary(search, summary)


@pytest.fixture
def completed_case(image_search_factory):  # noqa: F811
    search, _, _ = image_search_factory()
    # The harness pins synthetic donor pixels rather than published benchmark data.
    search.protocol["image_library_id"] = search.library.metadata()["library_id"]
    search.report["protocol_sha256"] = digest(search.protocol)
    summary = search.run()
    mark_short_synthetic_controls_terminated(search, summary)
    receipt = receipt_for(search.directory, summary)
    return search, summary, receipt


def save_summary(search, summary):
    (search.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def call_report(search, *, receipts=None, expected=None, directories=None):
    return build_report(
        directories or [search.directory],
        expected_episode_ids=expected or [search.entry["episode_id"]],
        protocol=search.protocol,
        audit_receipts=receipts,
    )


def test_full_audited_report_counts_failed_call_and_baseline_once(
    completed_case, tmp_path
):
    search, summary, receipt = completed_case
    episode_id = summary["episode_id"]
    report = call_report(search, receipts={episode_id: receipt})
    assert report["status"] == "complete" and report["efficacy_released"]
    pooled = report["groups"]["pooled"]
    assert pooled["physical_cost"]["rollouts"] == 12
    assert pooled["physical_cost"]["simulated_actions"] == 320
    assert pooled["physical_cost"]["velocity_evaluations"] == 1930
    usage = pooled["physical_cost"]["token_usage"]
    assert usage["provider_calls"] == 6 and usage["accepted_proposals"] == 5
    assert usage["tokens"]["total_tokens"]["sum"] == 78
    assert (
        pooled["physical_cost"]["rejected_call_usage"]["tokens"]["total_tokens"]["sum"]
        == 13
    )
    assert pooled["physical_cost"]["fallback_actions"] == 5
    arm = pooled["arms"]["astra_occlusion"]
    assert arm["rescues"] == 1 and arm["baseline_successes"] == 0
    assert arm["rescue_only"]["first_success_attempt"] == {
        "count": 1,
        "minimum": 3,
        "median": 3,
        "maximum": 3,
    }
    assert arm["rescue_only"]["full_rollout_revisions"]["median"] == 2
    assert arm["rescue_only"]["online_decisions_through_success"]["median"] == 4
    assert arm["rescue_only"]["complete_total_tokens_to_success"]["median"] == 52
    pair = pooled["paired_vs_random"]["astra_occlusion"]
    assert pair["on_failed_baselines"] == {
        "both_rescue": 0,
        "astra_only_rescue": 1,
        "random_only_rescue": 0,
        "neither_rescues": 0,
    }
    assert (
        len(report["case_arm_rows"]) == 5 and len(report["physical_attempt_rows"]) == 12
    )
    assert pooled["arms"]["random_occlusion"]["censored_without_success"] == 1
    destination = tmp_path / "safe_report"
    write_outputs(destination, report)
    assert len((destination / "physical_attempts.csv").read_text().splitlines()) == 13
    assert "Astra only 1" in (destination / "README.md").read_text()
    public = (destination / "report.json").read_text()
    assert "synthetic-image-search-key" not in public
    assert '"choices"' not in public and '"request_fingerprint"' not in public


@pytest.mark.parametrize("missing_case", [False, True])
def test_missing_audit_or_case_withholds_pooled_and_paired_claims(
    completed_case, missing_case
):
    search, summary, receipt = completed_case
    expected = [summary["episode_id"]] + (["missing-case"] if missing_case else [])
    report = call_report(
        search,
        expected=expected,
        receipts={summary["episode_id"]: receipt} if missing_case else None,
    )
    assert report["status"] == ("partial" if missing_case else "audit_pending")
    assert not report["efficacy_released"]
    assert report["groups"]["pooled"]["arms"] is None
    assert report["groups"]["pooled"]["paired_vs_random"] is None
    assert (
        report["groups"]["pooled"]["physical_cost"]["token_usage"]["provider_calls"]
        == 6
    )


@pytest.mark.parametrize(
    "field",
    [
        "summary_sha256",
        "library_id",
        "provider_bindings_verified",
        "counts",
        "provider",
    ],
)
def test_bad_receipt_never_releases_aggregated_success(completed_case, field):
    search, summary, receipt = completed_case
    bad = copy.deepcopy(receipt)
    if field == "counts":
        bad[field]["actions"] += 1
    elif field == "provider":
        bad[field]["tokens"]["total_tokens"]["sum"] += 1
    elif field == "provider_bindings_verified":
        bad[field] = False
    else:
        bad[field] = "0" * 64
    report = call_report(search, receipts={summary["episode_id"]: bad})
    assert report["status"] == "audit_failed" and not report["complete"]
    assert report["groups"]["pooled"]["paired_vs_random"] is None
    assert report["cases"][0]["audit"]["status"] == "failed"


@pytest.mark.parametrize(
    "mutation",
    [
        "physical_cost",
        "baseline_copy",
        "provider_usage",
        "missing_slot",
        "early_censor",
    ],
)
def test_summary_reconciliation_rejects_false_cost_coverage_or_stopping(
    completed_case, mutation
):
    search, summary, receipt = completed_case
    if mutation == "physical_cost":
        summary["physical_cost"]["token_usage"]["tokens"]["total_tokens"]["sum"] -= 13
    elif mutation == "baseline_copy":
        summary["arms"]["astra_occlusion"]["attempts"][0] = copy.deepcopy(
            summary["baseline"]
        )
        summary["arms"]["astra_occlusion"]["attempts"][0]["success"] = True
    elif mutation == "provider_usage":
        summary["arms"]["astra_occlusion"]["attempts"][1]["provider_records"][0][
            "token_usage"
        ]["total_tokens"] = 0
    elif mutation == "missing_slot":
        summary["arms"]["astra_occlusion"]["attempts"][1]["provider_records"].pop()
    else:
        summary["arms"]["random_occlusion"]["attempts"].pop()
    save_summary(search, summary)
    report = call_report(search, receipts={summary["episode_id"]: receipt})
    assert report["status"] == "audit_failed" and not report["efficacy_released"]
    assert report["problems"][0]["status"] == "invalid_case"


def test_unknown_failed_call_usage_remains_unknown_and_excluded_from_complete_token_median(
    completed_case,
):
    search, summary, _ = completed_case
    failed = summary["arms"]["astra_occlusion"]["attempts"][1]["provider_records"][1]
    del failed["response"]["usage"]
    failed["token_usage"] = {key: None for key in failed["token_usage"]}
    for arm in summary["arms"].values():
        arm["summary"] = arm_summary(arm["attempts"], search.budget)
        arm["summary"]["standalone_velocity_evaluations_through_success_or_cap"] = (
            arm["summary"]["velocity_evaluations_through_success_or_cap"] + 1290
        )
    search.report = summary
    summary["physical_cost"] = search.physical_cost()
    save_summary(search, summary)
    receipt = receipt_for(search.directory, summary)
    report = call_report(search, receipts={summary["episode_id"]: receipt})
    assert report["complete"]
    arm = report["groups"]["pooled"]["arms"]["astra_occlusion"]
    assert arm["rescue_only"]["rescues_with_incomplete_token_usage"] == 1
    assert arm["rescue_only"]["complete_total_tokens_to_success"]["median"] is None
    usage = report["groups"]["pooled"]["physical_cost"]["token_usage"]
    assert usage["tokens"]["total_tokens"] == {
        "sum": 65,
        "available_calls": 5,
        "missing_calls": 1,
        "complete": False,
    }


def test_baseline_wins_do_not_enter_rescue_statistics_even_with_forced_development_calls(
    image_search_factory,  # noqa: F811
):
    search, _, _ = image_search_factory(baseline_success=True, development=True)
    search.protocol["image_library_id"] = search.library.metadata()["library_id"]
    search.report["protocol_sha256"] = digest(search.protocol)
    summary = search.run()
    mark_short_synthetic_controls_terminated(search, summary)
    receipt = receipt_for(search.directory, summary)
    report = call_report(search, receipts={summary["episode_id"]: receipt})
    assert report["complete"]
    group = report["groups"]["pooled"]
    assert group["physical_cost"]["token_usage"]["tokens"]["total_tokens"]["sum"] == 52
    for arm in group["arms"].values():
        assert arm["baseline_successes"] == 1 and arm["rescues"] == 0
        assert arm["rescue_only"]["full_rollout_revisions"]["count"] == 0
    for row in report["case_arm_rows"]:
        assert row["complete_tokens_to_success"] == 0
    assert group["paired_vs_random"]["astra_demo_blend"]["baseline_failures"] == 0


def test_duplicate_and_unexpected_episode_inventories_fail_explicitly(completed_case):
    search, summary, _ = completed_case
    with pytest.raises(ValueError, match="duplicate"):
        call_report(search, directories=[search.directory, search.directory])
    with pytest.raises(ValueError, match="Unexpected"):
        call_report(search, expected=["unrelated-case"])
    with pytest.raises(ValueError, match="unique"):
        call_report(search, expected=[summary["episode_id"], summary["episode_id"]])


def test_paired_comparison_retains_all_four_outcomes_and_excludes_baseline_wins():
    cases = []
    for index, (astra, random, baseline) in enumerate(
        [
            (True, True, False),
            (True, False, False),
            (False, True, False),
            (False, False, False),
            (True, True, True),
        ]
    ):
        cases.append(
            {
                "episode_id": f"synthetic-pair-{index}",
                "baseline_success": baseline,
                "arm_rows": [
                    {
                        "arm": name,
                        "success": success,
                        "first_success_attempt": 1
                        if baseline
                        else 2
                        if success
                        else None,
                        "complete_tokens_to_success": 13
                        if success and name.startswith("astra_")
                        else None,
                        "winning_attempt_effects": None,
                    }
                    for name, success in (
                        ("astra_occlusion", astra),
                        ("random_occlusion", random),
                    )
                ],
            }
        )
    pair = _paired(cases, "astra_occlusion", "random_occlusion")
    assert pair["paired_cases"] == 5 and pair["baseline_failures"] == 4
    assert pair["on_failed_baselines"] == {
        "both_rescue": 1,
        "astra_only_rescue": 1,
        "random_only_rescue": 1,
        "neither_rescues": 1,
    }


def test_private_paths_never_enter_failed_receipt_report(completed_case, tmp_path):
    search, summary, receipt = completed_case
    missing = tmp_path / "private-operational-receipt" / "unavailable.json"
    report = call_report(search, receipts={summary["episode_id"]: missing})
    assert report["status"] == "audit_failed"
    assert str(tmp_path) not in json.dumps(report)
    receipt["source_sha256"] = {str(tmp_path / "audit.py"): "a" * 64}
    report = call_report(search, receipts={summary["episode_id"]: receipt})
    assert report["status"] == "audit_failed"
    assert str(tmp_path) not in json.dumps(report)


def test_receipt_file_hash_is_preserved(completed_case, tmp_path):
    search, summary, receipt = completed_case
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(receipt) + "\n")
    report = call_report(search, receipts={summary["episode_id"]: path})
    assert report["cases"][0]["audit"]["receipt_sha256"] == file_sha256(path)
    assert report["groups"]["pooled"]["physical_cost"][
        "token_usage"
    ] == summarize_calls(
        [
            call
            for arm in summary["arms"].values()
            for row in arm["attempts"][1:]
            for call in row["provider_records"]
        ]
    )
