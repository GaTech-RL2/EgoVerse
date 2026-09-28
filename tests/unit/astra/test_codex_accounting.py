"""Codex accounting retains unknown costs and never pools HTTP cohorts."""

import copy

import pytest

from astra_reversal.astra_client import ClientError
from astra_reversal.codex_accounting import (
    normalize_receipt_usage,
    summarize_codex_calls,
)
from astra_reversal.intervention_agent import normalize_usage


def row(*, usage=None, accepted=True):
    receipt = {
        "backend": "codex_exec",
        "provider_call": True,
        "accepted": accepted,
        "provider_unavailable": False,
        "returned_model": None,
        "raw_usage": usage,
        "raw_usage_events": [usage],
    }
    counts, _ = normalize_receipt_usage(receipt)
    receipt["token_usage"] = counts
    return {
        "backend": "codex_relay",
        "provider_call": True,
        "accepted": accepted,
        "latency_seconds": 2,
        "codex_receipt": receipt,
        "token_usage": counts,
    }


def test_unknown_usage_and_model_are_not_zero_filled():
    result = summarize_codex_calls([row()])
    assert result["provider_calls"] == result["codex_jobs"] == 1
    assert result["actual_models"] == {"unavailable": 1}
    assert result["tokens"]["input_tokens"] == {
        "sum": 0,
        "available_calls": 0,
        "missing_calls": 1,
        "complete": False,
        "lower_bound_calls": 0,
    }
    assert result["monetary_cost"] is None
    assert "HTTP" in result["counter_semantics"]


def test_rejected_completed_call_retains_true_cost():
    record = row(
        usage={"input_tokens": 100, "output_tokens": 20, "reasoning_output_tokens": 5},
        accepted=False,
    )
    result = summarize_codex_calls([record])
    assert result["failed_calls"] == 1
    assert result["tokens"]["total_tokens"]["sum"] == 120
    assert result["tokens"]["reasoning_tokens"]["sum"] == 5
    assert result["tokens"]["total_tokens"]["complete"]
    assert "response" not in record


def test_ambiguous_multiple_turns_are_lower_bound():
    record = row(accepted=False)
    receipt = record["codex_receipt"]
    receipt.pop("token_usage")
    receipt["raw_usage_events"] = [
        {"input_tokens": 100, "output_tokens": 20},
        {"input_tokens": 50},
    ]
    record["token_usage"], _ = normalize_receipt_usage(receipt)
    receipt["token_usage"] = record["token_usage"]
    result = summarize_codex_calls([record])
    assert result["tokens"]["input_tokens"]["sum"] == 150
    assert result["tokens"]["output_tokens"]["sum"] == 20
    assert not result["tokens"]["input_tokens"]["complete"]
    assert result["tokens"]["output_tokens"]["lower_bound_calls"] == 1


def test_mixed_backends_rejected_by_both_dispatchers():
    from astra_reversal import frs_agent, representation_agent

    for module in (frs_agent, representation_agent):
        codex = {
            **row(),
            "client_schema_version": module.SCHEMA_VERSION,
            "role": "paper_direction",
        }
        http = {**codex, "backend": "nvidia_http"}
        with pytest.raises(ClientError, match="cannot be pooled"):
            module.summarize_calls([codex, http])
        assert module.summarize_calls([codex])["backend"] == "codex_relay"


def test_tampered_counts_rejected():
    record = row(usage={"input_tokens": 10, "output_tokens": 4})
    record["codex_receipt"]["token_usage"] = copy.deepcopy(record["token_usage"])
    record["codex_receipt"]["token_usage"]["input_tokens"] = 999
    with pytest.raises(ClientError, match="disagree"):
        summarize_codex_calls([record])


def test_preflight_separate_from_unknown_start():
    no_job = {
        "backend": "codex_relay",
        "provider_call": False,
        "accepted": False,
        "token_usage": normalize_usage(None),
        "latency_seconds": 3,
        "error": "unavailable",
    }
    unknown = {**no_job, "job_start_unknown": True}
    result = summarize_codex_calls([no_job, unknown])
    assert result["provider_calls"] == 0 and result["client_attempts"] == 2
    assert result["preflight_failures"] == 1
    assert result["unknown_job_start_attempts"] == 1
    assert result["provider_call_count_is_lower_bound"]
    assert not result["tokens"]["total_tokens"]["complete"]
