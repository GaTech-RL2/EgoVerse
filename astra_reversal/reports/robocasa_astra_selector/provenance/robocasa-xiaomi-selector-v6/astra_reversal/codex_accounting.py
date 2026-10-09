"""Accounting for local Codex jobs, without synthesizing HTTP responses.

Counts include the Codex harness. One job may contain unobservable internal
requests/retries. Derived input+output totals are not provider-reported totals.
"""

import math
from collections import Counter

from .astra_client import ClientError
from .intervention_agent import normalize_usage


def normalize_receipt_usage(receipt):
    """Validate actual CLI usage and return (normalized counts, lower bound).

    All completed-event receipts survive ambiguous/failed jobs. Available event
    counts are summed; absent counts stay unknown. Partial/multi-turn receipts
    remain lower bounds rather than asserting that hidden work cost zero.
    """
    if not isinstance(receipt, dict):
        raise ClientError("Codex receipt must be an object")
    events = receipt.get("raw_usage_events")
    if events is None:
        events = [receipt.get("raw_usage")]
    elif not isinstance(events, list):
        raise ClientError("Codex raw_usage_events must be an array")
    if len(events) == 1 and receipt.get("raw_usage") != events[0]:
        raise ClientError("Codex raw usage differs from completed-event receipt")
    normalized = []
    for event in events:
        if event is not None and not isinstance(event, dict):
            raise ClientError("Codex event usage must be an object or null")
        mapped = dict(event or {})
        if "reasoning_output_tokens" in mapped:
            mapped["output_tokens_details"] = {
                "reasoning_tokens": mapped["reasoning_output_tokens"]
            }
        row = normalize_usage(mapped)
        if row["total_tokens"] is None and all(
            row[key] is not None for key in ("input_tokens", "output_tokens")
        ):
            row["total_tokens"] = row["input_tokens"] + row["output_tokens"]
        normalized.append(row)
    counts = {
        key: (
            sum(row[key] for row in normalized if row[key] is not None)
            if any(row[key] is not None for row in normalized)
            else None
        )
        for key in normalize_usage(None)
    }
    if "token_usage" in receipt and receipt["token_usage"] != counts:
        raise ClientError("Codex token counts disagree with actual usage events")
    lower_bound = len(events) > 1 or receipt.get("token_usage_is_lower_bound") is True
    return counts, lower_bound


def summarize_codex_calls(records):
    """Summarize one Codex cohort; reject mixed backend records."""
    records = list(records)
    jobs, preflight, uncertain = [], [], []
    for record in records:
        if not isinstance(record, dict) or record.get("backend") != "codex_relay":
            raise ClientError("Codex and HTTP provider records cannot be pooled")
        if any(
            type(record.get(key)) is not bool for key in ("provider_call", "accepted")
        ):
            raise ClientError("Codex ledger requires boolean job/acceptance fields")
        latency = record.get("latency_seconds")
        if (
            type(latency) not in (int, float)
            or not math.isfinite(latency)
            or latency < 0
        ):
            raise ClientError("Codex ledger latency must be finite and nonnegative")
        receipt = record.get("codex_receipt")
        if record["provider_call"]:
            if (
                not isinstance(receipt, dict)
                or receipt.get("backend") != "codex_exec"
                or receipt.get("provider_call") is not True
            ):
                raise ClientError("Started Codex job requires its execution receipt")
            counts, lower_bound = normalize_receipt_usage(receipt)
            if record.get("token_usage") != counts:
                raise ClientError(
                    "Codex ledger counts differ from its execution receipt"
                )
            if record["accepted"] and (
                receipt.get("accepted") is not True
                or receipt.get("provider_unavailable") is True
            ):
                raise ClientError("Accepted Codex proposal has no successful receipt")
            jobs.append((record, receipt, counts, lower_bound))
        else:
            if record["accepted"] or record.get("token_usage") != normalize_usage(None):
                raise ClientError(
                    "Unstarted Codex invocation cannot claim proposal or usage"
                )
            if isinstance(receipt, dict):
                counts, _ = normalize_receipt_usage(receipt)
                if (
                    any(value is not None for value in counts.values())
                    or receipt.get("provider_call") is True
                ):
                    raise ClientError("Unstarted Codex invocation claims a job receipt")
            (
                uncertain if record.get("job_start_unknown") is True else preflight
            ).append(record)
    tokens = {}
    for field in normalize_usage(None):
        known = [counts[field] for _, _, counts, _ in jobs if counts[field] is not None]
        incomplete = sum(
            counts[field] is not None and lower for _, _, counts, lower in jobs
        )
        tokens[field] = {
            "sum": sum(known),
            "available_calls": len(known),
            "missing_calls": len(jobs) - len(known),
            "complete": len(known) == len(jobs) and incomplete == 0 and not uncertain,
            "lower_bound_calls": incomplete,
        }
    accepted = sum(row["accepted"] for row, _, _, _ in jobs)
    usage_available = sum(
        any(value is not None for value in counts.values()) for _, _, counts, _ in jobs
    )
    return {
        "backend": "codex_relay",
        "provider_calls": len(jobs),
        "codex_jobs": len(jobs),
        "counter_semantics": "provider_calls counts started Codex CLI jobs, not raw provider HTTP requests",
        "internal_provider_requests": None,
        "internal_provider_retries": None,
        "provider_call_count_is_lower_bound": bool(uncertain),
        "unknown_job_start_attempts": len(uncertain),
        "unknown_job_start_latency_seconds": sum(
            row["latency_seconds"] for row in uncertain
        ),
        "accepted_proposals": accepted,
        "failed_calls": len(jobs) - accepted,
        "tokens": tokens,
        "usage_available_calls": usage_available,
        "usage_unavailable_calls": len(jobs) - usage_available,
        "reasoning_tokens_are_subset_of_output_tokens": True,
        "total_tokens_semantics": "Reported total when available; otherwise derived input_tokens + output_tokens",
        "derived_total_jobs": sum(
            receipt.get("total_tokens_derived") is True for _, receipt, _, _ in jobs
        ),
        "latency_seconds": sum(row["latency_seconds"] for row, _, _, _ in jobs),
        "actual_models": dict(
            Counter(
                receipt["returned_model"]
                if isinstance(receipt.get("returned_model"), str)
                and receipt["returned_model"]
                else "unavailable"
                for _, receipt, _, _ in jobs
            )
        ),
        "errors": dict(
            Counter(
                row.get("error", "unclassified_failure")
                for row, _, _, _ in jobs
                if not row["accepted"]
            )
        ),
        "client_attempts": len(records),
        "preflight_failures": len(preflight),
        "preflight_errors": dict(
            Counter(row.get("error", "unclassified_failure") for row in preflight)
        ),
        "preflight_latency_seconds": sum(row["latency_seconds"] for row in preflight),
        "monetary_cost": None,
        "monetary_cost_unavailable_reason": "ChatGPT plan usage is not a verified per-token API price.",
    }
