"""Explicit model replication of the frozen LIBERO interface study."""

import copy

DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
DEEPSEEK_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
ASTRA_MODELS = ("gpt-6-astra", "openai/openai/gpt-6-astra", "azure/openai/gpt-6-astra")


def validate_model(manifest):
    model = manifest["model"]
    if manifest["observer_model"] != model["identifier"]:
        raise ValueError("actor_observer_model_mismatch")
    replication = manifest.get("model_replication")
    if replication is None:
        if model["identifier"] not in ASTRA_MODELS:
            raise ValueError("Astra_model_required")
        if model.get("responses_dialect"):
            raise ValueError("unexpected_responses_dialect")
        return
    required = {
        "identifier": "deepseek-flash",
        "base_url": "https://api.deepseek.com/v1",
        "serving_provider": "deepseek",
        "responses_dialect": "deepseek-v4.1",
        "reasoning_effort": "high",
        "snapshot_pinned": False,
        "context_truncation": "disabled",
        "input_reservation": {"mode": "usage_only"},
        "repeat_task_instruction": True,
        "sampling": {},
    }
    if (
        replication.get("requested_model") != DEEPSEEK_MODEL
        or replication.get("requested_hf_revision") != DEEPSEEK_REVISION
        or replication.get("pool_with_astra") is not False
        or any(model.get(key) != value for key, value in required.items())
    ):
        raise ValueError("unrecognized_model_replication")


def deepseek_design(manifest):
    """Copy an original or adaptive design; never mutate or pool prior cohorts."""
    result = copy.deepcopy(manifest)
    result["study_date_utc"] = "2026-10-18T00:00:00+00:00"
    result["scope"] = (
        "simulation only; direct DeepSeek robot control; no policy or expert demonstrations"
    )
    result["model"].update(
        {
            "identifier": "deepseek-flash",
            "base_url": "https://api.deepseek.com/v1",
            "serving_provider": "deepseek",
            "responses_dialect": "deepseek-v4.1",
            "reasoning_effort": "high",
            "snapshot_pinned": False,
            "context_truncation": "disabled",
            "repeat_task_instruction": True,
            "input_reservation": {"mode": "usage_only"},
            "sampling": {},
        }
    )
    result["observer_model"] = result["model"]["identifier"]
    result["model_replication"] = {
        "requested_model": DEEPSEEK_MODEL,
        "requested_hf_revision": DEEPSEEK_REVISION,
        "reference_source_commit": "9c2278d2b9e52434ff06950b6b2a5b94071a971c",
        "pool_with_astra": False,
        "checkpoint_identity": "Official API documentation maps deepseek-flash to the requested checkpoint as of 2026-10-18. This is an unpinned serving alias, not a verified weights hash; recheck before launch.",
        "identity_source": "https://api-docs.deepseek.com/quick_start/pricing/",
        "transport_source": "https://api-docs.deepseek.com/guides/responses_api/",
        "thinking_source": "https://api-docs.deepseek.com/guides/thinking_mode/",
        "transport_amendments": [
            "Keep plain-text reasoning output verbatim in subsequent input; omit unsupported encrypted reasoning, store and truncation fields.",
            "Native API always enables parallel tool calls. Add exactly-one-tool-call instruction in all actor arms, preserve strict one-call execution validation; smoke must pass before trials.",
            "Keep full context; native API does not support automatic truncation. Context/vision limits are infrastructure failures, never silent history dropping.",
            "Use explicit high reasoning: DeepSeek maps medium/high/xhigh to high. Reasoning compute is not matched to Astra medium.",
            "Keep current task in instructions, matching Astra's automatic-truncation prompt. Observer remains stateless and uses the same DeepSeek model.",
        ],
        "historical_comparison": "Astra is a historical reference only. Different providers, dates, reasoning levels and load preclude a controlled cross-model wall-clock speed claim.",
    }
    result["speed_analysis"] = {
        "version": 1,
        "primary": "Verified success by elapsed actor-to-first-success wall time, with all assigned trials as denominator; missing trials shown separately.",
        "secondary": [
            "Verified success by native control steps",
            "Matched-pair time difference and ratio on jointly successful starts, conditional and descriptive",
            "Provider round trips, client admission/backoff, simulator/evaluator and residual local time",
            "Actor/observer tokens, calls and total-output-token throughput per provider round-trip second",
        ],
        "report_times_seconds": [60, 300, 600, 1200, 1800, 3600, 7200, 14400, 28800],
        "report_times_are_limits": False,
        "server_queue_and_decode_separable": False,
    }
    result["amendment"] = {
        "date": "2026-10-18",
        "authority": "User requested the same study with DeepSeek-V4.1-Flash-1017 and a focus on speed",
        "pool_with_prior_cohort": False,
        "usage_policy": "Record tokens, calls and elapsed time without experiment resource budgets.",
        "recovery": "Keep three transient transport retries, no action redispatch or trial restart; unresolved infrastructure failures stop collection.",
        "archive_policy": "Archive each completed trial and independent replay; retain the original and adaptive cohorts separately.",
    }
    if "extension" in result:
        result["extension"]["authority"] = (
            "User requested the same F-adaptive study with DeepSeek"
        )
    result["power"]["status"] = "pending_deepseek_replication_pilot"
    result["power"]["confirmation_frozen"] = False
    result["resolved"] = result["readiness"] = None
    validate_model(result)
    return result
