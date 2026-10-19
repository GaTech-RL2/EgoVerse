"""Explicit model replication of the frozen LIBERO interface study."""

import copy

DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
DEEPSEEK_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
VLLM_IMAGE = "docker.io/vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3"
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
    if replication.get("self_hosted") is True:
        required.update(
            identifier=DEEPSEEK_MODEL,
            base_url="http://127.0.0.1:18080/v1",
            serving_provider="osmo-vllm",
            responses_dialect="deepseek-v4.1-vllm",
            snapshot_pinned=True,
            numeric_reasoning_effort=75,
            sampling={"temperature": 1.0, "top_p": 1.0, "seed": 42},
        )
        if replication.get("serving_image") != VLLM_IMAGE:
            raise ValueError("unrecognized_serving_image")
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


def self_hosted_design(manifest):
    """Pin the exact weights and local serving protocol in a separate cohort."""
    result = deepseek_design(manifest)
    result["model"].update(
        identifier=DEEPSEEK_MODEL,
        base_url="http://127.0.0.1:18080/v1",
        serving_provider="osmo-vllm",
        responses_dialect="deepseek-v4.1-vllm",
        snapshot_pinned=True,
        numeric_reasoning_effort=75,
        sampling={"temperature": 1.0, "top_p": 1.0, "seed": 42},
    )
    result["observer_model"] = DEEPSEEK_MODEL
    result["model_replication"].update(
        self_hosted=True,
        serving_image=VLLM_IMAGE,
        checkpoint_identity="Hugging Face download pinned to requested_hf_revision; server receipt verifies revision, index, shard sizes and image identity before actors start.",
        identity_source="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170",
        transport_source="https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml",
    )
    result["model_replication"]["transport_amendments"].extend(
        [
            "Self-hosted vLLM 0.31.0 Responses API through a credential-free loopback relay inside one private OSMO group. Hosted API credentials are not read or forwarded.",
            "Numeric reasoning effort 75 matches the official API high mapping documented in the vLLM recipe; preserve prior reasoning explicitly (drop_thinking=false). Sampling is temperature=1, top_p=1, seed=42 in all arms.",
            "One TP8 H100 server with Engram CPU offload, DSpark speculative decoding, 1M context and 4096 batched tokens. Serving configuration is identical in all arms.",
            "The pool requires whole eight-GPU nodes. Eight evaluator workers receive one H100 each on a separate node; the TP8 server receives its own eight H100s. Model serving and simulator rendering are separate tasks.",
            "No external-provider rate limit or artificial client request pacing: eight actor workers use vLLM continuous batching (max_num_seqs=8). Provider round trips include actual model queue/compute; retry backoff remains separately timed.",
        ]
    )
    result["amendment"]["date"] = "2026-10-19"
    result["amendment"]["authority"] = (
        "User requested self-inference on L40/H100 and then instructed continue"
    )
    result["speed_analysis"]["server_metrics"] = (
        "Archive vLLM Prometheus snapshots. Per-request round trips remain the measured actor latency; aggregate server metrics cannot be subtracted from individual concurrent requests."
    )
    validate_model(result)
    return result
