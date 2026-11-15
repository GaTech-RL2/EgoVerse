"""NVIDIA transport with public-output-only provenance and strict policy validation."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

from ..data import fingerprint, write_json

MODEL = "azure/openai/gpt-6-astra"
ENDPOINT = "https://inference-api.nvidia.com/v1/chat/completions"
PROMPT_VERSION = "leduc-astra-actions-then-summary-v2"
SETTINGS = {"max_completion_tokens": 8192, "reasoning_effort": "medium"}
SYSTEM = (
    "You are a strategy adviser for two-player limit Leduc poker. Use only the "
    "provided rules, your private card, public card and public betting history. "
    "You cannot see the opponent's private card or a solver's policy. Return "
    "exactly the requested JSON object, without markdown or extra fields."
)


def strict_json(text):
    def pairs(items):
        result = dict(items)
        if len(result) != len(items):
            raise ValueError("Duplicate JSON keys")
        return result

    def constant(_):
        raise ValueError("Nonfinite JSON number")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def validate_policy(value, actions):
    if not isinstance(value, dict) or set(value) != {"action", "policy"}:
        raise ValueError("Expected only action and policy fields")
    policy = value["policy"]
    if not isinstance(policy, dict) or set(policy) != set(actions):
        raise ValueError("Policy must cover exactly the legal actions")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
           for p in policy.values()):
        raise ValueError("Invalid action probability")
    if not math.isclose(math.fsum(policy.values()), 1, abs_tol=1e-6, rel_tol=0):
        raise ValueError("Action probabilities must sum to one")
    if value["action"] not in actions or policy[value["action"]] != max(policy.values()):
        raise ValueError("Action must be a most probable legal action")
    # Preserve the actual teacher numbers, without filling, smoothing or normalizing.
    return value


def validate_summary(value):
    if not isinstance(value, dict) or set(value) != {"decision_summary"}:
        raise ValueError("Expected only a decision_summary field")
    summary = value["decision_summary"]
    if not isinstance(summary, str) or not 1 <= len(summary.split()) <= 120:
        raise ValueError("Decision summary must contain 1 to 120 words")
    if any(token in summary.lower() for token in ["<think", "</think", "policy:", "action:", "<|im_"]):
        raise ValueError("Decision summary contains output-control tokens")
    return value


def payload(state, rules, *, committed=None):
    # Explicit allowlist: expert_policy and any other oracle/hidden fields never
    # enter an API request. The information-state string is the player's view.
    context = {"rules": rules, "information_state": state["info_state"],
               "player": state["player"], "legal_actions": state["legal_actions"]}
    if committed is None:
        instruction = (
            'Choose a mixed strategy. Return only JSON: {"action": <most probable legal action>, '
            '"policy": {<each legal action>: <probability>}}. Include every legal '
            "action exactly once; probabilities must be finite numbers in [0,1] "
            "summing to one. Break ties in legal_actions order. Return actions and "
            "probabilities only, with no explanation."
        )
    else:
        validate_policy(committed, state["legal_actions"])
        context["previously_committed_decision"] = committed
        instruction = (
            "The supplied decision was already committed in an earlier call. "
            "Give a brief public explanation of that decision, in 1-3 sentences "
            "and at most 100 words. Summarize the relevant visible evidence, "
            "uncertainty and tradeoff. Do not provide an internal monologue or "
            "step-by-step private deliberation. Do not change the decision or "
            "invent hidden cards; acknowledge uncertainty or a possible weakness "
            'if appropriate. Return only JSON: {"decision_summary": <explanation>}.'
        )
    return {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(context, sort_keys=True) + "\n\n" + instruction}],
            "stream": False, "response_format": {"type": "json_object"},
            "cache": {"no-cache": True}, **SETTINGS}


def credential():
    key = os.environ.get("NVIDIA_INFERENCE_API_KEY", "").strip()
    if not key and sys.platform == "darwin":
        result = subprocess.run(
            ["/usr/bin/security", "find-generic-password", "-a", "rpunamiya",
             "-s", "egoverse.leact.nvidia-inference", "-w"],
            capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            key = result.stdout.strip()
    if not key:
        raise ValueError("NVIDIA_INFERENCE_API_KEY or the task Keychain item is required")
    return key


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def decode_json(envelope):
    if not isinstance(envelope, dict) or envelope.get("model") != MODEL:
        raise ValueError("Provider model differs from the pinned Astra model")
    choices = envelope.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected exactly one completion")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
        raise ValueError("Completion was truncated or did not finish normally")
    message = choice.get("message", {})
    if not isinstance(message, dict) or message.get("role") != "assistant" or message.get("refusal"):
        raise ValueError("Completion lacks an accepted assistant message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Completion has no public JSON output")
    return strict_json(content)


def decode(envelope, state, *, summary=False):
    value = decode_json(envelope)
    return validate_summary(value) if summary else validate_policy(value, state["legal_actions"])


def public_response(envelope):
    """Never ingest provider-private reasoning fields into the dataset or logs."""
    if not isinstance(envelope, dict):
        return {}
    result = {k: envelope[k] for k in ["id", "model", "created", "usage", "system_fingerprint"] if k in envelope}
    result["choices"] = [
        {"finish_reason": c.get("finish_reason"),
         "message": {k: c.get("message", {})[k] for k in ["role", "content", "refusal"]
                     if isinstance(c.get("message"), dict) and k in c["message"]}}
        for c in envelope.get("choices", []) if isinstance(c, dict)]
    return result


def call(state, rules, output: Path, *, oracle_sha256, committed=None, timeout=180):
    request = payload(state, rules, committed=committed)
    identity = {"state_id": state["state_id"], "oracle_sha256": oracle_sha256,
                "prompt_version": PROMPT_VERSION, "endpoint": ENDPOINT,
                "stage": "summary" if committed is not None else "actions",
                "request_sha256": fingerprint(request),
                "committed_decision_sha256": fingerprint(committed) if committed is not None else None}
    return call_request(request, output, identity=identity,
                        validate=lambda envelope: decode(envelope, state, summary=committed is not None),
                        timeout=timeout)


def call_request(request, output: Path, *, identity: dict, validate, timeout=180):
    """Execute one versioned public-output request with a task-specific validator."""
    if identity.get("request_sha256") != fingerprint(request):
        raise ValueError("Request fingerprint differs from its identity")
    if identity.get("endpoint") != ENDPOINT or request.get("model") != MODEL:
        raise ValueError("Only the pinned endpoint and model are supported")
    if output.exists():
        previous = json.loads(output.read_text())
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError("Existing Astra response belongs to a different request")
        if not previous.get("accepted"):
            raise ValueError("Previous call failed; retain its evidence and choose a new attempt")
        checked = validate(previous["response"])
        if checked != previous.get("result") or fingerprint(previous["request"]) != identity["request_sha256"]:
            raise ValueError("Existing Astra response failed the provenance audit")
        return previous
    if not 0 < timeout <= 300:
        raise ValueError("Timeout must be in (0,300]")
    key = credential()
    record = {**identity, "request": request, "accepted": False, "started_unix": time.time()}
    http = urllib.request.Request(ENDPOINT, data=json.dumps(request, allow_nan=False).encode(),
                                  headers={"Authorization": "Bearer " + key,
                                           "Content-Type": "application/json", "Accept": "application/json"},
                                  method="POST")
    started = time.monotonic()
    try:
        with urllib.request.build_opener(NoRedirect()).open(http, timeout=timeout) as response:
            record["http_status"] = response.status
            raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("Completion envelope exceeds size limit")
        envelope = strict_json(raw)
        record["response"] = public_response(envelope)
        record["result"] = validate(envelope)
        record["accepted"] = True
    except urllib.error.HTTPError as error:
        record["http_status"] = error.code
        record["error"] = f"NVIDIA inference returned HTTP {error.code}"
        # Retain a bounded, redacted diagnostic; never retain HTTP headers.
        record["provider_error"] = error.read(16384).decode(errors="replace").replace(key, "[REDACTED]")
    except (urllib.error.URLError, OSError, TimeoutError):
        record["error"] = "NVIDIA inference transport failed"
    except (ValueError, TypeError, KeyError, OverflowError):
        record["error"] = "Astra response rejected by the pinned output contract"
    finally:
        record["latency_seconds"] = time.monotonic() - started
        record["completed_unix"] = time.time()
        # The authorization header never enters record. Defensively redact an
        # exact secret should a remote service include it in public output.
        sanitized = json.loads(json.dumps(record, allow_nan=False).replace(key, "[REDACTED]"))
        write_json(output, sanitized)
    if not record["accepted"]:
        raise ValueError(record["error"])
    return sanitized
