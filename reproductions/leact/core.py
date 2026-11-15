"""Parsed-policy scoring, positive-delta selection, and SFT targets.

The operational score follows Appendix A.4, not decoder token likelihoods.
No model framework is needed to audit these calculations.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

Policy = dict[str, float]
KL_CAP = 10.0


def distribution(values: Mapping[str, float], actions: Sequence[str]) -> Policy:
    """Validate a normalized distribution with exactly the legal support."""
    if not actions or len(set(actions)) != len(actions):
        raise ValueError("Legal actions must be nonempty and unique")
    if set(values) != set(actions):
        raise ValueError("Policy keys must exactly match legal actions")
    if any(isinstance(v, bool) for v in values.values()):
        raise ValueError("Boolean probabilities are invalid")
    result = {a: float(values[a]) for a in actions}
    if any(not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError("Probabilities must be finite and nonnegative")
    total = math.fsum(result.values())
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"Probabilities must sum to one, got {total}")
    return {a: v / total for a, v in result.items()}


def kl(expert: Policy, predicted: Policy, cap: float | None = KL_CAP) -> float:
    """KL(expert || predicted); zeros on expert support imply infinity."""
    p = distribution(expert, list(expert))
    q = distribution(predicted, list(expert))
    if cap is not None and (not math.isfinite(cap) or cap <= 0):
        raise ValueError("KL cap must be finite and positive")
    terms = []
    for a, mass in p.items():
        if mass == 0:
            continue
        if q[a] == 0:
            return math.inf if cap is None else cap
        terms.append(mass * (math.log(mass) - math.log(q[a])))
    value = max(0.0, math.fsum(terms))
    return value if cap is None else min(value, cap)


def mean_policy(samples: Sequence[Policy]) -> Policy:
    if not samples:
        raise ValueError("At least one baseline sample is required")
    actions = list(samples[0])
    checked = [distribution(p, actions) for p in samples]
    return {a: math.fsum(p[a] for p in checked) / len(checked) for a in actions}


def forward_delta(expert: Policy, candidate: Policy, baselines: Sequence[Policy]) -> float:
    # Average probabilities BEFORE the logarithm/KL, not average per-sample KLs.
    return kl(expert, mean_policy(baselines)) - kl(expert, candidate)


@dataclass(frozen=True)
class ParsedPolicy:
    policy: Policy
    action: str
    fallback: str | None


def parse_policy(text: str, actions: Sequence[str]) -> ParsedPolicy:
    """Parse the final policy; never read a policy quoted inside a think block.

    Malformed policies use the parsed legal action (Appendix A.4). When even
    that is absent, use the first legal action, visibly tagged in the audit.
    Rounded policies are normalized only within 0.02 of unit mass.
    """
    if not actions or len(set(actions)) != len(actions):
        raise ValueError("Legal actions must be nonempty and unique")
    final = text.rsplit("</think>", 1)[-1]
    if "<think>" in final:
        final = ""  # An unclosed trace has no final action.
    action_lines = re.findall(r"^\s*Action:\s*([^\n]+)", final, re.MULTILINE)
    action = action_lines[-1].strip() if action_lines else None
    if action not in actions:
        action = None
    policy_lines = re.findall(r"^\s*Policy:\s*([^\n]+)", final, re.MULTILINE)
    if policy_lines:
        try:
            def unique_object(pairs):
                result = dict(pairs)
                if len(result) != len(pairs):
                    raise ValueError("Duplicate policy keys")
                return result

            raw = json.loads(policy_lines[-1], object_pairs_hook=unique_object)
            if not isinstance(raw, dict) or set(raw) != set(actions):
                raise ValueError("Wrong action support")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw.values()):
                raise ValueError("Non-numeric probability")
            if any(not math.isfinite(v) or v < 0 for v in raw.values()):
                raise ValueError("Invalid probability")
            total = math.fsum(raw.values())
            if total <= 0 or abs(total - 1.0) > 0.02:
                raise ValueError("Invalid probability mass")
            policy = distribution({a: raw[a] / total for a in actions}, actions)
            return ParsedPolicy(policy, action or max(policy, key=policy.get), None)
        except (ValueError, TypeError, OverflowError):
            pass
    reason = "parsed_action" if action is not None else "first_legal_action"
    action = action or actions[0]
    return ParsedPolicy({a: float(a == action) for a in actions}, action, reason)


def qualitative_summary(expert: Policy) -> str:
    """An explicitly reconstructed redaction template; contains no probabilities."""
    expert = distribution(expert, list(expert))
    phrases = []
    for action, p in sorted(expert.items(), key=lambda item: -item[1]):
        frequency = (
            "almost always" if p >= 0.95 else
            "usually" if p >= 0.7 else
            "often" if p >= 0.4 else
            "sometimes" if p >= 0.15 else
            "rarely" if p > 0 else "never"
        )
        phrases.append(f"{frequency} {action}")
    return "; ".join(phrases)


def extract_reasoning(text: str) -> str:
    text = text.strip()
    if text.startswith("<think>"):
        text = text[len("<think>"):]
    if "</think>" in text:
        text = text.split("</think>", 1)[0]
    # Backward generations are reasoning-only; discard unsolicited answer lines.
    text = re.split(r"(?m)^\s*(?:Action|Policy):", text, maxsplit=1)[0]
    return text.strip()


def policy_answer(policy: Policy) -> str:
    policy = distribution(policy, list(policy))
    return f"Action: {max(policy, key=policy.get)}\nPolicy: {json.dumps(policy, sort_keys=True)}"


def select_candidates(candidates: Sequence[dict], k: int, method: str) -> list[dict]:
    if k <= 0 or method not in {"leact", "exit"}:
        raise ValueError("Expected positive K and method leact or exit")
    if len({c["candidate_id"] for c in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate IDs")
    eligible = [c for c in candidates if c["reasoning"].strip() and c.get("complete", True)]
    if any(not math.isfinite(c["score"]) for c in eligible):
        raise ValueError("Non-finite candidate score")
    if method == "leact":
        eligible = [c for c in eligible if c["score"] > 0.0]
    return sorted(eligible, key=lambda c: (-c["score"], c["candidate_id"]))[:k]
