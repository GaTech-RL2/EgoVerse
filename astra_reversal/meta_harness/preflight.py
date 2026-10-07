"""Actual frozen-checkpoint hook gates, run only on an allocated CUDA worker."""

import numpy as np

from astra_reversal.demo_skill_conditioning import InputSkillConditioner
from astra_reversal.records import digest, to_numpy
from astra_reversal.vision_interpolation import weighted_vision_probe


def policy_gate(policy, bank):
    sources = list(bank.sources)
    if len(sources) < 2:
        raise ValueError("The hook preflight needs two audited standard sources")
    first, second = sources[:2]
    live = bank.observation(first, 0)
    target, source_a = bank.sources[first]["prompt"], bank.sources[second]["prompt"]
    conditioner = InputSkillConditioner(policy, bank, {})
    report = weighted_vision_probe(
        policy, live, target, conditioner._donor_pair(second, 0)
    )
    noise = policy.noise(np.random.default_rng(137))

    def solve(condition):
        result = policy.sample(
            condition, noise, solver="euler", steps=10, time_power=1.0
        )
        report["velocity_evaluations"] += result.velocity_evaluations
        return to_numpy(result.value)

    native = solve(policy.prepare(live, digest(live), target))
    midpoint, _ = policy.prepare_interpolated(
        live,
        digest(live),
        target,
        source_prompts=(target, source_a),
        alpha=0.5,
        operator="tli",
    )
    report["checks"]["tli_half_exact_native"] = bool(
        np.array_equal(native, solve(midpoint))
    )
    # TEI preserves the TARGET's protected slots and masks. Native-A parity
    # therefore needs a target with A's layout, not merely matching A/B sources.
    native_a = solve(policy.prepare(live, digest(live), source_a))
    endpoint, endpoint_provenance = policy.prepare_interpolated(
        live,
        digest(live),
        source_a,
        source_prompts=(source_a, target),
        alpha=0,
        operator="tei",
    )
    endpoint_actions = solve(endpoint)
    report["checks"]["tei_zero_selects_source_a_matching_masks"] = bool(
        np.array_equal(native_a, endpoint_actions)
    )
    report["tei_zero_endpoint"] = {
        "target_prompt": source_a,
        "source_prompts": [source_a, target],
        "native_a_max_abs": float(np.abs(native_a - endpoint_actions).max()),
        "provenance": endpoint_provenance,
    }
    # Also verify that alpha=0 is an active source-A edit when the original
    # target differs. Different target layouts need not equal native-A actions.
    edited, edited_provenance = policy.prepare_interpolated(
        live,
        digest(live),
        target,
        source_prompts=(source_a, source_a),
        alpha=0,
        operator="tei",
    )
    edited_actions = solve(edited)
    report["tei_zero_error_from_original_target"] = float(
        np.abs(native - edited_actions).max()
    )
    report["checks"]["tei_zero_original_target_modified"] = bool(
        edited_provenance["has_effect"] and not np.array_equal(native, edited_actions)
    )
    report["checks"]["tei_zero_target_layout_preserved"] = all(
        edited_provenance[key]
        for key in (
            "text_mask_fixed",
            "protected_embedding_slots_unchanged",
            "vision_prefix_unchanged",
        )
    )
    report["tei_zero_original_target_provenance"] = edited_provenance
    report["status"] = "passed" if all(report["checks"].values()) else "failed"
    report["scope"] = (
        "audited demonstration observations; frozen weighted interface gates, not rollout success"
    )
    return report
