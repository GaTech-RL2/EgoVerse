"""Policy-generated subgoal candidates, with no environment or action inversion."""

import numpy as np

from astra_reversal.action_adapter import ActionAdapter, ActionSpec
from astra_reversal.records import digest, to_numpy


def candidates(
    policy,
    adapter,
    observation,
    observation_id,
    instruction,
    subgoal,
    noise,
    *,
    include_tli=False,
):
    if not isinstance(subgoal, str) or not subgoal.strip() or len(subgoal) > 160:
        raise ValueError("A concrete bounded subgoal instruction is required")
    if subgoal.strip() == instruction.strip():
        raise ValueError("The subgoal duplicates the fixed native instruction")
    result = []
    original_noise = digest(to_numpy(noise))
    banks = None
    if include_tli:
        banks = {
            "a": policy.capture_text_latents(
                observation, subgoal, observation_id=observation_id
            ),
            "b": policy.capture_text_latents(
                observation, instruction, observation_id=observation_id
            ),
        }
    # Prepare and sample serially: conditioning hooks share the same frozen model.
    for index in range(5 if include_tli else 3):
        if index >= 3:
            alpha = (0.25, 0.0)[index - 3]
            condition, receipt = policy.prepare_interpolated(
                observation,
                observation_id,
                instruction,
                source_prompts=(subgoal, instruction),
                alpha=alpha,
                operator="tli",
                text_latents=banks,
            )
            name = f"subgoal_tli_{index - 2}"
        elif index:
            alpha = (0.33, 0.67)[index - 1]
            condition, receipt = policy.prepare_interpolated(
                observation,
                observation_id,
                instruction,
                source_prompts=(instruction, subgoal),
                alpha=alpha,
                operator="tei",
            )
            name = f"subgoal_tei_{index}"
        else:
            name = "subgoal_prompt"
            condition = policy.prepare(observation, observation_id, subgoal)
            receipt = {"operator": "prompt", "source_prompt": subgoal}
        sample = policy.sample(condition, noise, steps=10).value
        if digest(to_numpy(noise)) != original_noise:
            raise RuntimeError("Semantic candidate changed the shared native noise")
        commands, clipping = adapter.decode(sample, condition.state)
        result.append(
            (
                name,
                commands,
                {
                    "candidate_id": name,
                    "condition_id": condition.condition_id,
                    "observation_id": observation_id,
                    "noise_sha256": original_noise,
                    "conditioning": receipt,
                    "clipping": clipping,
                    "source": "current_policy_same_real_observation_and_noise",
                    "unexecuted": True,
                },
            )
        )
    return result


def preflight(policy, observation, instruction, *, include_tli=False):
    """Check the weighted candidate path and restore native conditioning exactly."""
    raw = {**observation, "prompt": instruction}
    oid = digest(raw)
    noise = policy.noise(np.random.default_rng(173))
    condition = policy.prepare(observation, oid, instruction)
    before = policy.sample(condition, noise, steps=10).value
    spec = ActionSpec(
        "semantic_preflight",
        policy.horizon,
        policy.action_dim,
        0.05,
        (-1,) * 7,
        (1,) * 7,
        {},
    )
    adapter = ActionAdapter(spec, policy.input_transform, policy.output_transform)
    reference, _ = adapter.decode(before, condition.state)
    rows = candidates(
        policy,
        adapter,
        observation,
        oid,
        instruction,
        "lift the object while keeping the gripper closed",
        noise,
        include_tli=include_tli,
    )
    zero_tli_error = None
    if include_tli:
        zero_condition, _ = policy.prepare_interpolated(
            observation,
            oid,
            instruction,
            source_prompts=(
                "lift the object while keeping the gripper closed",
                instruction,
            ),
            alpha=0.5,
            operator="tli",
        )
        zero_tli = policy.sample(zero_condition, noise, steps=10).value
        zero_tli_error = float(np.max(np.abs(to_numpy(before) - to_numpy(zero_tli))))
        if zero_tli_error != 0:
            raise RuntimeError("Zero-factor TLI differs from native sampling")
        if any(
            np.max(np.abs(commands - reference)) <= 1e-6
            for name, commands, _ in rows
            if name.startswith("subgoal_tli")
        ):
            raise RuntimeError(
                "Contrastive TLI produced no measurable controller effect"
            )
    after_condition = policy.prepare(observation, oid, instruction)
    after = policy.sample(after_condition, noise, steps=10).value
    error = float(np.max(np.abs(to_numpy(before) - to_numpy(after))))
    if error != 0:
        raise RuntimeError("Semantic candidate preparation changed the native policy")
    return {
        "native_restoration_max_abs": error,
        "zero_tli_max_abs": zero_tli_error,
        "environment_actions": 0,
        "policy_updates": 0,
        "synthetic_diagnostic_not_training_data": True,
        "candidates": [
            {
                "candidate_id": name,
                "max_controller_difference": float(
                    np.max(np.abs(commands - reference))
                ),
                "generation": receipt,
            }
            for name, commands, receipt in rows
        ],
    }
