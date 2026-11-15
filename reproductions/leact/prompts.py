"""Reconstructed prompts: no original prompt files were released with the paper."""

import json

from .core import qualitative_summary

PROMPT_VARIANTS = ("original", "concise_v1")


def concise_instruction() -> str:
    return (
        "Keep the reasoning inside <think> to at most 200 words in 3-5 sentences. "
        "Connect the visible hand, public card if present, and betting history to "
        "the tradeoffs among legal actions. Do not restate the rules, repeat your "
        "conclusion, or discuss how to format the answer. Finish by closing </think>. "
    )


def forward_prompt(state: dict, rules: str, *, thinking: bool = True,
                   variant: str = "original") -> str:
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {variant}")
    instruction = "Reason about the strategy in your thinking block. " if thinking else ""
    if thinking and variant == "concise_v1":
        instruction = concise_instruction()
    return (
        f"{rules}\n\nYour information state:\n{state['info_state']}\n"
        f"Legal actions: {json.dumps(state['legal_actions'])}\n\n"
        f"{instruction}Choose a mixed strategy using only the visible information. "
        "After the thinking block, output exactly two lines:\n"
        "Action: <one legal action>\n"
        "Policy: <JSON object mapping every legal action to its probability>\n"
        "Use probabilities summing to one."
    )


def backward_prompt(state: dict, rules: str, *, variant: str = "original") -> str:
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {variant}")
    instruction = (
        "Explain why this strategy makes sense from the visible information, including "
        "hand strength, betting history, uncertainty, and incentives where relevant. "
        "Write the strategic reasoning in your thinking block. "
    )
    if variant == "concise_v1":
        instruction = "Explain why the qualitative strategy makes sense. " + concise_instruction()
    return (
        f"{rules}\n\nYour information state:\n{state['info_state']}\n"
        f"Legal actions: {json.dumps(state['legal_actions'])}\n"
        f"An expert's qualitative strategy: {qualitative_summary(state['expert_policy'])}.\n\n"
        f"{instruction}Do not invent or quote "
        "exact expert probabilities. After the thinking block, output only DONE."
    )
