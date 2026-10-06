"""Fixed, fail-closed policy tools. Candidates cannot change these semantics."""

import copy
import math
from dataclasses import dataclass

from astra_reversal.demo_skill_program import validate_program
from astra_reversal.records import digest

VISION_GRID = (0, 0.1, 0.25, 0.5)
LANGUAGE_GRID = (0, 0.25, 0.5, 0.75, 1)
TOOLS = ("set_policy_program", "keep_policy_program", "clear_policy_program")


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def exact(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("Unexpected or missing fields")


def strength(value, grid):
    if type(value) not in (int, float) or not math.isfinite(value) or value not in grid:
        raise ValueError("Strength is outside the frozen grid")


@dataclass(frozen=True)
class Limits:
    action_budget: int = 300
    execute_steps: int = 5
    prediction_horizon: int = 50
    solver_steps: int = 10
    calls: int = 8
    max_age_actions: int = 20
    max_age_seconds: float = 10.0
    request_timeout: float = 240.0

    def __post_init__(self):
        if (
            self.action_budget,
            self.execute_steps,
            self.prediction_horizon,
            self.solver_steps,
        ) != (300, 5, 50, 10):
            raise ValueError(
                "The MVP freezes 300 actions, five-action execution, H50 and Euler10"
            )
        if self.calls not in (4, 8, 16):
            raise ValueError("Choose a declared 4/8/16-call comparison")
        if not integer(self.max_age_actions, 0, 100) or self.max_age_actions % 5:
            raise ValueError("Freshness must align with replan boundaries")
        for value in (self.max_age_seconds, self.request_timeout):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("Timeouts must be finite and positive")


def make_cards(bank):
    """Standard demonstration segments, without importing adapted OOD programs.

    Thirds are initial retrieval candidates, not claims about semantic phases or
    transfer success. Card identities include the complete audited source bank.
    """
    cards = []
    for source in bank.catalog():
        end = source["frame_count"]
        boundaries = sorted({0, end // 3, 2 * end // 3, end})
        for start, stop in zip(boundaries, boundaries[1:]):
            if start == stop:
                continue
            card = {
                "source_id": source["source_id"],
                "description": source["prompt"],
                "start_frame": start,
                "end_frame": stop,
                "playback": "advance",
                "min_actions": 5,
                "advance_when": "segment_end",
                "threshold": 0.0,
                "bank_id": bank.bank_id,
                "evidence": "standard demonstration only; transfer unverified",
            }
            card["skill_id"] = digest(card)
            cards.append(card)
    return cards


class Compiler:
    def __init__(self, bank, cards):
        self.bank = bank
        self.cards = {}
        self.catalog = bank.catalog()
        sources = {row["source_id"]: row for row in self.catalog}
        for card in cards:
            exact(
                card,
                (
                    "skill_id",
                    "source_id",
                    "description",
                    "start_frame",
                    "end_frame",
                    "playback",
                    "min_actions",
                    "advance_when",
                    "threshold",
                    "bank_id",
                    "evidence",
                ),
            )
            content = {k: v for k, v in card.items() if k != "skill_id"}
            if card["skill_id"] != digest(content) or card["skill_id"] in self.cards:
                raise ValueError("Duplicate or modified card identity")
            if card["bank_id"] != bank.bank_id or card["source_id"] not in sources:
                raise ValueError("Card references an ineligible source bank")
            if card["description"] != sources[card["source_id"]]["prompt"]:
                raise ValueError("Card changed its source instruction")
            self.cards[card["skill_id"]] = copy.deepcopy(card)
            # Exercise the upstream validator at admission, not just execution.
            self.compile(
                {
                    "tool": "set_policy_program",
                    "arguments": {
                        "observation_id": "admission",
                        "expected_stage_id": "native",
                        "skill_id": card["skill_id"],
                        "language": None,
                        "vision": None,
                        "max_actions": 100,
                    },
                },
                [card["skill_id"]],
            )
        self.identity = digest({"bank": bank.bank_id, "cards": cards})

    def compile(self, call, retrieved):
        exact(call, ("tool", "arguments"))
        tool, args = call["tool"], call["arguments"]
        if tool not in TOOLS:
            raise ValueError("Unsupported policy tool")
        required = {
            "set_policy_program": (
                "observation_id",
                "expected_stage_id",
                "skill_id",
                "language",
                "vision",
                "max_actions",
            ),
            "keep_policy_program": ("observation_id", "program_id"),
            "clear_policy_program": ("observation_id",),
        }
        exact(args, required[tool])
        if not isinstance(args["observation_id"], str) or not args["observation_id"]:
            raise ValueError("Missing observation identity")
        if tool != "set_policy_program":
            if tool == "keep_policy_program" and not isinstance(
                args["program_id"], str
            ):
                raise ValueError("Invalid program identity")
            return None
        if args["skill_id"] not in retrieved or args["skill_id"] not in self.cards:
            raise ValueError("Select a card actually supplied in this request")
        card = self.cards[args["skill_id"]]
        if (
            not integer(args["max_actions"], max(5, card["min_actions"]), 100)
            or args["max_actions"] % 5
        ):
            raise ValueError(
                "Program duration must be 5..100 actions in multiples of five"
            )
        language, vision = args["language"], args["vision"]
        if language is not None:
            exact(language, ("operator", "source_a_id", "source_b_id", "alpha"))
            if language["operator"] != "tei":
                raise ValueError("MVP enables only native, TEI and VEI")
            strength(language["alpha"], LANGUAGE_GRID)
            allowed = {self.cards[key]["source_id"] for key in retrieved}
            if any(
                language[key] not in allowed for key in ("source_a_id", "source_b_id")
            ):
                raise ValueError("Language references an unretrieved source")
        if vision is not None:
            exact(vision, ("operator", "alpha"))
            if vision["operator"] != "vei":
                raise ValueError("MVP enables only native, TEI and VEI")
            strength(vision["alpha"], VISION_GRID)
        if language is not None and vision is not None:
            raise ValueError(
                "Simultaneous TEI and VEI are unsupported, including zero-strength VEI"
            )
        stage = {
            **{
                key: card[key]
                for key in (
                    "source_id",
                    "start_frame",
                    "end_frame",
                    "playback",
                    "min_actions",
                    "advance_when",
                    "threshold",
                )
            },
            "max_actions": args["max_actions"],
            "language": copy.deepcopy(language),
            "alpha": 0.0 if vision is None else vision["alpha"],
            "vision_operator": "none" if vision is None else "vei",
            "state_mode": "live",
            "occlusion_box": None,
        }
        return validate_program(
            {"native": False, "stages": [stage]}, self.catalog, "input_skill_library"
        )


def tool_schemas(cards):
    def obj(properties):
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    sources = sorted({c["source_id"] for c in cards})
    language = obj(
        {
            "operator": {"type": "string", "enum": ["tei"]},
            "source_a_id": {"type": "string", "enum": sources},
            "source_b_id": {"type": "string", "enum": sources},
            "alpha": {"type": "number", "enum": list(LANGUAGE_GRID)},
        }
    )
    vision = obj(
        {
            "operator": {"type": "string", "enum": ["vei"]},
            "alpha": {"type": "number", "enum": list(VISION_GRID)},
        }
    )
    obs = {"observation_id": {"type": "string"}}
    specs = {
        "set_policy_program": {
            **obs,
            "expected_stage_id": {"type": "string"},
            "skill_id": {"type": "string", "enum": [c["skill_id"] for c in cards]},
            "language": {"anyOf": [language, {"type": "null"}]},
            "vision": {"anyOf": [vision, {"type": "null"}]},
            "max_actions": {"type": "integer", "enum": list(range(5, 101, 5))},
        },
        "keep_policy_program": {**obs, "program_id": {"type": "string"}},
        "clear_policy_program": obs,
    }
    return [
        {
            "type": "function",
            "name": name,
            "description": name.replace("_", " "),
            "parameters": obj(args),
            "strict": True,
        }
        for name, args in specs.items()
    ]
