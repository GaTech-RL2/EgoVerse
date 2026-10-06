"""Development-only Astra with explicit access to search evidence, never tests."""

import base64
import json
import time

from astra_reversal.astra_client import _strict_json
from astra_reversal.demo_segments import write_json

from .astra_worker import AstraWorker
from .search import SearchView

SYSTEM = """You are the development-only Meta-Harness proposer. Improve a frozen
runtime Astra's use of frozen pi0.5 through ONE coherent, falsifiable harness
change. Inspect earlier candidate source, SEARCH metrics and execution traces
with the supplied read tools. You cannot change models, tools, sensor access,
budgets, success definitions or final evaluation. No final-test outcomes are
available. Return a candidate using submit_candidate. Harness source must be one
function harness(observation, history, cards, memory) in the bounded Python
subset: assignments, if, bounded for loops, JSON values, indexing, slicing,
comparisons, +, -, integer %, and helpers get/len/min/max/rank/append only.
Return keys request(bool), card_ids(1..6), history_ids(0..1 previous observation),
instruction(string), memory(dict). No imports, attributes, ambient I/O, recursion
or evaluator access. rank(cards, query) ranks lexical overlap. Memory is a
hypothesis, not verified success. In prompt_only, edit only the returned literal
instruction; all executable code must remain identical to the initial harness.
"""


def _tool(name, properties):
    return {
        "type": "function",
        "name": name,
        "description": name,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


class Proposer:
    def __init__(self, *, model="gpt-6-astra", post=None, max_turns=8):
        if not (model == "gpt-6-astra" or model.startswith("gpt-6-astra-")):
            raise ValueError("Proposer must be Astra")
        self.model, self.post, self.max_turns = (
            model,
            post or AstraWorker._post,
            max_turns,
        )
        self.records = []

    def propose(self, archive, *, kind, baseline):
        if kind not in ("prompt_only", "meta_harness"):
            raise ValueError("Unknown search arm")
        if (archive.root / "selected_bundle").exists():
            raise ValueError("Proposer cannot run after freezing")
        view = SearchView(archive)
        input_items = [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "kind": kind,
                        "initial_harness": baseline,
                        "available_search_files": view.files(),
                    }
                ),
            }
        ]
        tools = [
            _tool(
                "read_search_file",
                {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "count": {"type": "integer"},
                },
            ),
            _tool("view_search_image", {"path": {"type": "string"}}),
            _tool(
                "submit_candidate",
                {"hypothesis": {"type": "string"}, "source": {"type": "string"}},
            ),
        ]
        records = []
        for turn in range(self.max_turns):
            began = time.monotonic()
            body = {
                "model": self.model,
                "instructions": SYSTEM,
                "input": input_items,
                "tools": tools,
                "tool_choice": "required",
                "parallel_tool_calls": False,
                "reasoning": {"effort": "medium"},
                "max_output_tokens": 4096,
                "store": False,
            }
            count = self.post(
                "responses/input_tokens",
                {key: body[key] for key in ("model", "instructions", "input", "tools")},
                240,
            )
            if (
                type(count.get("input_tokens")) is not int
                or not 0 < count["input_tokens"] <= 32768
            ):
                raise ValueError("Proposer context budget exceeded")
            response = self.post("responses", body, 240)
            records.append(
                {
                    "role": "proposer",
                    "kind": kind,
                    "turn": turn,
                    "response": response,
                    "latency_seconds": time.monotonic() - began,
                    "counted_input_tokens": count["input_tokens"],
                }
            )
            self.records.append(records[-1])
            log = archive.root / "proposer_records.jsonl"
            with log.open("a") as stream:
                stream.write(json.dumps(records[-1], allow_nan=False) + "\n")
            if (
                response.get("model") != self.model
                or response.get("status") != "completed"
            ):
                raise ValueError("Proposer identity or completion failure")
            outputs = response.get("output", [])
            calls = [item for item in outputs if item.get("type") == "function_call"]
            if len(calls) != 1:
                raise ValueError("Proposer must issue exactly one allowed tool")
            call = calls[0]
            args = _strict_json(call["arguments"].encode())
            if call["name"] == "submit_candidate":
                if set(args) != {"source", "hypothesis"}:
                    raise ValueError("Malformed candidate submission")
                return archive.register(
                    args["source"], args["hypothesis"], kind=kind, baseline=baseline
                )
            input_items.extend(outputs)
            images = []
            try:
                if call["name"] == "read_search_file":
                    result = view.read(
                        args["path"], offset=args["offset"], count=args["count"]
                    )
                elif call["name"] == "view_search_image":
                    path = view.root / args["path"]
                    if (
                        args["path"] not in view.files()
                        or path.suffix != ".png"
                        or not path.resolve().is_relative_to(view.root)
                        or path.stat().st_size > 2 * 1024 * 1024
                    ):
                        raise ValueError("Unknown search image")
                    images = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": "data:image/png;base64,"
                                    + base64.b64encode(path.read_bytes()).decode(),
                                }
                            ],
                        }
                    ]
                    result = {"attached_search_image": args["path"]}
                else:
                    raise ValueError("Unknown proposer tool")
            except (ValueError, KeyError, OSError) as error:
                result = {"error": type(error).__name__}
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": json.dumps(result),
                }
            )
            input_items.extend(images)
        raise ValueError("Proposer exhausted its fixed tool-turn budget")


def run_search(archive, proposer, evaluate_candidate, baseline):
    """Eight candidate attempts, four rounds. Rejections also consume attempts.

    Use one distinct archive per prompt-only/meta arm with identical reset IDs
    and fixed manifests. The callback is owned by the trusted evaluator, not by
    candidate code. Failures persist, and no retry can erase a failed attempt.
    """
    kind = archive.manifest["search_kind"]
    if kind not in ("prompt_only", "meta_harness"):
        raise ValueError("Search arm is not pinned")
    for round_index in range(4):
        for slot in range(2):
            row = proposer.propose(archive, kind=kind, baseline=baseline)
            if row["status"] != "admitted":
                continue
            directory = archive.root / "candidates" / row["candidate_id"]
            write_json(directory / "round.json", {"round": round_index, "slot": slot})
            results, aggregate = evaluate_candidate(
                directory / "harness.py", directory / "traces"
            )
            archive.record(row["candidate_id"], results, aggregate)
