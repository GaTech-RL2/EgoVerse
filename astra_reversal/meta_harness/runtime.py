"""One outstanding bounded Astra request while System 1 continues executing."""

import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from astra_reversal.records import digest

from .programs import Programs, public_observation
from .schema import exact


class Trace:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)

    def event(self, stream, value):
        with (self.directory / (stream + ".jsonl")).open("a") as output:
            output.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")

    def observation(self, raw, public):
        paths = {}
        for camera in ("observation/image", "observation/wrist_image"):
            relative = f"{public['observation_id']}_{camera.split('/')[-1]}.png"
            Image.fromarray(raw[camera]).save(self.directory / relative)
            paths[camera] = relative
        self.event(
            "observations", {**public, "raw_images": paths, "raw_sha256": digest(raw)}
        )


class Runtime:
    def __init__(
        self,
        *,
        compiler,
        limits,
        harness,
        worker,
        original_goal,
        episode_id,
        trace,
        mode="async",
        clock=time.monotonic,
    ):
        if mode not in ("async", "synchronous_diagnostic", "native"):
            raise ValueError(
                "Choose actual asynchronous execution or labelled synchronous debugging"
            )
        if (mode == "native") != (worker is None):
            raise ValueError("Native has no reasoner; runtime arms require a reasoner")
        self.compiler, self.limits, self.harness = compiler, limits, harness
        self.worker, self.goal, self.episode_id = worker, original_goal, episode_id
        self.trace, self.mode, self.clock = trace, mode, clock
        self.programs = Programs(compiler, limits)
        self.pool = ThreadPoolExecutor(max_workers=1) if worker else None
        self.pending = None
        self.calls = 0
        self.memory, self.history, self.receipts = {}, [], []
        self.contract_errors = []
        self.provider_records = []
        self.closed = False

    def _finish(self, live, *, ended=False):
        future, request = self.pending
        self.pending = None
        try:
            result = future.result()
        except Exception as error:
            result = {
                "call": None,
                "record": {"error": type(error).__name__, "usage": None},
            }
        result["record"].setdefault(
            "latency_seconds", self.clock() - request["binding"]["captured_at"]
        )
        self.provider_records.append(result["record"])
        self.trace.event(
            "astra_responses", {"request_id": request["request_id"], **result}
        )
        if ended:
            receipt = {
                "requested_call": result.get("call"),
                "actually_executed_call": None,
                "validation_error": "episode_ended",
                **self.programs.status(),
            }
        elif result.get("call") is None:
            receipt = {
                "requested_call": None,
                "actually_executed_call": None,
                "validation_error": "provider_or_token_contract_error",
                "fallback": "retain_valid_program_else_native",
                **self.programs.status(),
            }
        else:
            receipt = self.programs.apply(
                result["call"], request["binding"], live, self.clock()
            )
        receipt.update(
            request_id=request["request_id"],
            action=self.programs.step,
            source_observation_id=request["binding"]["observation_id"],
        )
        if not ended:
            receipt.update(
                live_observation_sha256=digest(live),
                live_proprioception=live["observation/state"].tolist(),
            )
        self.receipts.append(receipt)
        self.trace.event("tool_events", receipt)

    def boundary(self, live, step):
        if self.closed:
            raise ValueError("Episode is already closed")
        self.programs.boundary(live, step)
        if self.pending is not None and self.pending[0].done():
            self._finish(live)
        now = self.clock()
        raw, observation = public_observation(
            live,
            step=step,
            now=now,
            stage_id=self.programs.stage_id,
            episode_id=self.episode_id,
        )
        observation.update(
            original_goal=self.goal, active_program=self.programs.status()
        )
        self.trace.observation(raw, observation)
        frame = {
            "observation_id": observation["observation_id"],
            "action": step,
            "images": {
                key: raw[key]
                for key in ("observation/image", "observation/wrist_image")
            },
        }
        self.history.append({"public": observation, "frame": frame})
        self.history = self.history[-16:]
        if (
            self.worker is None
            or self.calls >= self.limits.calls
            or self.pending is not None
        ):
            return self.programs.choice
        try:
            public_history = [row["public"] for row in self.history[:-1]]
            output = self.harness.run(
                observation,
                public_history,
                list(self.compiler.cards.values()),
                {**self.memory, "recent_tool_events": self.receipts[-4:]},
            )
            exact(
                output, ("request", "card_ids", "history_ids", "instruction", "memory")
            )
            if type(output["request"]) is not bool or not isinstance(
                output["memory"], dict
            ):
                raise ValueError("Invalid scheduling or memory output")
            ids, history_ids = output["card_ids"], output["history_ids"]
            if (
                not isinstance(ids, list)
                or not 1 <= len(ids) <= 6
                or len(set(ids)) != len(ids)
                or any(key not in self.compiler.cards for key in ids)
            ):
                raise ValueError("Retrieve one to six eligible cards")
            history_index = {
                row["public"]["observation_id"]: row for row in self.history[:-1]
            }
            if (
                not isinstance(history_ids, list)
                or len(history_ids) > 1
                or any(key not in history_index for key in history_ids)
            ):
                raise ValueError("Select at most one genuine previous raw image pair")
            if (
                not isinstance(output["instruction"], str)
                or len(output["instruction"]) > 2000
            ):
                raise ValueError("Harness instruction exceeds its bounded context")
            if len(json.dumps(output["memory"]).encode()) > 4096:
                raise ValueError("Memory exceeds four KB")
            self.memory = output["memory"]
            if not output["request"]:
                return self.programs.choice
            request = {
                "request_id": f"{self.episode_id}:runtime:{self.calls}",
                "binding": {**observation, "retrieved_ids": ids},
                "context": {
                    "current": observation,
                    "original_goal": self.goal,
                    "cards": [self.compiler.cards[key] for key in ids],
                    "harness_instruction": output["instruction"],
                    "memory": self.memory,
                    "memory_status": "candidate hypotheses, not verified facts",
                    "history": [history_index[key]["public"] for key in history_ids],
                    "tool_events": self.receipts[-4:],
                },
                "frames": [history_index[key]["frame"] for key in history_ids]
                + [frame],
            }
            request = copy.deepcopy(request)
            self.trace.event(
                "astra_requests",
                {
                    "request_id": request["request_id"],
                    "context": request["context"],
                    "binding": request["binding"],
                    "frame_observation_ids": [
                        f["observation_id"] for f in request["frames"]
                    ],
                    "request_sha256": digest(request),
                },
            )
            self.calls += 1  # Reserve before dispatch; failures and repairs consume the same budget.
            self.pending = self.pool.submit(self.worker, request), request
            if self.mode == "synchronous_diagnostic":
                self._finish(live)
        except Exception as error:
            self.contract_errors.append(type(error).__name__ + ": " + str(error)[:200])
            self.trace.event(
                "contract_errors", {"action": step, "error": self.contract_errors[-1]}
            )
        return self.programs.choice

    def close(self):
        """Drain the single bounded request after motion stops, never apply it late."""
        self.closed = True
        if self.pending is not None:
            self._finish(None, ended=True)
        if self.pool is not None:
            self.pool.shutdown(wait=True)
        return {
            "mode": self.mode,
            "runtime_requests": self.calls,
            "contract_errors": self.contract_errors,
            "tool_events": self.receipts,
            "provider_records": self.provider_records,
        }
