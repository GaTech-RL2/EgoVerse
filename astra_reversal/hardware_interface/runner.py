"""Fresh per-trial sessions, common control authority, and explicit termination."""

import json
import time
from dataclasses import asdict

import jsonschema

from .common import digest, strict_json, write_json
from .protocol import SHARED_PROMPT
from .provider import BudgetEnd, Meter, ModelFailure, Observer, Session
from .proxy import CommandError


def obj(properties):
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def tools_for(condition, action_dim=7):
    integer, string = {"type": "integer"}, {"type": "string"}
    vector = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": action_dim,
        "maxItems": action_dim,
    }
    definitions = [
        (
            "source_search",
            "Search the shared audited robotics source view with a literal substring.",
            obj({"query": string}),
        ),
        (
            "source_read",
            "Read an allowlisted relative file; begin at 1; at most 200 lines.",
            obj({"path": string, "start": integer, "lines": integer}),
        ),
        (
            "scratch_execute",
            "Execute standard-library Python in isolated scratch. Set result to a JSON value. Robot calls use documented proxy functions; each nested call consumes a tool call. No simulator or network access.",
            obj({"code": string}),
        ),
        (
            "finish",
            "End this actor trial; the evaluator decides the outcome privately.",
            obj({}),
        ),
    ]
    if condition == "F":
        definitions.extend(
            [
                (
                    "describe_device",
                    "Inspect typed channels, units, bounds and controller conventions.",
                    obj({}),
                ),
                (
                    "read",
                    "Read a current sensor channel while physics is paused.",
                    obj(
                        {
                            "channel": string,
                            "max_age_ms": {"type": "number", "minimum": 0},
                        }
                    ),
                ),
                (
                    "read_latest",
                    "Read all allowed observation channels at the current simulator step.",
                    obj({}),
                ),
                (
                    "act",
                    "Apply one atomic controller command for bounded repeated steps.",
                    obj(
                        {
                            "envelope": obj(
                                {
                                    "channel": {
                                        "type": "string",
                                        "enum": ["robot.controller_command"],
                                    },
                                    "value": vector,
                                    "duration_steps": integer,
                                    "metadata": obj(
                                        {
                                            "mode": {
                                                "type": "string",
                                                "enum": ["configured_controller"],
                                            },
                                            "units": {
                                                "type": "string",
                                                "enum": ["normalized_controller_input"],
                                            },
                                            "episode_id": string,
                                            "observation_step": integer,
                                        }
                                    ),
                                }
                            )
                        }
                    ),
                ),
            ]
        )
    else:
        definitions.extend(
            [
                (
                    "observe",
                    "Read native sensor keys discovered in the audited source. Nonempty keys required.",
                    obj(
                        {
                            "keys": {
                                "type": "array",
                                "items": string,
                                "minItems": 1,
                            }
                        }
                    ),
                ),
                (
                    "step",
                    "Submit a native controller vector; discover conventions from source. The observation step must be current.",
                    obj(
                        {
                            "action": vector,
                            "repeat_steps": integer,
                            "observation_step": integer,
                        }
                    ),
                ),
            ]
        )
    if condition == "B":
        definitions.append(
            (
                "describe_visible",
                "Ask the observer for visible evidence from current cameras only; no advice or plans.",
                obj(
                    {
                        "request_kind": {
                            "type": "string",
                            "enum": list(Observer.REQUESTS),
                        }
                    }
                ),
            )
        )
    return [
        {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": schema,
            "strict": True,
        }
        for name, description, schema in definitions
    ]


class Router:
    def __init__(self, proxy, source, condition, *, scratch=None, observer=None):
        self.proxy, self.source, self.condition = proxy, source, condition
        self.scratch, self.observer = scratch, observer
        self.tools = tools_for(condition, proxy.action_dim)
        self.schemas = {t["name"]: t["parameters"] for t in self.tools}
        self.calls, self.invalid_commands, self.recoveries = 0, 0, 0
        self.previous_rejected = False

    def dispatch(self, tool, arguments):
        if (
            self.proxy.limits.tool_calls is not None
            and self.calls >= self.proxy.limits.tool_calls
        ):
            raise BudgetEnd("tool_call_limit")
        self.proxy.available()
        self.calls += 1
        self.proxy.events.emit(
            "tool_request",
            tool=tool,
            arguments=arguments,
            simulator_step=self.proxy.step_count,
        )
        started = time.monotonic()
        rejected = False
        try:
            if tool not in self.schemas:
                raise CommandError("tool_not_allowed")
            jsonschema.validate(arguments, self.schemas[tool])
            if tool == "describe_device":
                result = self.proxy.describe()
            elif tool == "read":
                result = self.proxy.read(**arguments)
            elif tool == "read_latest":
                result = [
                    self.proxy.read(channel, 1000) for channel in self.proxy.sensors
                ]
            elif tool == "act":
                result = self.proxy.act(**arguments)
            elif tool == "observe":
                result = self.proxy.observe(**arguments)
            elif tool == "step":
                result = self.proxy.step(**arguments)
            elif tool == "source_search":
                result = self.source.search(**arguments)
            elif tool == "source_read":
                result = self.source.read(**arguments)
            elif tool == "scratch_execute":
                if self.scratch is None:
                    raise RuntimeError("scratch_isolation_unavailable")
                remaining = self.proxy.remaining_wall_seconds()
                result = self.scratch.execute(
                    arguments["code"],
                    self.dispatch,
                    seconds=15 if remaining is None else min(15, remaining),
                )
            elif tool == "describe_visible":
                if self.observer is None:
                    raise RuntimeError("observer_unavailable")
                visible = self.proxy.observe(self.proxy.camera_keys)
                remaining = self.proxy.remaining_wall_seconds()
                result = self.observer.describe(
                    arguments["request_kind"], visible, timeout=remaining
                )
            else:
                result = self.proxy.finish()
            rejected = isinstance(result, dict) and (
                result.get("accepted") is False or "error" in result
            )
        except (CommandError, ValueError, jsonschema.ValidationError) as error:
            rejected = True
            # No source paths or exception locals cross into actor context.
            reason = (
                error.reason if isinstance(error, CommandError) else "invalid_arguments"
            )
            result = {"error": reason, "simulator_step": self.proxy.step_count}
        self.invalid_commands += int(rejected)
        self.recoveries += int(self.previous_rejected and not rejected)
        self.previous_rejected = rejected
        self.proxy.events.emit(
            "tool_response",
            tool=tool,
            response=result,
            response_sha256=digest(result),
            simulator_step=self.proxy.step_count,
            latency_seconds=time.monotonic() - started,
        )
        return result


def run_trial(
    proxy,
    source,
    condition,
    task_instruction,
    model,
    *,
    scratch,
    post=None,
    shared_prompt=SHARED_PROMPT,
    source_entry="libero/libero/envs/env_wrapper.py",
):
    meter = Meter(proxy.limits.workflow_tokens)
    actor = Session(model, proxy.limits, meter, proxy.events, post=post)
    observer = (
        Observer(
            Session(
                model, proxy.limits, meter, proxy.events, post=post, role="observer"
            )
        )
        if condition == "B"
        else None
    )
    router = Router(proxy, source, condition, scratch=scratch, observer=observer)
    common = (
        "Shared source entry point: " + source_entry + ". "
        "Inspect native controller and observation conventions through source tools. "
        "Only normalized controller inputs within [-1,1] are accepted. Physics advances only on accepted steps and pauses during reasoning. "
        "Camera images are delivered upright in every arm. "
        "The episode horizon, controller bounds and repeated-step limit are enforced. Scratch has standard-library Python, /sources read-only, and /scratch writable; "
        "scratch functions are read(channel,max_age_ms)/act(envelope) in F and observe(keys)/step(action,repeat_steps,observation_step) in B/B0. "
        "No hidden simulator objects exist there. Usage is recorded. A null resource setting means no experiment cap.\nEpisode and request settings: "
        + json.dumps(asdict(proxy.limits))
    )
    instructions = shared_prompt + "\n\n" + common
    # Keep the goal in every request even if the provider needs to truncate old
    # conversation items at its actual context-window boundary.
    if model.get("context_truncation") == "auto":
        instructions += "\nCurrent user task: " + json.dumps(task_instruction)
    if condition == "F":
        instructions += (
            "\nUse only the universal interface for device calls.\nDevice description: "
            + json.dumps(proxy.describe())
        )
    elif condition == "B":
        instructions += "\nThe separate observer can report current visible evidence with describe_visible; its usage is included in the workflow totals."
    actor.history = [{"role": "user", "content": task_instruction}]
    proxy.wall_start = proxy.clock()
    proxy.events.emit(
        "trial_start",
        condition=condition,
        instruction=task_instruction,
        system_prompt=instructions,
        system_prompt_sha256=digest(instructions),
        tools_sha256=digest(router.tools),
        limits=asdict(proxy.limits),
        model=model,
        source_sha256=source.manifest["sha256"],
    )
    actor_started = True
    try:
        while proxy.terminal is None:
            proxy.available()
            remaining = proxy.remaining_wall_seconds()
            response = actor.request(instructions, router.tools, timeout=remaining)
            # A response after the wall deadline cannot trigger an action.
            proxy.available()
            calls = [r for r in response["output"] if r.get("type") == "function_call"]
            if len(calls) != 1:
                raise ModelFailure("one_tool_call_required")
            call = calls[0]
            result = router.dispatch(call["name"], strict_json(call["arguments"]))
            actor.tool_result(call["call_id"], result)
    except BudgetEnd as error:
        proxy.terminal = (
            "TIMEOUT_WALL" if str(error) == "wall_limit" else "BUDGET_EXHAUSTED"
        )
        proxy.events.emit("error", error_class="BudgetEnd", reason=str(error))
    except CommandError as error:
        proxy.terminal = proxy.terminal or "INVALID_ACTION"
        proxy.events.emit("error", error_class="CommandError", reason=error.reason)
    except ModelFailure as error:
        proxy.terminal = "MODEL_ERROR"
        proxy.events.emit("error", error_class="ModelFailure", reason=str(error))
    except Exception as error:
        proxy.terminal = "TOOL_ERROR"
        proxy.events.emit("error", error_class=type(error).__name__)
    finally:
        try:
            source.verify()
        except Exception as error:
            proxy.terminal = "PROTOCOL_DEVIATION"
            proxy.events.emit(
                "error", error_class=type(error).__name__, reason="source_view_changed"
            )
    outcome = {
        "success": proxy.success,
        "terminal_reason": proxy.terminal,
        "sim_steps": proxy.step_count,
        "wall_s": proxy.clock() - proxy.wall_start,
        "simulator_and_step_evaluator_seconds": proxy.simulator_seconds,
        "provider_wall_seconds": sum(
            v["provider_wall_seconds"] for v in meter.usage.values()
        ),
        "tool_latency_note": "Per-tool timings are in events; scratch and observer tool latencies include nested calls and must not be summed as disjoint phases.",
        "actor_started": actor_started,
        "tool_calls": router.calls,
        "invalid_commands": router.invalid_commands,
        "invalid_actions": proxy.invalid_actions,
        "safety_attempts": proxy.safety_attempts,
        "applied_safety_violations": proxy.applied_violations,
        "recoveries": router.recoveries,
        "observer_turns": observer.calls if observer else 0,
        "usage_by_role": meter.usage,
        "known_workflow_tokens": meter.total,
        "unknown_usage_records": meter.unknown,
        "estimated_cost_usd": None,
        "cost_unavailable_reason": "No verified price schedule for the configured serving endpoint",
        "censored_wall": proxy.terminal == "TIMEOUT_WALL",
        "model_snapshot_pinned": model["snapshot_pinned"],
    }
    proxy.events.emit("trial_end", **outcome)
    write_json(
        proxy.events.directory / "outcome.json", {**proxy.events.identity, **outcome}
    )
    return outcome
