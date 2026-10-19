"""Generated Python control versus direct actions through one adaptive interface."""

import copy
import json

from .adaptive import CONDITION
from .runner import Router

COMMON = """This trial uses the F-adaptive hardware interface. You can enable hardware capabilities, measure dynamics, and evolve observation/coordinate-conversion helpers and interface notes. Interface helper definitions must be read_only=true in both policy modes. They may read and transform observations but cannot actuate. Calibration probes use the same bounded native commands and count in the episode. No task-specific skills, demonstrations, hidden object poses, reward or success queries are provided. The episode ends automatically on native task success or at its full control-step horizon. Voluntary finish is unavailable: keep observing, acting and revising your approach while the episode is active, including after an unsuccessful attempt. These mode rules take precedence over generic interface examples."""
INSTRUCTIONS = {
    "CAP": COMMON
    + "\nYour policy is generated Python code (Code-as-Policies style). Use scratch_execute to run closed-loop programs over read(channel,max_age_ms), read_capability(name), and act(envelope). Programs can react to each new observation and issue multiple validated commands per model call. Every act uses current observation_step and the same max_duration_steps bound. Set result to JSON summarizing observed progress. Use subsequent model calls to revise your program when needed. Robot-control code is the policy; top-level act calls are not available.",
    "direct": COMMON
    + "\nYou are the direct robot policy. Select each next native act envelope yourself from current observations, using top-level act calls. Each command may repeat for the documented bounded duration. Re-observe and replan through model calls. Observation helpers are read-only; generated controller code, executable robot loops, and scratch_execute are not available in this mode.",
}


class PolicyRouter(Router):
    def __init__(self, *args, policy_mode, allow_early_finish=False, **kwargs):
        if policy_mode not in INSTRUCTIONS:
            raise ValueError("unknown_policy_mode")
        super().__init__(*args, **kwargs)
        if self.condition != CONDITION:
            raise ValueError("F_adaptive_required_for_policy_comparison")
        self.policy_mode = policy_mode
        hidden = {"act"} if policy_mode == "CAP" else {"scratch_execute"}
        if not allow_early_finish:
            hidden.add("finish")
            # Remove the dispatch schema too, so nested generated calls cannot
            # terminate an episode through the inherited Router dispatcher.
            self.schemas.pop("finish", None)
        self.tools = [copy.deepcopy(t) for t in self.tools if t["name"] not in hidden]
        for tool in self.tools:
            if tool["name"] == "scratch_execute":
                # Removing the top-level act tool must preserve its wire contract
                # for generated controllers. Direct sees this same schema in act.
                tool["description"] += (
                    "\nScratch functions: read(channel,max_age_ms=1000), "
                    "read_capability(name), act(envelope). read returns a sensor "
                    "envelope whose metadata contains episode_id and simulator_step. "
                    "For each act, copy the current read metadata.episode_id into "
                    "command metadata.episode_id and metadata.simulator_step into "
                    "command metadata.observation_step. duration_steps is bounded "
                    "by the device's max_duration_steps. act takes the object "
                    "described by this JSON Schema.\nact(envelope) schema: "
                    + json.dumps(
                        self.schemas["act"]["properties"]["envelope"], sort_keys=True
                    )
                )
            if tool["name"] == "define_interface_tool":
                tool["parameters"]["properties"]["read_only"] = {
                    "type": "boolean",
                    "enum": [True],
                }
                tool["description"] += (
                    " This comparison permits observation helpers only (read_only=true)."
                )
                self.schemas[tool["name"]] = tool["parameters"]
        self.model_tools = {t["name"] for t in self.tools}

    def dispatch_model(self, tool, arguments):
        if tool not in self.model_tools:
            return self.dispatch(
                "policy_mode_tool_not_allowed", {"attempted_tool": tool}
            )
        return self.dispatch(tool, arguments)
