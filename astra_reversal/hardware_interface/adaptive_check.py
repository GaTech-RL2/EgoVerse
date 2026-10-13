"""Unscored Linux checks of evolving tools against identical native commands."""

import json

import numpy as np

from .adaptive import CONDITION
from .common import Events, digest, write_json
from .isolation import Scratch
from .proxy import Proxy
from .runner import Router


def check(environment_factory, source, runtime, directory, limits, *, sensors=None):
    """Run on an allocated simulator GPU; no model requests or learned actions."""
    directory.mkdir(exist_ok=False)
    rows = []
    for condition in ("F", CONDITION):
        events = Events(
            directory / condition,
            {"kind": "adaptive_commission", "condition": condition},
        )
        env = environment_factory()
        try:
            proxy = Proxy(
                env,
                env.observation,
                episode_id="adaptive-calibration",
                controller=env.controller,
                events=events,
                limits=limits,
                sensors=sensors,
            )
            scratch = Scratch(
                runtime["root"],
                runtime["executable"],
                directory / (condition + "-jail"),
                source.root,
                condition,
            )
            router = Router(proxy, source, condition, scratch=scratch)
            action = [0.0] * proxy.action_dim
            gripper = env.controller["action_order"].index("gripper")
            action[gripper] = -1.0
            if "base_mode" in env.controller["action_order"]:
                action[env.controller["action_order"].index("base_mode")] = -1.0
            action[0] = 0.02

            def envelope(repeat):
                return {
                    "channel": "robot.controller_command",
                    "value": action,
                    "duration_steps": repeat,
                    "metadata": {
                        "mode": "configured_controller",
                        "units": "normalized_controller_input",
                        "episode_id": proxy.episode_id,
                        "observation_step": proxy.step_count,
                    },
                }

            if condition == CONDITION:
                for capability in router.evolution.capabilities:
                    enabled = router.dispatch(
                        "enable_capability",
                        {"name": capability, "reason": "native commissioning"},
                    )
                    read = router.dispatch("read_capability", {"name": capability})
                    if "error" in enabled or "error" in read:
                        raise ValueError("adaptive_capability_check_failed")
                    json.dumps(read, allow_nan=False)
                probe = router.dispatch(
                    "probe_dynamics",
                    {
                        "action": action,
                        "repeat_steps": 2,
                        "channels": ["robot.eef_position"],
                        "reason": "native calibration check",
                    },
                )
                if probe.get("executed_steps") != 2:
                    raise ValueError("adaptive_probe_step_accounting")
                definition = {
                    "name": "native_step",
                    "description": "commissioning proxy passthrough",
                    "code": "result = act(params['command'])",
                    "read_only": False,
                    "reason": "verify isolated tool execution",
                }
                revision = router.dispatch("define_interface_tool", definition)
                moved = router.dispatch(
                    "call_interface_tool",
                    {
                        "name": "native_step",
                        "arguments_json": json.dumps({"command": envelope(1)}),
                    },
                )
                if not moved.get("result", {}).get("accepted"):
                    raise ValueError("adaptive_scratch_command_failed")
                router.dispatch(
                    "define_interface_tool", {**definition, "read_only": True}
                )
                blocked = router.dispatch(
                    "call_interface_tool",
                    {
                        "name": "native_step",
                        "arguments_json": json.dumps({"command": envelope(1)}),
                    },
                )
                if (
                    blocked.get("result", {}).get("error") != "read_only_interface_tool"
                    or proxy.step_count != 3
                ):
                    raise ValueError("adaptive_readonly_command_guard")
                router.dispatch(
                    "restore_interface",
                    {
                        "revision": revision["revision"],
                        "reason": "verify version restore",
                    },
                )
                if router.evolution.state["tools"]["native_step"]["read_only"]:
                    raise ValueError("adaptive_restore_failed")
            else:
                for repeat in (2, 1):
                    if not router.dispatch("act", {"envelope": envelope(repeat)})[
                        "accepted"
                    ]:
                        raise ValueError("adaptive_control_check_failed")
            env.terminal_snapshot(events.directory / "terminal.npz")
            with np.load(events.directory / "terminal.npz") as terminal:
                state_hash = digest(
                    {key: terminal[key].tolist() for key in terminal.files}
                )
            rows.append(
                {
                    "condition": condition,
                    "reset": env.reset_receipt,
                    "state_sha256": state_hash,
                    "steps": proxy.step_count,
                }
            )
        finally:
            env.close()
            events.close()
    passed = all(
        rows[0][key] == rows[1][key] for key in ("reset", "state_sha256", "steps")
    )
    result = {"passed": passed, "model_calls": 0, "scored": False, "rows": rows}
    write_json(directory / "receipt.json", result)
    if not passed:
        raise ValueError("adaptive_native_equivalence_failed")
    return result
