"""Unscored GPU commissioning of all eight MAX events on one pinned source.

No model calls and no benchmark-success claims. Fixed-step event probes are
explicitly separate from the unchanged release manifest and its final scores.
"""

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from astra_reversal.hardware_interface.common import Events, digest, write_json
from astra_reversal.hardware_interface.liberomax import (
    SOURCE_COMMITS,
    Environment,
    check_checkout,
    commissioning_cases,
    event_probe,
    load_release,
    verify_pair,
)
from astra_reversal.hardware_interface.proxy import NATIVE_KEYS, Limits, Proxy
from astra_reversal.hardware_interface.schema import validate_description


def probe_arm(case, arm, interface, directory):
    directory.mkdir(parents=True, exist_ok=False)
    events = Events(directory / "events", {"condition": interface, "arm": arm})
    env = None
    try:
        env = Environment(case, arm, directory / "trace.jsonl", horizon=6)
        if env.reset_receipt["initial_success"]:
            raise ValueError("initially_solved_commissioning_case")
        limits = Limits(
            **{
                **asdict(Limits()),
                "steps": 6,
                "max_repeat": 1,
                **{
                    k: None
                    for k in (
                        "wall_seconds",
                        "workflow_tokens",
                        "tool_calls",
                        "observer_calls",
                        "actor_output_tokens",
                        "observer_output_tokens",
                        "context_tokens",
                    )
                },
            }
        )
        proxy = Proxy(
            env,
            env.observation,
            episode_id="commission",
            controller=env.controller,
            events=events,
            limits=limits,
        )
        validate_description(proxy.describe())
        actions = (
            [0, 0, 0, 0, 0, 0, -1],
            [0.02, 0, 0.02, 0, 0, 0, -1],
            [0, 0.02, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, -1],
            [-0.02, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, -1],
        )
        for action in actions:
            if proxy.terminal:
                break
            observed = proxy.observe(sorted(NATIVE_KEYS))
            # Query records describe this deterministic commissioning controller,
            # not an Astra request or a learned policy.
            env.wrapped.record_policy_query([action], source="commissioning_controller")
            if interface == "F":
                result = proxy.act(
                    {
                        "channel": "robot.controller_command",
                        "value": action,
                        "duration_steps": 1,
                        "metadata": {
                            "mode": "configured_controller",
                            "units": "normalized_controller_input",
                            "episode_id": "commission",
                            "observation_step": observed["simulator_step"],
                        },
                    }
                )
            else:
                result = proxy.step(action, 1, observed["simulator_step"])
            if not result["accepted"]:
                raise RuntimeError("commissioning_native_action_rejected")
        row = env.wrapped.record_outcome(proxy.success)
        row["settled_state_sha256"] = env.reset_receipt["settled_state_sha256"]
        row["terminal_state_sha256"] = digest(
            np.asarray(env.env.get_sim_state()).tolist()
        )
        row["controller_sha256"] = digest(env.controller)
        env.terminal_snapshot(directory / "terminal.npz")
        for key, name in (
            ("agentview_image", "front"),
            ("robot0_eye_in_hand_image", "wrist"),
        ):
            Image.fromarray(np.ascontiguousarray(env.observation[key][::-1])).save(
                directory / (name + ".png")
            )
        write_json(directory / "receipt.json", row)
        return row
    finally:
        if env is not None:
            env.close()
        events.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source", choices=("plus", "pro"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or os.environ.get("MUJOCO_GL") != "egl":
        raise ValueError("commission_on_osmo_linux_egl")
    check_checkout(args.source_root, SOURCE_COMMITS[args.source])
    manifest = load_release(args.max_root, "lite")
    args.out.mkdir(parents=True, exist_ok=False)
    pairs = []
    for original in commissioning_cases(manifest, args.source):
        case = event_probe(original)
        event = original["scenario"]["change_type"]
        parent = args.out / event
        write_json(
            parent / "probe-identity.json",
            {
                "released_case": original,
                "probe_case": case,
                "scored": False,
                "model_calls": 0,
                "changed_for_commissioning": ["scenario_id", "trigger", "horizon"],
            },
        )
        baseline = {}
        for interface in ("F", "B0", "B"):
            rows = {
                arm: probe_arm(
                    copy.deepcopy(case), arm, interface, parent / interface / arm
                )
                for arm in ("control", "intervention")
            }
            matched = verify_pair(rows["control"], rows["intervention"])
            if not matched["trigger_reached"] or matched["event_step"] != 2:
                raise RuntimeError("event_probe_did_not_fire_exactly_once_at_step_two")
            for arm, row in rows.items():
                signature = {
                    key: row[key]
                    for key in (
                        "settled_state_sha256",
                        "terminal_state_sha256",
                        "controller_sha256",
                    )
                }
                if arm in baseline and baseline[arm] != signature:
                    raise RuntimeError("interface_native_equivalence_failed")
                baseline.setdefault(arm, signature)
            pairs.append({"event": event, "interface": interface, **matched})
            print(
                json.dumps(
                    {
                        "commissioned_event": event,
                        "interface": interface,
                        "source": args.source,
                        "native_prefix_passed": True,
                        "scored": False,
                    }
                ),
                flush=True,
            )
    write_json(
        args.out / "commissioning.json",
        {
            "status": "passed",
            "source": args.source,
            "pairs": pairs,
            "events": 8,
            "interfaces": 3,
            "model_calls": 0,
            "scored": False,
            "live_astra_ready": False,
            "remaining_gate": "live Astra continuation with exact shared prefix and provider-seed accounting",
        },
    )


if __name__ == "__main__":
    main()
