"""Unscored live transport checks, using only commissioning camera frames."""

import base64

from .common import Events, strict_json, write_json
from .provider import HTTP, Meter, Observer, Session
from .proxy import Limits


def transport_smoke(manifest, prepared, destination):
    limits = Limits(**manifest["limits"])
    events = Events(destination, {"kind": "model_transport_commissioning"})
    meter = Meter(limits.workflow_tokens)
    post = HTTP(manifest["model"]["base_url"])
    actor = Session(manifest["model"], limits, meter, events, post=post)
    cameras = {}
    for label in ("front", "wrist"):
        path = prepared / "calibration-F" / (label + ".png")
        cameras[label] = {
            "encoding": "image/png",
            "shape": [manifest["image_size"], manifest["image_size"], 3],
            "base64": base64.b64encode(path.read_bytes()).decode(),
        }
    observation = {
        "simulator_step": 0,
        "timestamp": "commissioning frame; simulator is paused",
        "observations": cameras,
    }
    inspect = {
        "type": "function",
        "name": "inspect_frame",
        "strict": True,
        "description": "Request the current camera frames.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }
    report = {
        "type": "function",
        "name": "report_camera_views",
        "strict": True,
        "description": "Name the supplied camera views and state whether a robot is visible.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "camera_views": {"type": "array", "items": {"type": "string"}},
                "robot_visible": {"type": "boolean"},
            },
            "required": ["camera_views", "robot_visible"],
        },
    }
    try:
        actor.history = [{"role": "user", "content": "Use inspect_frame once."}]
        response = actor.request(
            "API transport check. Call the requested tool.", [inspect]
        )
        calls = [r for r in response["output"] if r.get("type") == "function_call"]
        if len(calls) != 1 or calls[0]["name"] != "inspect_frame":
            raise ValueError("transport_inspect_tool_failed")
        actor.tool_result(calls[0]["call_id"], observation)
        response = actor.request(
            "Report the supplied camera views using the tool.", [report]
        )
        calls = [r for r in response["output"] if r.get("type") == "function_call"]
        if len(calls) != 1 or calls[0]["name"] != "report_camera_views":
            raise ValueError("transport_image_tool_failed")
        value = strict_json(calls[0]["arguments"])
        if len(value.get("camera_views", [])) != 2:
            raise ValueError("transport_camera_pair_failed")
        observer = Observer(
            Session(
                manifest["model"], limits, meter, events, post=post, role="observer"
            )
        )
        observer.describe("scene", observation)
        receipt = {
            "passed": True,
            "scored_trial": False,
            "model": manifest["model"],
            "usage_by_role": meter.usage,
            "workflow_tokens": meter.total,
            "tool_round_trip": True,
            "image_pair": True,
            "observer_schema_and_output_cap": True,
        }
        write_json(destination / "transport.json", receipt)
        return receipt
    finally:
        events.close()
