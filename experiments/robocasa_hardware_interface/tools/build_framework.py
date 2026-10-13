"""Build standalone framework documentation from the implemented tool contracts.

Run from the repository root with python -m; no simulator or provider calls.
Examples use synthetic sensors and never count as benchmark evidence.
"""

import html
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import jsonschema
import numpy as np

from astra_reversal.hardware_interface.proxy import SENSORS as LIBERO_SENSORS
from astra_reversal.hardware_interface.proxy import Limits, Proxy
from astra_reversal.hardware_interface.robocasa import SENSORS as ROBOCASA_SENSORS
from astra_reversal.hardware_interface.runner import Router, tools_for
from astra_reversal.hardware_interface.schema import validate_description

from .framework_diagrams import adaptive as adaptive_diagram
from .framework_diagrams import overview as overview_diagram

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/robocasa_hardware_interface/figures"
STAMP = "2026-10-13T00:00:00+00:00"
F, B0, B = "#087f70", "#ab5e19", "#7750b3"
INK, MUTED, LINE = "#142d46", "#49627d", "#ccd9e6"


def pretty(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)


class ExampleEnvironment:
    def __init__(self, sensors):
        self.observation = {
            key: np.zeros(shape or [128, 128, 3], dtype=float if shape else np.uint8)
            for key, _, _, shape in sensors.values()
        }
        for key, _, _, shape in sensors.values():
            if "quat" in key:
                self.observation[key][-1] = 1.0
        self.observation[sensors["robot.eef_position"][0]] = np.array([0.4, 0.1, 0.3])

    def read_sensors(self):
        return self.observation

    def step(self, action):
        return self.observation, None, None, None

    def check_success(self):
        return False


class ExampleEvents:
    def emit(self, *args, **kwargs):
        pass


def contracts():
    result = {
        "notice": "Synthetic documentation examples. No simulator, robot, model or evaluation was run to create them.",
        "schema_version": "hardware-1",
        "implementations": {},
    }
    for name, sensors, size, device in (
        ("LIBERO", LIBERO_SENSORS, 7, "libero-panda"),
        ("RoboCasa", ROBOCASA_SENSORS, 12, "robocasa-panda-omron"),
    ):
        env = ExampleEnvironment(sensors)
        controller = {
            "version": "documentation-fixture",
            "device_id": device,
            "frequency_hz": 20,
            "input_min": [-1.0] * size,
            "input_max": [1.0] * size,
            "description": "Illustrative native controller; read live metadata for order, scaling and frames.",
        }
        proxy = Proxy(
            env,
            env.observation,
            episode_id="example-episode",
            controller=controller,
            events=ExampleEvents(),
            sensors=sensors,
            limits=Limits(wall_seconds=None),
            clock=lambda: 0.0,
        )
        description = proxy.describe()
        validate_description(description)
        hardware_schema = json.loads(
            (
                ROOT
                / "experiments/libero_hardware_interface/schemas/hardware.schema.json"
            ).read_text()
        )
        jsonschema.validate(description, hardware_schema)
        action = [0.0] * size
        action[0] = 0.1
        native_key = sensors["robot.eef_position"][0]
        calls = {
            "describe_device": {},
            "read": {"channel": "robot.eef_position", "max_age_ms": 1000},
            "read_latest": {},
            "act": {
                "envelope": {
                    "channel": "robot.controller_command",
                    "value": action,
                    "duration_steps": 3,
                    "metadata": {
                        "mode": "configured_controller",
                        "units": "normalized_controller_input",
                        "episode_id": "example-episode",
                        "observation_step": 0,
                    },
                }
            },
            "observe": {"keys": [native_key]},
            "step": {"action": action, "repeat_steps": 3, "observation_step": 0},
            "describe_visible": {"request_kind": "gripper_and_objects"},
            "source_search": {
                "query": "input_ref_frame" if size == 12 else "use_delta"
            },
            "source_read": {
                "path": "robosuite/controllers/parts/arm/osc.py"
                if size == 12
                else "robosuite/controllers/osc.py",
                "start": 1,
                "lines": 40,
            },
            "scratch_execute": {"code": "result = read('robot.eef_position', 1000)"},
            "finish": {},
            "interface_catalog": {},
            "enable_capability": {
                "name": "additional_proprioception",
                "reason": "Inspect additional robot encoders available in this fixture.",
            },
            "read_capability": {"name": "additional_proprioception"},
            "define_interface_tool": {
                "name": "eef_position_mm",
                "description": "Read end-effector position in millimetres, preserving its frame.",
                "code": "sample = read('robot.eef_position', 1000)\nresult = {'position_mm': [1000 * v for v in sample['value']], 'frame': sample['metadata']['frame'], 'simulator_step': sample['metadata']['simulator_step']}",
                "read_only": True,
                "reason": "Make a reusable sensor conversion for this episode.",
            },
            "call_interface_tool": {
                "name": "eef_position_mm",
                "arguments_json": "{}",
            },
            "probe_dynamics": {
                "action": action,
                "repeat_steps": 2,
                "channels": ["robot.eef_position"],
                "reason": "Measure a local response; this fixture has synthetic constant sensors.",
            },
            "revise_interface_notes": {
                "notes": "This documentation fixture has constant sensors; it is not a dynamics estimate.",
                "reason": "Keep measurement provenance explicit.",
            },
            "restore_interface": {
                "revision": 0,
                "reason": "Restore the empty interface.",
            },
        }
        tools = {arm: tools_for(arm, size) for arm in ("F", "B0", "B", "F_adaptive")}
        for arm_tools in tools.values():
            for tool in arm_tools:
                jsonschema.Draft202012Validator.check_schema(tool["parameters"])
                jsonschema.validate(calls[tool["name"]], tool["parameters"])
        read = proxy.read(**calls["read"])
        observe = proxy.observe(**calls["observe"])
        camera_channel = next(k for k in sensors if k.startswith("camera."))
        camera = proxy.read(camera_channel, 1000)
        accepted = proxy.act(**calls["act"])
        rejected = proxy.step(**calls["step"])
        assert accepted["accepted"] and accepted["applied_steps"] == 3
        assert rejected["rejection_reason"] == "stale_observation_step"
        assert proxy.step_count == 3
        # Exercise actual adaptive routing with synthetic sensors. No simulator,
        # inference call, generated program execution or benchmark trial occurs.
        with TemporaryDirectory(prefix="hardware-contract-example-") as directory:
            env.observation["robot0_joint_acc"] = np.zeros(7)
            events = ExampleEvents()
            events.directory = Path(directory)
            adaptive_proxy = Proxy(
                env,
                env.observation,
                episode_id="example-adaptive-episode",
                controller=controller,
                events=events,
                sensors=sensors,
                limits=Limits(wall_seconds=None),
                clock=lambda: 0.0,
            )
            router = Router(adaptive_proxy, None, "F_adaptive")
            adaptive_responses = {}
            for operation in (
                "interface_catalog",
                "enable_capability",
                "read_capability",
                "define_interface_tool",
                "probe_dynamics",
                "revise_interface_notes",
                "restore_interface",
            ):
                response = router.dispatch(operation, calls[operation])
                assert "error" not in response, (operation, response)
                adaptive_responses[operation] = response
            assert adaptive_proxy.step_count == 2
            assert router.evolution.state == {
                "capabilities": [],
                "tools": {},
                "notes": "",
            }
        result["implementations"][name] = {
            "action_dimension": size,
            "channels": [
                {
                    "channel": channel,
                    "native_key": key,
                    "units": unit,
                    "frame": frame,
                    "shape": shape or [128, 128, 3],
                }
                for channel, (key, unit, frame, shape) in sensors.items()
            ],
            "tool_schemas": tools,
            "requests": calls,
            "adaptive_responses": adaptive_responses,
            "responses": {
                "describe_device": description,
                "read": read,
                "observe": observe,
                "camera_read": camera,
                "act_or_step": accepted,
                "rejected_stale_step": rejected,
                "describe_visible": {
                    "frame_step": 0,
                    "visible_facts": ["Gripper is visible above the counter."],
                    "uncertainties": ["Contact cannot be confirmed from this image."],
                    "occlusions": [],
                },
            },
        }
    return result


def make_svg():
    s = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1440" viewBox="0 0 1800 1440" role="img" aria-labelledby="title desc">',
        '<title id="title">Astra hardware interface: JSON contracts and code architecture</title>',
        '<desc id="desc">Three actor interfaces, shared execution proxy, native hardware adapters, private evaluation and replay. Full JSON examples and sensor tables accompany this diagram in the standalone HTML.</desc>',
        '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto-start-reverse"><path d="M0,0 L9,4.5 L0,9" fill="#49627d"/></marker></defs>',
        '<rect width="1800" height="1440" fill="#f5f8fc"/>',
    ]

    def text(x, y, value, size=20, color=INK, weight=400, mono=False):
        face = (
            "ui-monospace, Menlo, monospace" if mono else "Arial, Helvetica, sans-serif"
        )
        s.append(
            f'<text x="{x}" y="{y}" font-family="{face}" font-size="{size}" font-weight="{weight}" fill="{color}">{html.escape(value)}</text>'
        )

    def box(x, y, w, h, title, lines, color=LINE, fill="white", title_color=INK):
        s.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="15" fill="{fill}" stroke="{color}" stroke-width="1.6"/>'
        )
        text(x + 20, y + 34, title, 23, title_color, 700)
        for index, line in enumerate(lines):
            text(x + 20, y + 67 + 27 * index, line, 18, MUTED)

    def arrow(path, label=None, x=0, y=0, both=False):
        start = ' marker-start="url(#arrow)"' if both else ""
        s.append(
            f'<path d="{path}" fill="none" stroke="{MUTED}" stroke-width="2.1" marker-end="url(#arrow)"{start}/>'
        )
        if label:
            text(x, y, label, 17, MUTED)

    text(48, 55, "Astra → hardware: interfaces, JSON and code", 36, weight=700)
    text(
        48,
        89,
        "Fresh actor per rollout • Fixed model weights • One selected interface • Same native robot authority",
        21,
        MUTED,
    )
    box(
        48,
        123,
        1704,
        103,
        "1  runner.run_trial + provider.Session",
        [
            "Official task instruction + allowed tools → model function call {name, arguments} → strict JSON parsing → Router.dispatch(tool, arguments)"
        ],
    )
    for x in (310, 898, 1488):
        arrow(f"M{x},226 V264")
    box(
        48,
        273,
        524,
        260,
        "F  Typed hardware interface",
        [
            "describe_device() → hardware-1 description",
            "read(channel, max_age_ms) / read_latest()",
            "act(envelope) → Proxy.act → Proxy.step",
            "Envelope: channel, value, duration_steps,",
            "metadata {mode, units, episode_id,",
            "observation_step}; units/frames declared.",
        ],
        F,
        "#eaf7f3",
        F,
    )
    box(
        636,
        273,
        524,
        260,
        "B0  Native codebase interface",
        [
            "source tools reveal native conventions",
            "observe(keys) → native sensor dictionary",
            "step(action, repeat_steps, observation_step)",
            "→ Proxy.step with the same vector as F",
            "JSON input schema fixes action dimension;",
            "the actor discovers ordering in source.",
        ],
        B0,
        "#fff5ea",
        B0,
    )
    box(
        1224,
        273,
        528,
        260,
        "B  Native interface + observer",
        [
            "Same observe / step tools as B0",
            "describe_visible(request_kind)",
            "→ current camera images → Observer",
            "JSON: frame_step, visible_facts,",
            "uncertainties, occlusions",
            "Observer has no action tools or task plans.",
        ],
        B,
        "#f4eefb",
        B,
    )
    box(
        48,
        555,
        1704,
        103,
        "Available in every arm: source_search • source_read • scratch_execute • finish",
        [
            "SourceView exposes audited robot/controller code. Isolated Python scratch can call only that arm’s proxy tools. No env handle or network."
        ],
    )
    arrow("M310,533 V544 H22 V701 H68")
    arrow("M898,533 V544 H1790 V701 H1732")
    s.append(
        f'<path d="M1488,533 V544 H1790" fill="none" stroke="{MUTED}" stroke-width="2.1"/>'
    )
    box(
        78,
        684,
        1654,
        155,
        "2  Router → shared Proxy: validate, execute, record",
        [
            "runner.py: JSON Schema + allowed tool → proxy.py: finite vector, exact 7D/12D shape, bounds, repeat ≤ 10, current observation step",
            "F adds channel / mode / units / episode checks. Rejected commands apply no action. All arms execute the same normalized native controller.",
            "Return {accepted, applied_action, applied_steps, simulator_step, timestamp, episode_ended}; read again before the next action.",
        ],
    )
    arrow("M468,839 V872", both=True)
    text(240, 862, "native actions ↓", 17, MUTED)
    text(525, 862, "images / sensors ↑", 17, MUTED)
    arrow("M1332,839 V872", both=True)
    text(1104, 862, "native actions ↓", 17, MUTED)
    text(1389, 862, "images / sensors ↑", 17, MUTED)
    box(
        48,
        879,
        816,
        196,
        "3a  libero.Environment — fixed-base Panda",
        [
            "7 values: Δx, Δy, Δz, axis-angle Δrotation (3), gripper",
            "OSC_POSE; pose commands in world frame; 20 Hz",
            "6 numeric channels: joints, end effector, gripper",
            "2 RGB cameras: front + wrist; upright PNG, 128 × 128",
            "Readback: typed envelope (F) or native keys (B0/B)",
        ],
    )
    box(
        936,
        879,
        816,
        196,
        "3b  robocasa.Environment — mobile PandaOmron",
        [
            "12 values: arm (6), gripper, base (3), torso, base mode",
            "HYBRID_MOBILE_BASE; native arm controller uses base frame",
            "5 numeric channels: base, end effector, gripper",
            "3 RGB cameras: left + right + wrist; upright PNG, 128 × 128",
            "Live controller resolves exact slices / scaling / control frequency",
        ],
    )
    arrow("M456,1075 V1111")
    s.append(
        f'<path d="M1344,1075 V1095 H456" fill="none" stroke="{MUTED}" stroke-width="2.1"/>'
    )
    box(
        48,
        1118,
        816,
        180,
        "4  Private evaluator + common.Events",
        [
            "Official success predicate after each applied native step",
            "Stop at first success, horizon, finish, or infrastructure error",
            "events.jsonl: chained hashes; actions, observations, model calls",
            "outcome.json: success, reason, steps, time, actor/observer usage",
        ],
    )
    box(
        936,
        1118,
        816,
        180,
        "5  Worker replay + analysis",
        [
            "Separate process: same pinned reset + recorded native actions",
            "Compare first-success step and terminal simulator state",
            "independent-replay.json → verified evidence + analysis.json",
            "No model calls during replay; it verifies live control.",
        ],
    )
    arrow("M864,1209 H927", "logs", 883, 1188)
    text(48, 1343, "IMPLEMENTED BOUNDARY", 17, MUTED, 700)
    text(
        48,
        1372,
        "Actor sees robot sensors, camera images and audited source. Task internals, object state and evaluator labels stay private.",
        21,
    )
    text(
        48,
        1409,
        "See standalone HTML for every channel, exact request / response JSON, complete generated tool schemas and module ownership.",
        18,
        MUTED,
    )
    s.append("</svg>")
    return "\n".join(s)


def table(headers, rows):
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join(f"<th>{html.escape(v)}</th>" for v in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def code(title, value, note="", opened=False):
    data = pretty(value) if not isinstance(value, str) else value
    return f'<details {"open" if opened else ""}><summary>{html.escape(title)}</summary><p>{html.escape(note)}</p><pre><code>{html.escape(data)}</code></pre></details>'


def make_html(svg, data):
    implementations = data["implementations"]
    shared = [
        (
            "source_search",
            "query",
            "Literal substring search in the shared source allowlist.",
        ),
        (
            "source_read",
            "path, start, lines",
            "Allowlisted relative file; 1-based start; at most 200 lines.",
        ),
        (
            "scratch_execute",
            "code",
            "Standard-library Python; set result to JSON. Arm-specific robot calls route back to Router.",
        ),
        (
            "finish",
            "{}",
            "Ends the rollout as TASK_FAILURE if official success has not already ended it. The actor cannot self-award success.",
        ),
    ]
    modules = [
        (
            "protocol.py / robocasa_protocol.py",
            "Experiment contract",
            "Limits, fixed task/scenario schedule, condition pairing, protocol validation.",
        ),
        (
            "osmo_worker.py + pilot_worker.py / robocasa_worker.py",
            "Trial orchestration",
            "Prepare pinned runtime, commission hardware, reset each trial, call run_trial, replay and archive receipts.",
        ),
        (
            "runner.py · tools_for",
            "Tool definitions",
            "Strict JSON Schemas per arm; action vector dimension comes from Proxy.action_dim.",
        ),
        (
            "runner.py · run_trial / Router",
            "Live control loop",
            "Fresh actor, parse calls, validate JSON, dispatch tools, return observations and record outcomes.",
        ),
        (
            "adaptive.py · AdaptiveInterface",
            "F-adaptive interface evolution",
            "Per-episode capabilities, versioned Python tools, measured probes, notes and restore; every action uses the shared proxy.",
        ),
        (
            "provider.py · Session / Meter / Observer",
            "Inference and usage",
            "NVIDIA Responses transport, actor and observer histories, PNG image transport, token and latency accounting.",
        ),
        (
            "proxy.py · Proxy / validate_action",
            "Only action execution boundary",
            "Map typed/native observations; reject invalid/stale commands; repeat native steps; check private success after every step.",
        ),
        (
            "schema.py + hardware.schema.json",
            "Device-description validation",
            "JSON structure and semantic checks for channel shape, frequency, bounds and safe range.",
        ),
        (
            "libero.py / robocasa.py · Environment",
            "Benchmark adapters",
            "Native reset, sensor readback, controller metadata, action step, private success and terminal snapshot.",
        ),
        (
            "sources.py · SourceView / build_view",
            "Read-only robot source",
            "Positive file/projection allowlist and SHA-256 verification. RoboCasa has its own build_view in robocasa.py.",
        ),
        (
            "isolation.py + scratch_worker.py",
            "Scratch execution",
            "Chroot, dropped privileges, seccomp, read-only /sources, writable /scratch; nested calls route through Router.",
        ),
        (
            "common.py · Events",
            "Evidence",
            "Append-only events.jsonl with sequence, previous_sha256 and current sha256.",
        ),
        (
            "analysis.py / reporting.py",
            "Results",
            "Success, termination reasons and usage. RoboCasa paired_success_times compares jointly successful verified starts.",
        ),
    ]
    hardware = []
    for name, item in implementations.items():
        hardware.append(f'<h2>{name}: {item["action_dimension"]}-value controller</h2>')
        hardware.append(
            table(
                [
                    "F / F-adaptive channel",
                    "B0 / B native key",
                    "Shape",
                    "Units",
                    "Reference frame",
                ],
                [
                    (
                        r["channel"],
                        r["native_key"],
                        " × ".join(map(str, r["shape"])),
                        r["units"],
                        r["frame"],
                    )
                    for r in item["channels"]
                ],
            )
        )
    hardware.append(
        '<div class="note"><strong>Frames are part of the contract.</strong> LIBERO robot.eef_position is world-referenced; RoboCasa uses mobile-base coordinates. F and F-adaptive receive the frame explicitly; B0/B discover the native convention in audited source. F-adaptive can separately enable available extra encoders or camera calibration through its capability catalog.</div>'
    )
    hardware.append(
        table(
            ["Hardware part", "LIBERO", "RoboCasa"],
            [
                (
                    "Arm",
                    "6 OSC_POSE deltas; world reference frame",
                    "6 OSC_POSE deltas; controller base reference frame",
                ),
                (
                    "Gripper",
                    "1 normalized integrated command; − opens / + closes",
                    "1 normalized integrated command; − opens / + closes",
                ),
                ("Mobile base", "Absent", "3 normalized velocity values: x, y, yaw"),
                ("Torso", "Absent", "1 normalized vertical joint-position delta"),
                (
                    "Base mode",
                    "Absent",
                    "1 mode value: +1 tracks desired arm goal; −1 updates from achieved pose",
                ),
                (
                    "Controller metadata",
                    "20 Hz; live bounds, pose scaling and controller version",
                    "Live action_order / parts resolve slice indices, scaling and control frequency",
                ),
                (
                    "Execution",
                    "Same 7-value vector in F/B0/B",
                    "Same 12-value vector in F/B0/B; copy on every step prevents native in-place mutation",
                ),
            ],
        )
    )
    api = [
        '<div class="note"><strong>These are real wire contracts with synthetic example values.</strong> Tool names are shown as headings; the request boxes contain the JSON arguments. Images, coordinates, timestamps and outcomes here are illustrative, not benchmark results. Full JSON Schema is generated directly by runner.tools_for.</div>'
    ]
    api.append(
        table(
            ["Arm", "Hardware tools", "Additional context"],
            [
                (
                    "F",
                    "describe_device, read, read_latest, act",
                    "Device description is also included in the initial system prompt.",
                ),
                (
                    "F-adaptive",
                    "F tools + interface_catalog, enable/read_capability, define/call_interface_tool, probe_dynamics, revise_interface_notes, restore_interface",
                    "Versioned per-episode tools and optional extra hardware information; native execution remains shared.",
                ),
                (
                    "B0",
                    "observe, step",
                    "Native controller and observation conventions via source tools.",
                ),
                (
                    "B",
                    "observe, step, describe_visible",
                    "Same native tools; separate stateless image-only observer.",
                ),
            ],
        )
    )
    for name, item in implementations.items():
        api.append(f"<h2>{name} JSON examples</h2>")
        req, res = item["requests"], item["responses"]
        for tool in (
            "describe_device",
            "read",
            "act",
            "observe",
            "step",
            "describe_visible",
        ):
            api.append(
                code(
                    f"{tool} — request arguments",
                    req[tool],
                    opened=(name == "LIBERO" and tool in ("act", "step")),
                )
            )
            key = "act_or_step" if tool in ("act", "step") else tool
            api.append(
                code(
                    f"{tool} — response",
                    res[key],
                    "F act and native step produce the same execution receipt."
                    if tool in ("act", "step")
                    else "",
                )
            )
        api.append(
            code(
                "Camera read — PNG envelope",
                res["camera_read"],
                "provider.observation_content replaces the base64 payload in the text with an image reference and sends the actual PNG as input_image to the model.",
            )
        )
        api.append(
            code(
                "Rejected action — stale observation",
                res["rejected_stale_step"],
                "After three steps, resubmitting observation_step=0 is rejected; no extra native step is executed.",
            )
        )
        for arm, definitions in item["tool_schemas"].items():
            api.append(code(f"Complete {name} / {arm} tool schemas", definitions))
    api.append(
        "<h2>Shared tools</h2>" + table(["Tool", "Arguments", "Behavior"], shared)
    )
    api.append(
        code(
            "scratch_execute — F example",
            implementations["LIBERO"]["requests"]["scratch_execute"],
        )
    )
    api.append(
        code(
            "scratch_execute — B0/B example",
            {"code": "result = observe(['robot0_eef_pos'])"},
        )
    )
    api.append(
        "<p>Scratch exposes read/act in F and observe/step in B0/B. Each nested hardware call is validated and logged through Router. The observer receives current camera images plus the requested evidence category, with no task instruction, source tools or action tools.</p>"
    )
    trace = [
        (
            "01",
            "Prepare",
            "Worker loads frozen task / seed / arm, verifies source and runtime, resets the native environment and records its controller and initial-state identity.",
        ),
        (
            "02",
            "Discover",
            "F receives hardware-1 JSON; describe_device can refresh it. B0/B inspect audited source to discover native keys, action order, scaling and frames.",
        ),
        (
            "03",
            "Sense",
            "F read/read_latest or B0/B observe samples paused current sensors. JSON carries the current simulator step. PNG images become model image inputs.",
        ),
        (
            "04",
            "Reason",
            "provider.Session sends the task, history, tools and observations to Astra. Physics is paused. Weights stay fixed; scratch is optional.",
        ),
        (
            "05",
            "Validate",
            "Router validates allowed tools and strict JSON; Proxy validates finite numbers, exact vector length, controller bounds, repeat count and freshness. F also checks channel, mode, units and episode.",
        ),
        (
            "06",
            "Execute",
            "Proxy.step applies the same native vector for the requested repeats, at most ten. The adapter advances physics; private official success is checked after each native step. First success stops the remaining repeats.",
        ),
        (
            "07",
            "Observe again",
            "Actor receives an execution receipt, never reward/done/info or the private success label. It reads current sensors and submits a fresh observation_step for the next action. episode_ended signals termination.",
        ),
        (
            "08",
            "Verify",
            "Worker records outcome and terminal state. Independent replay reconstructs the reset and saved actions in another process, compares first success and terminal state, then archives evidence. No model calls in replay.",
        ),
    ]
    trace_html = "".join(
        f'<article class="trace"><span>{n}</span><div><h3>{title}</h3><p>{text}</p></div></article>'
        for n, title, text in trace
    )
    artifacts = table(
        ["Artifact / representation", "Purpose", "Actor can read?"],
        [
            (
                "preregistration.yaml → resolved manifest JSON",
                "Frozen design, resolved source / controller / scenario receipts",
                "Only permitted prompt settings, task instruction and F device description",
            ),
            (
                "hardware-1 device JSON",
                "Observation channel + action specification",
                "F; B0/B use native source discovery",
            ),
            (
                "Tool request / response JSON",
                "Read sensors, issue commands, receive validation and execution receipts",
                "Yes, for tools available to the selected arm",
            ),
            (
                "PNG image payload",
                "Upright RGB camera observation",
                "Yes, identical camera authority in all arms",
            ),
            (
                "allowlist.json",
                "Allowed source projections and their content hashes",
                "Source access restricted to those entries",
            ),
            (
                "events.jsonl",
                "Chained private trace, including actions and evaluator results",
                "No direct file access",
            ),
            (
                "outcome.json + terminal state",
                "Scored success, termination reason, usage and state for audit",
                "No",
            ),
            (
                "independent-replay.json",
                "Audit verdict: same reset, actions, success step, terminal state",
                "No",
            ),
            (
                "analysis.json",
                "Aggregate results, paired comparisons and limitations",
                "No",
            ),
        ],
    )
    template = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Astra hardware interfaces · code and JSON</title>
<style>
:root{font-family:Inter,Arial,sans-serif;color:#142d46;background:#f5f8fc;font-size:16px;line-height:1.55}*{box-sizing:border-box}body{margin:0}header,main,footer{max-width:1540px;margin:auto;padding:28px 36px}header{padding-bottom:10px}h1{font-size:clamp(28px,3vw,42px);letter-spacing:-1px;line-height:1.15;margin:8px 0 12px}h2{font-size:25px;margin-top:32px}h3{margin:0 0 5px}p{color:#49627d}.eyebrow{font-size:12px;letter-spacing:2px;font-weight:700;color:#087f70}.subtitle{max-width:1040px}.buttons,nav{display:flex;gap:9px;flex-wrap:wrap}button{background:white;color:#142d46;border:1px solid #ccd9e6;border-radius:8px;padding:10px 16px;font:inherit;font-size:14px;cursor:pointer}button:hover,button:focus-visible{border-color:#087f70;outline:2px solid #bddfd7}nav{margin-top:22px;border-bottom:1px solid #ccd9e6;padding-bottom:12px}nav button[aria-selected=true]{background:#142d46;color:white;border-color:#142d46}main{padding-top:8px}.panel{display:none}.panel.active{display:block}.diagram{overflow-x:auto;border:1px solid #dce5ee;border-radius:12px;background:white}.diagram svg{display:block;width:100%;height:auto;min-width:900px}.note{background:#eaf3fb;border-left:4px solid #5e86a9;border-radius:5px;padding:16px 20px;margin:20px 0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card{padding:18px 22px;border:1px solid #ccd9e6;border-radius:12px;background:white}.card p{margin-bottom:0}.f{border-top:4px solid #087f70}.b0{border-top:4px solid #ab5e19}.b{border-top:4px solid #7750b3}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{text-align:left;border-bottom:1px solid #dce5ee;padding:13px 15px;vertical-align:top}th{background:#eaf0f7}td:first-child{font-weight:600}details{margin:12px 0;border:1px solid #ccd9e6;background:white;border-radius:10px;overflow:hidden}summary{font-weight:600;cursor:pointer;padding:14px 18px}details p{padding:0 18px;margin:0 0 10px;font-size:14px}pre{background:#12283c;color:#eef6ff;margin:0;padding:18px;overflow:auto;font-size:13px;line-height:1.55}code{font-family:ui-monospace,Menlo,monospace}.trace{display:flex;gap:20px;padding:21px 0;border-bottom:1px solid #ccd9e6;max-width:1120px}.trace span{font-size:28px;font-weight:700;color:#087f70}.trace p{margin:0}footer{font-size:13px;color:#49627d}.boundary{background:#fff5ea;padding:18px;border-radius:10px;margin-top:20px}a{color:#087f70}@media(max-width:780px){header,main,footer{padding:20px 16px}.cards{grid-template-columns:1fr}.diagram svg{min-width:1100px}}@media print{nav,.buttons{display:none}.panel{display:block!important;break-before:page}.panel:first-child{break-before:auto}.diagram svg{min-width:0}details{break-inside:avoid}.note{print-color-adjust:exact}}
</style></head><body><header><div class="eyebrow">ASTRA · DIRECT ROBOT CONTROL · IMPLEMENTATION GUIDE</div><h1>Hardware interfaces, JSON contracts and code design</h1><p class="subtitle">How the same Astra actor controls LIBERO’s fixed-base Panda and RoboCasa’s mobile PandaOmron through F, B0 and B. The device abstraction describes the native controller; it does not convert an abstract task into a motion plan.</p><div class="buttons"><button id="download-svg">Download architecture SVG</button><button id="download-json">Download contracts JSON</button><button onclick="window.print()">Print / save PDF</button></div><nav role="tablist" aria-label="Framework views"><button role="tab" id="tab-overview" data-target="overview" aria-controls="overview" aria-selected="true">Architecture diagram</button><button role="tab" id="tab-hardware" data-target="hardware" aria-controls="hardware" aria-selected="false">All hardware channels</button><button role="tab" id="tab-json" data-target="json" aria-controls="json" aria-selected="false">JSON contracts</button><button role="tab" id="tab-code" data-target="code" aria-controls="code" aria-selected="false">Code design</button><button role="tab" id="tab-trace" data-target="trace" aria-controls="trace" aria-selected="false">One control cycle</button></nav></header><main>
<section class="panel active" id="overview" role="tabpanel" aria-labelledby="tab-overview"><div class="diagram">@@SVG@@</div><div class="cards"><article class="card f"><h3>F · typed interface</h3><p>Explicit channel semantics, units, frames, bounds and command metadata. The model still chooses every native action.</p></article><article class="card b0"><h3>B0 · native tools</h3><p>Observe native keys and submit the native controller vector. The model discovers the conventions through shared audited source.</p></article><article class="card b"><h3>B · native + observer</h3><p>B0’s interface plus a separate stateless Astra observer. Its current-image descriptions contain evidence and uncertainty, with no action tools.</p></article></div><div class="note">The framework has a working LIBERO pilot. RoboCasa is an implemented adapter awaiting native commissioning; this architecture diagram makes no claim that the RoboCasa benchmark has passed. Results and workflow status are reported separately.</div></section>
<section class="panel" id="hardware" role="tabpanel" aria-labelledby="tab-hardware">@@HARDWARE@@</section>
<section class="panel" id="json" role="tabpanel" aria-labelledby="tab-json">@@JSON@@</section>
<section class="panel" id="code" role="tabpanel" aria-labelledby="tab-code"><h2>Module ownership</h2><p>Paths below are under <code>astra_reversal/hardware_interface/</code>, except experiment schemas, configuration and launcher scripts. A single shared actor/router/proxy layer sits above benchmark-specific adapters.</p>@@MODULES@@<h2>JSON and artifact boundaries</h2>@@ARTIFACTS@@<div class="boundary"><strong>Read-only actor boundary:</strong> no direct simulator handle, hidden object poses, task/reward implementations, evaluator success labels, other rollouts or credentials. The observer sees current images, not task state. Private logs can contain model text and are not part of this shareable document.</div></section>
<section class="panel" id="trace" role="tabpanel" aria-labelledby="tab-trace"><h2>One live control cycle</h2>@@TRACE@@<div class="note">Success is checked inside each repeated command. For example, a requested 10-step action can return applied_steps=4 when success occurs on its fourth native step. F and B0/B share this exact behavior.</div></section></main><footer>Generated from implemented contracts; examples validated against the tool schemas and the hardware description schema. Synthetic values are labeled. This HTML includes its SVG, JSON, styling and navigation and works offline. Code source hashes are included in the contracts download.</footer><script id="contracts-data" type="application/json">@@DATA@@</script><script>
function activate(id){document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id===id));document.querySelectorAll('nav button').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.target===id)));history.replaceState(null,'','#'+id)}document.querySelectorAll('nav button').forEach(b=>b.addEventListener('click',()=>activate(b.dataset.target)));const wanted=location.hash.slice(1);if(document.getElementById('tab-'+wanted))activate(wanted);document.querySelector('nav').addEventListener('keydown',e=>{if(!['ArrowRight','ArrowLeft','Home','End'].includes(e.key))return;e.preventDefault();const bs=[...document.querySelectorAll('nav button')];let i=bs.indexOf(document.activeElement);i=e.key==='Home'?0:e.key==='End'?bs.length-1:(i+(e.key==='ArrowRight'?1:-1)+bs.length)%bs.length;bs[i].focus();activate(bs[i].dataset.target)});function download(name,type,content){const url=URL.createObjectURL(new Blob([content],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}document.getElementById('download-svg').onclick=()=>download('astra-hardware-framework.svg','image/svg+xml',document.querySelector('.diagram svg').outerHTML);document.getElementById('download-json').onclick=()=>download('astra-hardware-contracts.json','application/json',document.getElementById('contracts-data').textContent);window.addEventListener('beforeprint',()=>document.querySelectorAll('details').forEach(d=>{d.dataset.wasOpen=d.open;d.open=true}));window.addEventListener('afterprint',()=>document.querySelectorAll('details').forEach(d=>d.open=d.dataset.wasOpen==='true'));
</script></body></html>"""
    for marker, content in {
        "SVG": svg,
        "HARDWARE": "".join(hardware),
        "JSON": "".join(api),
        "MODULES": table(["Module", "Responsibility", "Implemented behavior"], modules),
        "ARTIFACTS": artifacts,
        "TRACE": trace_html,
        "DATA": pretty(data).replace("<", "\\u003c"),
    }.items():
        template = template.replace("@@" + marker + "@@", content)
    assert "@@" not in template
    return template


def enhanced_html(execution_svg, overview_svg, adaptive_svg, data):
    """Keep the complete original execution diagram beside the adaptive view."""
    document = make_html(overview_svg, data)
    document = (
        document.replace(
            "Hardware interfaces, JSON contracts and code design</h1>",
            "Agent-legible hardware</h1>",
        )
        .replace(
            "through F, B0 and B.",
            "through F, B0, B and F-adaptive.",
        )
        .replace(
            "grid-template-columns:repeat(3,1fr)",
            "grid-template-columns:repeat(4,minmax(0,1fr))",
        )
    )
    old_status = "The framework has a working LIBERO pilot. RoboCasa is an implemented adapter awaiting native commissioning; this architecture diagram makes no claim that the RoboCasa benchmark has passed. Results and workflow status are reported separately."
    document = document.replace(
        old_status,
        "These diagrams describe the implemented interfaces. Benchmark results are reported separately. All four actors can receive images directly. F-adaptive combines tool evolution with optional extra hardware information; this comparison does not isolate those two effects.",
    )
    document = document.replace(
        '<div class="buttons"><button id="download-svg">',
        '<p><a href="harness-example.html">Recorded F / F-adaptive before-and-after example: exact code, revisions and videos</a></p><div class="buttons"><button id="download-svg">',
        1,
    )
    document = document.replace(
        '<article class="card b0">',
        '<article class="card" style="border-top:4px solid #235eb0"><h3>F-adaptive · evolving interface</h3><p>F plus tools Astra can write and revise, optional hardware information, and measured calibration. Every episode starts fresh.</p></article><article class="card b0">',
        1,
    )
    document = document.replace(
        "<h2>Module ownership</h2>",
        '<h2>Original F / B0 / B execution paths</h2><div class="diagram">'
        + execution_svg.replace('id="title"', 'id="execution-title"')
        .replace('id="desc"', 'id="execution-desc"')
        .replace(
            'aria-labelledby="title desc"',
            'aria-labelledby="execution-title execution-desc"',
        )
        .replace('id="arrow"', 'id="execution-arrow"')
        .replace("url(#arrow)", "url(#execution-arrow)")
        + "</div><p>F-adaptive adds the versioned interface layer shown in its dedicated view, above the same execution proxy.</p><h2>Module ownership</h2>",
        1,
    )
    adaptive_svg = (
        adaptive_svg.replace('id="title"', 'id="adaptive-title"')
        .replace('id="description"', 'id="adaptive-description"')
        .replace(
            'aria-labelledby="title description"',
            'aria-labelledby="adaptive-title adaptive-description"',
        )
        .replace('id="arrow"', 'id="adaptive-arrow"')
        .replace("url(#arrow)", "url(#adaptive-arrow)")
    )
    adaptive_content = [
        "<h2>How Astra evolves its harness</h2><p>The editable harness is a versioned layer of tools, enabled capabilities and notes above the native robot API. The model chooses how to use it during the episode.</p>",
        '<p><a href="harness-example.html">Inspect a real rollout: camera calibration, new control tools, and a triangulation-code revision at native step 211.</a></p>',
        "<p>The top-level model tool schemas stay fixed. Newly written programs live in an episode-local registry and are invoked by name through <code>call_interface_tool</code>; the model receives their descriptions, version hashes and current interface notes.</p>",
        '<div class="buttons"><button id="download-adaptive">Download adaptive SVG</button></div>',
        '<div class="diagram">' + adaptive_svg + "</div>",
        table(
            ["Astra can change", "Implementation"],
            [
                (
                    "Available hardware information",
                    "Discover and enable allowlisted additional encoders or camera calibration, when the adapter supplies them.",
                ),
                (
                    "Python interface tools",
                    "Define or replace a named program; call it with JSON parameters in isolated scratch. A read_only program cannot dispatch act.",
                ),
                (
                    "Measured response model or operating notes",
                    "Probe a bounded native command, inspect before/after readings, and retain a hypothesis in per-episode notes.",
                ),
                (
                    "Interface version",
                    "Restore earlier capabilities, tool definitions and notes. Robot motion and consumed steps remain.",
                ),
            ],
        ),
        "<p><strong>Fixed underneath:</strong> model weights, native controller, execution checks and private success evaluator. No tool or learned notes transfer into the next trial. Probe actions and control tools use the same episode horizon.</p>",
        '<div class="note">Synthetic examples below use the actual Router and adaptive implementation with fixture sensors. They are schema-validated documentation, not recorded Astra behavior or benchmark outcomes. The named tool is defined but is not executed by this documentation generator.</div>',
    ]
    for name, item in data["implementations"].items():
        adaptive_content.append(f"<h2>{name}: example interface revision</h2>")
        for operation in (
            "interface_catalog",
            "enable_capability",
            "read_capability",
            "define_interface_tool",
            "call_interface_tool",
            "probe_dynamics",
            "revise_interface_notes",
            "restore_interface",
        ):
            adaptive_content.append(
                code(
                    operation + " — request arguments",
                    item["requests"][operation],
                    opened=name == "LIBERO" and operation == "define_interface_tool",
                )
            )
            if operation in item["adaptive_responses"]:
                adaptive_content.append(
                    code(
                        operation + " — synthetic response",
                        item["adaptive_responses"][operation],
                    )
                )
    commit = data["runtime_source_commit"]
    base = "https://github.com/GaTech-RL2/EgoVerse/blob/" + commit + "/"
    links = []
    for path in data["code_sha256"]:
        links.append(
            '<li><a href="'
            + html.escape(base + path)
            + '">'
            + html.escape(path)
            + "</a></li>"
        )
    adaptive_content.append(
        "<h2>Implementing code</h2><p>Runtime source snapshot <code>"
        + commit
        + "</code>. Exact file hashes are in the contracts JSON.</p><ul>"
        + "".join(links)
        + "</ul>"
    )
    document = (
        document.replace(
            "</nav></header>",
            '<button role="tab" id="tab-adaptive" data-target="adaptive" aria-controls="adaptive" aria-selected="false">Harness evolution</button></nav></header>',
            1,
        )
        .replace(
            "</main><footer>",
            '<section class="panel" id="adaptive" role="tabpanel" aria-labelledby="tab-adaptive">'
            + "".join(adaptive_content)
            + "</section></main><footer>",
            1,
        )
        .replace(
            "document.getElementById('download-json').onclick=",
            "document.getElementById('download-adaptive').onclick=()=>download('adaptive-interface.svg','image/svg+xml',document.querySelector('#adaptive svg').outerHTML);document.getElementById('download-json').onclick=",
            1,
        )
    )
    return document


def main():
    import hashlib
    import subprocess

    OUT.mkdir(exist_ok=True, parents=True)
    with patch("astra_reversal.hardware_interface.proxy.utc", return_value=STAMP):
        data = contracts()
    files = [
        "astra_reversal/hardware_interface/runner.py",
        "astra_reversal/hardware_interface/proxy.py",
        "astra_reversal/hardware_interface/schema.py",
        "astra_reversal/hardware_interface/robocasa.py",
        "astra_reversal/hardware_interface/libero.py",
        "astra_reversal/hardware_interface/provider.py",
        "astra_reversal/hardware_interface/adaptive.py",
        "experiments/libero_hardware_interface/schemas/hardware.schema.json",
    ]
    data["code_sha256"] = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files
    }
    data["runtime_source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    svg = make_svg()
    overview_svg, adaptive_svg = overview_diagram(), adaptive_diagram()
    (OUT / "framework.svg").write_text(svg)
    (OUT / "agent-legible-hardware.svg").write_text(overview_svg)
    (OUT / "adaptive-interface.svg").write_text(adaptive_svg)
    dashboard_assets = (
        ROOT / "experiments/libero_hardware_interface/tools/process_dashboard"
    )
    (dashboard_assets / "agent-legible-hardware.svg").write_text(overview_svg)
    (dashboard_assets / "adaptive-interface.svg").write_text(adaptive_svg)
    (OUT / "framework.html").write_text(
        enhanced_html(svg, overview_svg, adaptive_svg, data)
    )
    (OUT / "framework-contracts.json").write_text(pretty(data) + "\n")
    (OUT / "framework.mmd").write_text("""flowchart TD
  task[Official task + frozen task / seed / arm] --> run[runner.run_trial: fresh actor]
  run <--> session[provider.Session: Astra Responses API]
  session -->|function call: name + arguments JSON| router[Router.dispatch + strict JSON Schema]
  router --> F[F: describe_device / read / read_latest / act envelope]
  router --> FA[F-adaptive: F tools plus versioned interface evolution]
  FA <--> evolve[adaptive.py: capabilities / tools / probes / notes / restore]
  evolve -->|sandboxed read and act| proxy
  router --> B0[B0: observe native keys / step native vector]
  router --> B[B: B0 tools + describe_visible]
  B <--> obs[provider.Observer: current images → evidence JSON]
  router <--> src[SourceView: audited robot files]
  router <--> scratch[Isolated scratch Python → same Router]
  F -->|Proxy.act metadata checks| proxy[Proxy.step: dimensions / bounds / repeats / freshness]
  B0 --> proxy
  B --> proxy
  proxy <-->|7D actions / allowed sensors| lib[libero.Environment: fixed Panda]
  proxy <-->|12D actions / allowed sensors| casa[robocasa.Environment: mobile PandaOmron]
  proxy --> eval[Private success after each native step]
  proxy --> events[common.Events: hashed events.jsonl]
  eval --> outcome[outcome.json + terminal state]
  events --> replay[Worker: independent action replay]
  outcome --> replay
  replay --> analysis[independent-replay.json → analysis.json]
""")
    print(
        pretty(
            {
                "status": "generated_and_schema_validated",
                "files": [
                    str(OUT / n)
                    for n in (
                        "framework.svg",
                        "framework.html",
                        "framework-contracts.json",
                        "framework.mmd",
                        "agent-legible-hardware.svg",
                        "adaptive-interface.svg",
                    )
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
