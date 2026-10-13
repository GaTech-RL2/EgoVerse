"""Editable SVG figures for the implemented agent-legible hardware interfaces."""

import html


class Figure:
    def __init__(self, title, description, height):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="{height}" viewBox="0 0 1600 {height}" role="img" aria-labelledby="title description">',
            f'<title id="title">{html.escape(title)}</title>',
            f'<desc id="description">{html.escape(description)}</desc>',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#63788b"/></marker></defs>',
            f'<rect width="1600" height="{height}" fill="#f6f8fb"/>',
        ]

    def text(self, x, y, value, size=20, color="#203649", weight=400):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{html.escape(value)}</text>'
        )

    def box(self, x, y, w, h, title, lines, accent="#d2dde6", fill="white"):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{accent}" stroke-width="1.5"/>'
        )
        self.text(x + 24, y + 39, title, 23, weight=700)
        for i, line in enumerate(lines):
            self.text(x + 24, y + 77 + 29 * i, line, 18, "#4b657a")

    def arrow(self, path, dashed=False):
        dash = ' stroke-dasharray="7 6"' if dashed else ""
        self.parts.append(
            f'<path d="{path}" fill="none" stroke="#63788b" stroke-width="2.5" marker-end="url(#arrow)"{dash}/>'
        )

    def finish(self):
        return "\n".join(self.parts + ["</svg>"])


def overview():
    f = Figure(
        "Agent-legible hardware",
        "The typed F interface gives Astra explicit sensor and command semantics. Observations go from the robot through a device contract to Astra; commands return through validation to the native controller. Four experimental interfaces and two benchmark adapters are shown.",
        1080,
    )
    f.text(56, 52, "AGENT-LEGIBLE HARDWARE", 15, "#087f70", 700)
    f.text(56, 103, "A robot interface an agent can inspect and use", 37, weight=700)
    f.text(
        56,
        141,
        "Typed interface shown: F, and the starting interface for F-adaptive. Astra chooses the robot actions.",
        21,
        "#4b657a",
    )
    f.box(
        56,
        202,
        410,
        210,
        "Robot observations",
        [
            "RGB cameras + robot proprioception",
            "LIBERO: front + wrist cameras",
            "RoboCasa: left + right + wrist",
            "Joint, end-effector and gripper state",
        ],
    )
    f.box(
        574,
        202,
        458,
        210,
        "Discoverable device contract",
        [
            "describe_device() → hardware-1 JSON",
            "Channel name, shape, units and frame",
            "Action bounds and controller convention",
            "read() → value + step + timestamp",
        ],
        "#087f70",
        "#eaf7f3",
    )
    f.box(
        1140,
        202,
        404,
        210,
        "Astra actor",
        [
            "Task + tools + current observations",
            "Camera PNGs become image inputs",
            "Inspect → choose → act → observe",
            "Fresh session; fixed model weights",
        ],
    )
    f.arrow("M467 305H564")
    f.text(483, 279, "sensors", 17, "#4b657a")
    f.arrow("M1033 305H1130")
    f.text(1048, 279, "readback", 17, "#4b657a")
    f.box(
        1140,
        475,
        404,
        174,
        "Action request JSON",
        [
            "act({envelope: …})",
            "Channel, native vector, duration",
            "Episode + observation-step metadata",
        ],
    )
    f.box(
        574,
        475,
        458,
        174,
        "Validate, execute, return a receipt",
        [
            "Router → Proxy → native controller",
            "Shape, bounds, repeats and freshness",
            "Read again after accepted execution",
        ],
        "#087f70",
        "#eaf7f3",
    )
    f.box(
        56,
        475,
        410,
        174,
        "Native robot action",
        [
            "LIBERO: 7-value OSC_POSE command",
            "RoboCasa: 12-value mobile command",
            "Physics advances on accepted steps",
        ],
    )
    f.arrow("M1342 413V465")
    f.arrow("M1139 555H1042")
    f.arrow("M573 555H476")
    f.arrow("M260 474V422")
    f.text(
        56,
        712,
        "FOUR INTERFACES TO THE SAME NATIVE CONTROL AUTHORITY",
        15,
        "#4b657a",
        700,
    )
    cards = [
        (
            56,
            "F · typed hardware",
            [
                "Device semantics are explicit.",
                "read / read_latest / act",
                "Astra still plans every action.",
            ],
            "#087f70",
            "#eaf7f3",
        ),
        (
            436,
            "B0 · source discovery",
            [
                "Astra inspects robot source.",
                "observe / step",
                "Discovers native conventions.",
            ],
            "#ab5e19",
            "#fff5ea",
        ),
        (
            816,
            "B · source + observer",
            [
                "B0 plus a separate visual helper.",
                "describe_visible",
                "Helper reports image evidence.",
            ],
            "#7750b3",
            "#f4eefb",
        ),
        (
            1196,
            "F-adaptive · evolve",
            [
                "F plus optional hardware info.",
                "Define, probe and revise tools.",
                "Changes last for this episode.",
            ],
            "#235eb0",
            "#edf4ff",
        ),
    ]
    for x, title, lines, accent, fill in cards:
        f.box(x, 738, 348, 188, title, lines, accent, fill)
    f.text(
        56,
        970,
        "All actors receive images directly and share source access, scratch execution and the native action checks.",
        21,
    )
    f.text(
        56,
        1004,
        "F-adaptive can add available encoders or camera calibration; its information access is part of the treatment.",
        21,
    )
    f.text(
        56,
        1041,
        "Code: provider.py → runner.py → proxy.py → libero.py / robocasa.py   |   Private evaluation and replay are separate.",
        18,
        "#4b657a",
    )
    return f.finish()


def adaptive():
    f = Figure(
        "F-adaptive: evolving the agent's hardware interface",
        "Astra can inspect and enable available hardware information, measure native action responses, define versioned tools, execute them through the shared proxy, and revise or restore its interface. These are optional operations chosen by the actor, not a prescribed task plan.",
        1010,
    )
    f.text(56, 52, "F-ADAPTIVE · PER-EPISODE INTERFACE EVOLUTION", 15, "#235eb0", 700)
    f.text(
        56,
        103,
        "Astra can improve the tools it uses to control the robot",
        35,
        weight=700,
    )
    f.text(
        56,
        141,
        "Possible feedback loop; Astra chooses which operations to use and their order. There is no required recipe.",
        21,
        "#4b657a",
    )
    cards = [
        (
            56,
            208,
            "Inspect available information",
            [
                "interface_catalog",
                "enable_capability / read_capability",
                "Optional native encoders or calibration",
                "Only capabilities the adapter provides",
            ],
        ),
        (
            574,
            208,
            "Measure local action response",
            [
                "probe_dynamics",
                "Numeric readings before and after",
                "Bounded native actions move the robot",
                "Executed probe steps count in episode",
            ],
        ),
        (
            1092,
            208,
            "Create or revise an interface tool",
            [
                "define_interface_tool",
                "Named Python code + read/control role",
                "revise_interface_notes",
                "Store measured conventions or models",
            ],
        ),
        (
            1092,
            502,
            "Run the current tool version",
            [
                "call_interface_tool",
                "JSON parameters → isolated Python",
                "read / act / read_capability",
                "Read-only tools cannot dispatch actions",
            ],
        ),
        (
            574,
            502,
            "Execute through the shared proxy",
            [
                "Same native controller and action checks",
                "Bounds, shape, repeat count, freshness",
                "Control tools cannot bypass validation",
                "Robot motion consumes episode steps",
            ],
        ),
        (
            56,
            502,
            "Observe and update",
            [
                "Read current cameras and robot state",
                "Compare observations with hypotheses",
                "Revise tools or notes, or restore a version",
                "Restore does not reverse robot motion",
            ],
        ),
    ]
    for x, y, title, lines in cards:
        f.box(x, y, 452, 220, title, lines, "#aac4e8", "#edf4ff")
    f.arrow("M509 318H564")
    f.arrow("M1027 318H1082")
    f.arrow("M1318 429V492")
    f.arrow("M1091 612H1036")
    f.arrow("M573 612H518")
    f.arrow("M282 501V438", dashed=True)
    f.text(306, 473, "repeat as needed", 17, "#4b657a")
    f.box(
        56,
        785,
        720,
        165,
        "Versioned evidence",
        [
            "adaptive.py: code, hashes, capabilities, notes and reasons",
            "interface-evolution/revision-*.json + events.jsonl",
            "Independent replay includes probe and tool-generated actions",
        ],
    )
    f.box(
        824,
        785,
        720,
        165,
        "Reset and interpretation",
        [
            "Empty tools and notes at each new trial; no cross-trial transfer",
            "Simulator, native limits and evaluator stay outside the editor",
            "This treatment combines tool evolution and optional extra info",
        ],
    )
    f.text(
        56,
        987,
        "Example behavior, not a fixed policy. Architecture diagrams and synthetic JSON examples are separate from trial evidence.",
        18,
        "#4b657a",
    )
    return f.finish()
