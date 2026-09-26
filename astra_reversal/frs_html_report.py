"""Render audited FRS report data without running inference or collecting results.

``python -m astra_reversal.frs_html_report report.json --output NEW_DIRECTORY``

The input contract is documented in reports/frs_policy_improvement/HTML_SCHEMA.md.
Coverage and arithmetic are checked here; supplied audit hashes bind the separate
recording audits. This renderer does not independently replay model or physics.
"""

import argparse
import hashlib
import html
import io
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from .interpolation_catalog import ORACLES
from .records import file_sha256

SCHEMA_VERSION = "frs-policy-report-1.0"
TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens")
TASKS = {
    f"{row.suite}:{row.task_id}": row.task_name.replace("_", " ") for row in ORACLES
}
EPISODE = re.compile(r"(libero_(?:goal|spatial)_ood):seed(\d+):task([0-9]):state(\d+)")
COLORS = ("#176b87", "#8956a3", "#dc7d23", "#31764b", "#bf4661", "#386bc0")


def require(value, message):
    if not value:
        raise ValueError(message)


def _integer(value, label):
    require(type(value) is int and value >= 0, f"{label} must be a nonnegative integer")
    return value


def _number(value, label):
    require(
        type(value) in (int, float) and math.isfinite(value) and value >= 0,
        f"{label} must be finite and nonnegative",
    )
    return value


def _sha(value, label):
    require(
        type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value),
        f"{label} must be a SHA256 digest",
    )
    return value


def _text(value, label):
    require(type(value) is str and value.strip(), f"{label} must be nonempty text")
    return value


def _strings(values, label):
    require(type(values) is list, f"{label} must be a list")
    for value in values:
        _text(value, label)
    require(len(set(values)) == len(values), f"{label} contains duplicates")
    return values


def _episode(value):
    match = EPISODE.fullmatch(_text(value, "episode_id"))
    require(match is not None, "Episode ID must namespace suite, seed, task and reset")
    suite, seed, task, state = match.groups()
    return f"{suite}:{task}", int(seed), int(state)


def _usage(value):
    fields = {"calls", "accepted_calls", "failed_calls", "preflight_failures", "tokens"}
    require(type(value) is dict and set(value) == fields, "Malformed provider usage")
    for key in fields - {"tokens"}:
        _integer(value[key], key)
    require(
        value["accepted_calls"] + value["failed_calls"] == value["calls"],
        "Accepted/failed calls do not sum to physical calls",
    )
    require(
        type(value["tokens"]) is dict and set(value["tokens"]) == set(TOKEN_FIELDS),
        "Token fields are missing or extra",
    )
    for field, row in value["tokens"].items():
        require(
            type(row) is dict and set(row) == {"sum", "missing_calls"},
            "Malformed token availability",
        )
        _integer(row["sum"], field)
        _integer(row["missing_calls"], f"{field}.missing_calls")
        require(
            row["missing_calls"] <= value["calls"],
            "Missing usage exceeds physical calls",
        )
        require(
            row["missing_calls"] < value["calls"] or row["sum"] == 0,
            "A field missing from every call has no known token sum",
        )
        require(
            value["calls"] or row["sum"] == row["missing_calls"] == 0,
            "Zero physical calls cannot have provider tokens",
        )
    tokens = value["tokens"]
    if all(tokens[name]["missing_calls"] == 0 for name in TOKEN_FIELDS[:3]):
        require(
            tokens["total_tokens"]["sum"]
            == tokens["input_tokens"]["sum"] + tokens["output_tokens"]["sum"],
            "Complete total tokens must equal input plus output",
        )
    if (
        tokens["reasoning_tokens"]["missing_calls"]
        == tokens["output_tokens"]["missing_calls"]
        == 0
    ):
        require(
            tokens["reasoning_tokens"]["sum"] <= tokens["output_tokens"]["sum"],
            "Reasoning tokens are a subset of output, not additional cost",
        )
    return value


def _sum_usage(rows):
    rows = list(rows)
    return {
        **{
            name: sum(row[name] for row in rows)
            for name in (
                "calls",
                "accepted_calls",
                "failed_calls",
                "preflight_failures",
            )
        },
        "tokens": {
            field: {
                key: sum(row["tokens"][field][key] for row in rows)
                for key in ("sum", "missing_calls")
            }
            for field in TOKEN_FIELDS
        },
    }


def _read_json(path):
    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "Duplicate JSON object key")
            value[key] = item
        return value

    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    return json.loads(
        Path(path).read_text(), object_pairs_hook=object_pairs, parse_constant=nonfinite
    )


def _source(source):
    require(
        type(source) is dict and {"label", "href"} <= set(source),
        "Malformed report source",
    )
    _text(source["label"], "source label")
    href = _text(source["href"], "source href")
    parsed = urlsplit(href)
    require(
        (parsed.scheme in ("http", "https") and parsed.netloc)
        or (
            not parsed.scheme and not parsed.netloc and not href.startswith(("/", "\\"))
        ),
        "Sources must use public HTTP(S) or relative paths",
    )
    if source.get("sha256") is not None:
        _sha(source["sha256"], "source sha256")


def validate_report(value):
    """Validate complete coverage, prompt identity, cost arithmetic and reuse."""
    require(
        type(value) is dict and value.get("schema_version") == SCHEMA_VERSION,
        "Unsupported FRS HTML report schema",
    )
    _text(value["title"], "title")
    require(
        value["status"] in ("planned", "partial", "complete"), "Unknown report status"
    )
    require(
        value["phase"] in ("development", "evaluation"),
        "Development and evaluation must be separate reports",
    )
    require(
        type(value["tasks"]) is list and len(value["tasks"]) == 20,
        "Report must preserve the full 20-task inventory",
    )
    tasks = {}
    for task in value["tasks"]:
        key = task["task_key"]
        require(
            key not in tasks and key in TASKS and task["instruction"] == TASKS[key],
            "Task names/order IDs differ from released 20-task inventory",
        )
        tasks[key] = task
    require(set(tasks) == set(TASKS), "Missing released OOD task")
    methods = {}
    for row in value["methods"]:
        name = _text(row["id"], "method id")
        require(name not in methods, "Duplicate method id")
        _text(row["label"], "method label")
        _text(row["description"], "method description")
        methods[name] = row
    require(bool(methods), "Declare at least one method")
    prompts = set()
    for prompt in value["prompts"]:
        require(prompt["id"] not in prompts, "Duplicate prompt id")
        prompts.add(_text(prompt["id"], "prompt id"))
        text = _text(prompt["text"], "exact prompt text")
        _text(prompt["scope"], "prompt scope")
        require(
            prompt["sha256"] == hashlib.sha256(text.encode()).hexdigest(),
            "Exact prompt text does not match its SHA256",
        )
    require(type(value["protocol"]) is dict, "Missing protocol description")
    _strings(value["notes"], "notes")
    for source in value["sources"]:
        _source(source)
    protocol = value["protocol"]
    _integer(protocol["seed"], "protocol seed")
    _integer(protocol["adaptation_state"], "adaptation reset")
    _integer(protocol["rounds"], "protocol rounds")
    require(
        protocol["rounds"] > 0
        and type(protocol["evaluation_states"]) is list
        and bool(protocol["evaluation_states"]),
        "Missing round/reset plan",
    )
    require(
        all(
            type(state) is int and state >= 0 for state in protocol["evaluation_states"]
        )
        and len(set(protocol["evaluation_states"]))
        == len(protocol["evaluation_states"])
        and protocol["adaptation_state"] not in protocol["evaluation_states"],
        "Adaptation and evaluation resets must be distinct",
    )
    physical, cohort_ids, summaries = {}, set(), []
    for cohort in value["cohorts"]:
        name = _text(cohort["id"], "cohort id")
        require(name not in cohort_ids, "Duplicate cohort id")
        cohort_ids.add(name)
        _text(cohort["label"], "cohort label")
        require(
            cohort["status"] in ("planned", "partial", "complete"),
            "Unknown cohort status",
        )
        require(
            cohort["curve_kind"] in ("checkpoint_evaluation", "best_of_attempts"),
            "Learning and retry curves require an explicit estimand",
        )
        require(
            type(cohort.get("historical", False)) is bool, "historical must be boolean"
        )
        expected = _strings(cohort["expected_episode_ids"], "expected episodes")
        require(bool(expected), "A cohort needs a declared denominator")
        for episode in expected:
            _episode(episode)
        cohort_methods = _strings(cohort["method_ids"], "cohort methods")
        require(
            bool(cohort_methods) and set(cohort_methods) <= set(methods),
            "Unknown or empty cohort method set",
        )
        protocol = value["protocol"]
        if not cohort.get("historical", False) and value["phase"] == "evaluation":
            seed = protocol["seed"]
            states = (
                [protocol["adaptation_state"]]
                if cohort["curve_kind"] == "best_of_attempts"
                else protocol["evaluation_states"]
            )
            prescribed = {
                f"{key.split(':')[0]}:seed{seed}:task{key.split(':')[1]}:state{state}"
                for key in TASKS
                for state in states
            }
            require(
                set(expected) == prescribed,
                "Evaluation/adaptation denominator differs from the full frozen task/reset protocol",
            )
            method_field = (
                "adaptation_methods"
                if cohort["curve_kind"] == "best_of_attempts"
                else "evaluation_methods"
            )
            prescribed_methods = _strings(protocol[method_field], method_field)
            require(
                set(cohort_methods) == set(prescribed_methods),
                "Cohort omits or adds a frozen protocol method",
            )
        rounds = cohort["rounds"]
        require(type(rounds) is list, "rounds must be a list")
        indices, used_methods = [], set()
        row_count = 0
        for round_ in rounds:
            index = _integer(round_["index"], "round index")
            if not cohort.get("historical", False):
                require(
                    index <= protocol["rounds"], "Round exceeds the frozen protocol cap"
                )
            indices.append(index)
            _text(round_["label"], "round label")
            selected = _strings(round_["method_ids"], "round methods")
            require(
                bool(selected) and set(selected) <= set(cohort_methods),
                "Round contains undeclared methods",
            )
            used_methods.update(selected)
            if cohort["curve_kind"] == "best_of_attempts":
                require(
                    set(selected) == set(cohort_methods),
                    "Retry curves require every declared arm at each fixed round",
                )
            keys = set()
            for row in round_["episodes"]:
                key = (row["episode_id"], row["method_id"])
                require(
                    key not in keys and key[0] in expected and key[1] in selected,
                    "Duplicate, unexpected or misassigned episode/method row",
                )
                keys.add(key)
                require(
                    type(row["success"]) is bool, "Recorded success must be boolean"
                )
                require(
                    row["error"] is None
                    or (type(row["error"]) is str and bool(row["error"])),
                    "Malformed execution error",
                )
                _integer(row["actions"], "actions")
                require(
                    not row["success"] or (row["actions"] > 0 and row["error"] is None),
                    "Zero-action/error success cannot be credited",
                )
                _integer(row["velocity_evaluations"], "velocity evaluations")
                _number(row["wall_seconds"], "wall seconds")
                _usage(row["provider_usage"])
                _sha(row["policy_sha256"], "policy identity")
                _sha(row["source_sha256"], "recorded outcome source")
                require(
                    row["audit"]["status"] in ("passed", "pending", "failed"),
                    "Unknown audit status",
                )
                if row["audit"]["status"] == "passed":
                    _sha(row["audit"]["sha256"], "audit receipt")
                if cohort["status"] == "complete":
                    require(
                        row["audit"]["status"] == "passed" and row["error"] is None,
                        "Complete efficacy requires passed audits and no execution errors",
                    )
                run = _text(row["physical_run_id"], "physical run id")
                require(type(row["reused"]) is bool, "Physical reuse must be explicit")
                identity = {
                    key: val
                    for key, val in row.items()
                    if key not in ("method_id", "reused", "notes")
                }
                if run in physical:
                    require(
                        row["reused"] and physical[run] == identity,
                        "Reused physical run has changed evidence/cost or lacks an explicit reuse marker",
                    )
                else:
                    require(
                        not row["reused"],
                        "Reused run refers to no prior physical recording",
                    )
                    physical[run] = identity
            row_count += len(keys)
            if cohort["status"] == "complete":
                require(
                    keys
                    == {
                        (episode, method) for episode in expected for method in selected
                    },
                    "Complete round lacks its exact episode×method coverage",
                )
        require(
            indices == sorted(set(indices)),
            "Round indices must be unique and increasing",
        )
        if cohort["curve_kind"] == "best_of_attempts" and indices:
            require(
                indices == list(range(indices[-1] + 1)),
                "Retry curves must start at baseline0 without missing revisions",
            )
        if cohort["status"] == "complete":
            require(
                bool(rounds) and used_methods == set(cohort_methods),
                "Complete cohort omits declared methods/rounds",
            )
            if (
                not cohort.get("historical", False)
                and cohort["curve_kind"] == "best_of_attempts"
            ):
                require(
                    indices == list(range(protocol["rounds"] + 1)),
                    "Fixed adaptation rounds are incomplete",
                )
            if (
                not cohort.get("historical", False)
                and cohort["curve_kind"] == "checkpoint_evaluation"
                and "learned_noise" in cohort_methods
            ):
                learned_rounds = {
                    row["index"]
                    for row in rounds
                    if "learned_noise" in row["method_ids"]
                }
                require(
                    set(range(1, protocol["rounds"] + 1)) <= learned_rounds,
                    "A learned checkpoint evaluation round is missing",
                )
        if value["status"] == "complete":
            require(
                cohort["status"] == "complete", "Final report contains a pending cohort"
            )
        summaries.append(
            {
                "cohort_id": name,
                "recorded_rows": row_count,
                "expected_episodes": len(expected),
                "status": cohort["status"],
            }
        )
    require(
        value["status"] != "complete" or bool(cohort_ids),
        "An empty report is not a completed study",
    )
    overheads, overhead_ids = [], set()
    for row in value["overheads"]:
        name = _text(row["id"], "overhead id")
        require(
            name not in overhead_ids and name not in physical,
            "Duplicate overhead/physical cost identity",
        )
        overhead_ids.add(name)
        _text(row["label"], "overhead label")
        _usage(row["provider_usage"])
        _integer(row["velocity_evaluations"], "overhead velocity evaluations")
        _integer(row["training_steps"], "training steps")
        _number(row["wall_seconds"], "overhead wall seconds")
        _sha(row["source_sha256"], "overhead source")
        attribution = row.get("attribution")
        if attribution is not None:
            require(
                type(attribution) is dict
                and set(attribution)
                == {"cohort_id", "episode_id", "method_id", "round_index"},
                "Malformed overhead attribution",
            )
            cohort = next(
                (
                    item
                    for item in value["cohorts"]
                    if item["id"] == attribution["cohort_id"]
                ),
                None,
            )
            require(
                cohort is not None
                and attribution["episode_id"] in cohort["expected_episode_ids"]
                and attribution["method_id"] in cohort["method_ids"]
                and attribution["round_index"]
                in {item["index"] for item in cohort["rounds"]},
                "Overhead attribution names an absent cohort/episode/method/round",
            )
        overheads.append(row)
    if value["status"] == "complete" and any(
        row["provider_usage"]["calls"] for row in [*physical.values(), *overheads]
    ):
        require(
            bool(prompts),
            "A complete provider-backed study must include its exact prompt text",
        )
    return {
        "cohorts": summaries,
        "physical_cost": {
            "unique_rollouts": len(physical),
            "actions": sum(row["actions"] for row in physical.values()),
            "velocity_evaluations": sum(
                row["velocity_evaluations"] for row in [*physical.values(), *overheads]
            ),
            "training_steps": sum(row["training_steps"] for row in overheads),
            "summed_wall_seconds": sum(
                row["wall_seconds"] for row in [*physical.values(), *overheads]
            ),
            "provider_usage": _sum_usage(
                row["provider_usage"] for row in [*physical.values(), *overheads]
            ),
        },
    }


def _rows_at(cohort, method, index):
    rounds = [
        row
        for row in cohort["rounds"]
        if row["index"] <= index and method in row["method_ids"]
    ]
    if cohort["curve_kind"] == "checkpoint_evaluation":
        rounds = [row for row in rounds if row["index"] == index]
    rows = [
        row
        for round_ in rounds
        for row in round_["episodes"]
        if row["method_id"] == method
    ]
    groups = defaultdict(list)
    for row in rows:
        groups[row["episode_id"]].append(row)
    return groups


def derived_results(value):
    """Compute displayed counts/curves from verified rows, never supplied rates."""
    results = {}
    for cohort in value["cohorts"]:
        if cohort["status"] != "complete" or (
            value["status"] != "complete" and not cohort.get("historical", False)
        ):
            continue
        methods = {}
        for method in cohort["method_ids"]:
            relevant_overheads = [
                row
                for row in value["overheads"]
                if (row.get("attribution") or {}).get("cohort_id") == cohort["id"]
                and row["attribution"]["method_id"] == method
            ]
            method_rows = [
                row
                for round_ in cohort["rounds"]
                for row in round_["episodes"]
                if row["method_id"] == method
            ]
            unique_rows = {row["physical_run_id"]: row for row in method_rows}
            points = []
            for round_ in cohort["rounds"]:
                if method not in round_["method_ids"]:
                    continue
                groups = _rows_at(cohort, method, round_["index"])
                points.append(
                    {
                        "round": round_["index"],
                        "label": round_["label"],
                        "successes": sum(
                            any(row["success"] for row in rows)
                            for rows in groups.values()
                        ),
                        "episodes": len(groups),
                    }
                )
            final = _rows_at(cohort, method, points[-1]["round"])
            per_task = defaultdict(lambda: {"successes": 0, "episodes": 0})
            for episode, rows in final.items():
                key = _episode(episode)[0]
                per_task[key]["episodes"] += 1
                per_task[key]["successes"] += any(row["success"] for row in rows)
            detail = {
                "points": points,
                "per_task": dict(per_task),
                "rescue": None,
                "provider_usage": _sum_usage(
                    row["provider_usage"]
                    for row in [*unique_rows.values(), *relevant_overheads]
                ),
                "standalone_unique_rollouts": len(unique_rows),
                "standalone_actions": sum(
                    row["actions"] for row in unique_rows.values()
                ),
            }
            if cohort["curve_kind"] == "best_of_attempts":
                first, rescued, rescue_tokens, missing_tokens, baseline_failed = (
                    [],
                    [],
                    [],
                    0,
                    0,
                )
                for episode in cohort["expected_episode_ids"]:
                    rows = [
                        next(
                            row
                            for row in round_["episodes"]
                            if row["episode_id"] == episode
                            and row["method_id"] == method
                        )
                        for round_ in cohort["rounds"]
                    ]
                    index = next(
                        (i for i, row in enumerate(rows) if row["success"]), None
                    )
                    if index is not None:
                        first.append(index)
                    if not rows[0]["success"]:
                        baseline_failed += 1
                        if index is not None:
                            rescued.append(index)
                            prefix_overheads = [
                                row
                                for row in relevant_overheads
                                if row["attribution"]["episode_id"] == episode
                                and row["attribution"]["round_index"] <= index
                            ]
                            usage = _sum_usage(
                                row["provider_usage"]
                                for row in [*rows[: index + 1], *prefix_overheads]
                            )
                            total = usage["tokens"]["total_tokens"]
                            if total["missing_calls"]:
                                missing_tokens += 1
                            else:
                                rescue_tokens.append(total["sum"])
                detail["rescue"] = {
                    "baseline_failed": baseline_failed,
                    "rescued": len(rescued),
                    "censored": len(cohort["expected_episode_ids"]) - len(first),
                    "median_revision_among_rescues": statistics.median(rescued)
                    if rescued
                    else None,
                    "median_first_success_among_successes": statistics.median(first)
                    if first
                    else None,
                    "median_tokens_among_rescues_with_complete_usage": statistics.median(
                        rescue_tokens
                    )
                    if rescue_tokens
                    else None,
                    "rescues_missing_usage": missing_tokens,
                }
            methods[method] = detail
        results[cohort["id"]] = methods
    return results


def _escape(value):
    return html.escape(str(value), quote=True)


def _table(headers, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{_escape(item)}</th>" for item in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{_escape(item)}</td>" for item in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def _tokens(value):
    return f"{value['sum']:,}" + (
        f" known; {value['missing_calls']} calls missing usage"
        if value["missing_calls"]
        else ""
    )


def _diagram():
    boxes = [
        (20, 36, "Raw observation", "external + wrist; task + robot state"),
        (
            335,
            36,
            "Astra coarse direction / edit",
            "VLM-only calibrated view if enabled",
        ),
        (650, 36, "Reverse then denoise", "Euler10 → noise → Euler10 → actions"),
        (650, 181, "Execute and record", "simulator outcome kept separate"),
        (335, 181, "Astra comparison", "same-arm rollout evidence; no success label"),
        (20, 181, "Optional noise-policy BC", "accepted replay; frozen base policy"),
    ]
    parts = [
        '<svg class="diagram" viewBox="0 0 950 300" role="img" aria-label="FRS steering and iterative policy improvement"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#557184"/></marker></defs>'
    ]
    for x, y, title, sub in boxes:
        parts.append(
            f'<rect x="{x}" y="{y}" width="280" height="84" rx="12" fill="#eef4f8" stroke="#bdd0dc"/><text x="{x + 140}" y="{y + 30}" text-anchor="middle" font-size="15" font-weight="600">{_escape(title)}</text><text x="{x + 140}" y="{y + 55}" text-anchor="middle" font-size="11">{_escape(sub)}</text>'
        )
    for d in (
        "M300 78H332",
        "M615 78H647",
        "M790 120V178",
        "M650 223H618",
        "M335 223H303",
        "M160 181V123",
    ):
        parts.append(
            f'<path d="{d}" stroke="#557184" stroke-width="2" fill="none" marker-end="url(#arrow)"/>'
        )
    parts.append(
        '<text x="475" y="290" text-anchor="middle" font-size="12">Evaluate updated checkpoints on separate captured resets; never return evaluation feedback to adaptation.</text></svg>'
    )
    return "".join(parts)


def _curve_svg(cohort, methods, labels):
    from matplotlib import rc_context
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure

    with rc_context(
        {"svg.fonttype": "none", "svg.hashsalt": "frs-report-1", "font.size": 10}
    ):
        figure = Figure(figsize=(8.4, 3.8), layout="constrained")
        axes = figure.subplots()
        for number, (method, detail) in enumerate(methods.items()):
            points = detail["points"]
            axes.plot(
                [row["round"] for row in points],
                [100 * row["successes"] / row["episodes"] for row in points],
                marker="o",
                linewidth=2,
                color=COLORS[number % len(COLORS)],
                label=labels[method],
            )
        axes.set(
            ylim=(0, 100),
            ylabel="Success (%)",
            xlabel="Completed adaptation round; each point evaluates its checkpoint"
            if cohort["curve_kind"] == "checkpoint_evaluation"
            else "Up to this full-rollout revision; 0 is the common baseline",
        )
        axes.set_xticks(
            sorted(
                {
                    point["round"]
                    for detail in methods.values()
                    for point in detail["points"]
                }
            )
        )
        axes.grid(axis="y", alpha=0.2)
        axes.legend(fontsize=8, loc="best")
        stream = io.StringIO()
        FigureCanvasSVG(figure).print_svg(stream, metadata={"Date": None})
        source = stream.getvalue()
        return source[source.index("<svg") :]


CSS = """
:root{color-scheme:light;--ink:#173247;--muted:#546979;--line:#d9e2e9;--paper:#fff;--accent:#176b87}
*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:var(--ink);font:16px/1.55 system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:36px 28px 72px}header{padding:28px 0;border-bottom:1px solid var(--line)}
h1{font-size:clamp(1.9rem,4vw,3rem);line-height:1.12;max-width:960px}h2{font-size:1.55rem;margin-top:0}h3{font-size:1.1rem}
.eyebrow{letter-spacing:.12em;text-transform:uppercase;color:var(--accent);font-size:.8rem;font-weight:700}
.muted{color:var(--muted)}section{background:var(--paper);padding:28px;border:1px solid var(--line);border-radius:16px;margin:22px 0}
.status{display:inline-block;background:#e9f1f5;border-radius:24px;padding:5px 13px;font-weight:650}.pending{background:#fff3d9;color:#805917}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}.card{padding:18px;border:1px solid var(--line);border-radius:12px}.card strong{display:block;font-size:1.75rem}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.86rem}th,td{text-align:left;padding:11px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{background:#eef4f8;white-space:nowrap}td:first-child{min-width:200px}tr:hover td{background:#f8fafc}
details{border-top:1px solid var(--line);padding:14px 0}summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f9;padding:18px;border-radius:9px;font:12px/1.55 ui-monospace,monospace}code{overflow-wrap:anywhere}
svg{max-width:100%;height:auto}.diagram{font-family:system-ui,sans-serif;fill:var(--ink)}a{color:var(--accent)}nav{display:flex;gap:18px;flex-wrap:wrap;margin-top:22px}.scope{border-left:4px solid #d59a36;padding-left:15px}.chart{margin:18px auto;max-width:950px}footer{font-size:.85rem;color:var(--muted)}
@media(max-width:650px){main{padding:18px 12px}section{padding:18px}.diagram{min-width:650px}}
@media print{body{background:white}section{break-inside:avoid}details{display:block}pre{font-size:9px}nav{display:none}}
"""


def render_report(report_path, output):
    """Write a new portable HTML/JSON/SVG bundle, preserving original input bytes."""
    report_path, output = Path(report_path), Path(output)
    require(not output.exists(), "Refusing to overwrite an existing report directory")
    raw = report_path.read_bytes()
    value = _read_json(report_path)
    checked = validate_report(value)
    results = derived_results(value)
    labels = {row["id"]: row["label"] for row in value["methods"]}
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{_escape(value['title'])}</title><style>{CSS}</style></head><body><main>",
        '<header><p class="eyebrow">Frozen generalist · Flow reversal steering · Audited evidence</p>',
        f'<h1>{_escape(value["title"])}</h1><span class="status {"pending" if value["status"] != "complete" else ""}">{_escape(value["phase"])} · {_escape(value["status"])}</span>',
        '<p class="scope">Known published OOD compositions. New reset samples are not new tasks. Retry success and evaluation of learned checkpoints are separate quantities.</p><nav><a href="#coverage">Coverage</a><a href="#approach">Approach</a><a href="#results">Results</a><a href="#cost">Cost</a><a href="#prompts">Prompts</a><a href="#provenance">Provenance</a></nav></header>',
    ]
    parts += [
        '<section id="coverage"><h2>Coverage and completion</h2>',
        "<p>All 20 released task names remain in the inventory. Complete efficacy requires the entire declared reset×method grid and passed recording audits.</p>",
        _table(
            ("Cohort", "State", "Expected episodes", "Recorded method/round rows"),
            [
                (
                    row["cohort_id"],
                    row["status"],
                    row["expected_episodes"],
                    row["recorded_rows"],
                )
                for row in checked["cohorts"]
            ],
        ),
        "</section>",
        '<section id="approach"><h2>What changes, and what stays frozen</h2><div class="scroll">',
        _diagram(),
        "</div>",
    ]
    for method in value["methods"]:
        parts.append(
            f"<h3>{_escape(method['label'])}</h3><p>{_escape(method['description'])}</p>"
        )
    parts.append(
        f'<details><summary>Exact supplied protocol</summary><pre>{_escape(json.dumps(value["protocol"], indent=2, sort_keys=True))}</pre></details></section><div id="results">'
    )
    curves = {}
    for number, cohort in enumerate(value["cohorts"]):
        parts.append(f"<section><h2>{_escape(cohort['label'])}</h2>")
        methods = results.get(cohort["id"])
        if methods is None:
            parts.append(
                '<p class="scope">Pending complete study/audit. No success rate, rescue count or learning curve is released from these partial records.</p></section>'
            )
            continue
        is_retry = cohort["curve_kind"] == "best_of_attempts"
        parts.append(
            "<p>"
            + (
                "Cumulative success with reset access: each point counts success by that revision. This is not a learned-policy evaluation."
                if is_retry
                else "Each point evaluates the indicated frozen checkpoint on the declared evaluation resets. Success is not accumulated across checkpoints."
            )
            + "</p>"
        )
        chart = _curve_svg(cohort, methods, labels)
        curves[f"cohort_{number:02d}_curves.svg"] = chart.encode()
        parts.append(f'<div class="chart">{chart}</div>')
        method_costs = []
        for method, detail in methods.items():
            final = detail["points"][-1]
            usage = detail["provider_usage"]
            method_costs.append(
                [
                    labels[method],
                    f"{final['successes']}/{final['episodes']}",
                    detail["standalone_unique_rollouts"],
                    usage["calls"],
                    usage["failed_calls"],
                    _tokens(usage["tokens"]["input_tokens"]),
                    _tokens(usage["tokens"]["output_tokens"]),
                    _tokens(usage["tokens"]["total_tokens"]),
                ]
            )
        parts.append(
            _table(
                (
                    "Method",
                    "Final success",
                    "Recorded rollouts",
                    "Calls",
                    "Failed calls",
                    "Input tokens",
                    "Output tokens",
                    "Total tokens",
                ),
                method_costs,
            )
        )
        parts.append(
            '<p class="muted">Per-method costs include every recorded round and its attributed overheads, including capped failures. Shared baselines may appear in several method attributions; use the deduplicated physical total below for actual experiment cost.</p>'
        )
        rows = []
        for key, instruction in TASKS.items():
            row = [f"{key} · {instruction}"]
            for method in cohort["method_ids"]:
                task = methods[method]["per_task"].get(key)
                row.append(
                    f"{task['successes']}/{task['episodes']}"
                    if task
                    else "Outside this cohort"
                )
            rows.append(row)
        parts.append(
            _table(["Task", *[labels[name] for name in cohort["method_ids"]]], rows)
        )
        if is_retry:
            rows = []
            for name, detail in methods.items():
                rescue = detail["rescue"]
                rows.append(
                    [
                        labels[name],
                        f"{rescue['rescued']}/{rescue['baseline_failed']}",
                        rescue["median_revision_among_rescues"]
                        if rescue["rescued"]
                        else "—",
                        rescue["median_tokens_among_rescues_with_complete_usage"]
                        if rescue["median_tokens_among_rescues_with_complete_usage"]
                        is not None
                        else "—",
                        rescue["rescues_missing_usage"],
                        rescue["censored"],
                    ]
                )
            parts.append("<h3>Conditional rescue and censoring</h3>")
            parts.append(
                _table(
                    (
                        "Method",
                        "Rescues / baseline failures",
                        "Median revisions among rescues",
                        "Median tokens among complete-usage rescues",
                        "Rescues with unknown token totals",
                        "Censored failures",
                    ),
                    rows,
                )
            )
            parts.append(
                '<p class="muted">Medians exclude baseline successes and failures. Failed/rejected calls before each rescue remain in token cost; censored searches remain in physical totals.</p>'
            )
        parts.append("</section>")
    cost = checked["physical_cost"]
    usage = cost["provider_usage"]
    parts.append('</div><section id="cost"><h2>Recorded physical cost</h2>')
    parts.append(
        '<div class="cards">'
        + "".join(
            f'<div class="card"><strong>{number:,}</strong>{_escape(label)}</div>'
            for number, label in [
                (cost["unique_rollouts"], "unique physical rollouts"),
                (cost["actions"], "executed actions"),
                (usage["calls"], "physical provider calls"),
                (usage["failed_calls"], "failed/rejected physical calls"),
            ]
        )
        + "</div>"
    )
    parts.append(
        _table(
            ("Quantity", "Recorded total"),
            [
                *[
                    (field.replace("_", " "), _tokens(usage["tokens"][field]))
                    for field in TOKEN_FIELDS
                ],
                ("Preflight failures (no network call)", usage["preflight_failures"]),
                (
                    "Velocity evaluations incl. supplied overheads",
                    cost["velocity_evaluations"],
                ),
                ("Auxiliary optimizer steps", cost["training_steps"]),
                (
                    "Sum of recorded run/overhead seconds",
                    f"{cost['summed_wall_seconds']:.2f}",
                ),
            ],
        )
    )
    parts.append(
        '<p class="muted">Shared physical recordings are counted once. Reasoning tokens are part of output tokens. Known token sums with missing usage are lower bounds; no dollar price is assumed. Summed task times are not parallel elapsed time or a pooled latency percentile.</p>'
    )
    parts.append(
        "<details><summary>Additional cost entries: setup, critique, judgments and training</summary>"
        + _table(
            ("Label", "Provider calls", "Known total tokens", "Optimizer steps"),
            [
                (
                    row["label"],
                    row["provider_usage"]["calls"],
                    _tokens(row["provider_usage"]["tokens"]["total_tokens"]),
                    row["training_steps"],
                )
                for row in value["overheads"]
            ],
        )
        + '</details></section><section id="prompts"><h2>Exact prompts and method instructions</h2>'
    )
    if not value["prompts"]:
        parts.append("<p>Prompt text has not been supplied in this planned report.</p>")
    for prompt in value["prompts"]:
        parts.append(
            f"<details><summary>{_escape(prompt['id'])}</summary><p>{_escape(prompt['scope'])}</p><code>SHA256 {_escape(prompt['sha256'])}</code><pre>{_escape(prompt['text'])}</pre></details>"
        )
    parts.append('</section><section id="provenance"><h2>Scope and provenance</h2>')
    for note in value["notes"]:
        parts.append(f"<p>{_escape(note)}</p>")
    parts.append("<ul>")
    for source in value["sources"]:
        checksum = f" · SHA256 {source['sha256']}" if source.get("sha256") else ""
        parts.append(
            f'<li><a href="{_escape(source["href"])}">{_escape(source["label"])}</a>{_escape(checksum)}</li>'
        )
    parts.append(
        '</ul><p>This renderer validates declared coverage, arithmetic, reuse and prompt hashes. Referenced audit receipts establish recording integrity; no simulator or model is rerun here.</p><p><a href="report.json">Exact input JSON</a> · <a href="derived.json">Derived display numbers</a> · <a href="manifest.json">Artifact hashes</a></p></section><footer>Self-contained HTML with inline CSS/SVG; no scripts, external fonts, remote images or inference requests.</footer></main></body></html>'
    )
    require(report_path.read_bytes() == raw, "Report input changed during rendering")
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_bytes(raw)
    (output / "index.html").write_text("".join(parts))
    (output / "derived.json").write_text(
        json.dumps(
            {"validation": checked, "cohorts": results},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    for name, data in curves.items():
        (output / name).write_bytes(data)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "renderer_sha256": file_sha256(__file__),
        "status": value["status"],
        "files": {
            path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in sorted(output.iterdir())
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = render_report(args.report, args.output)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "input_sha256": manifest["input_sha256"],
                "files": len(manifest["files"]),
            }
        )
    )


if __name__ == "__main__":
    main()
