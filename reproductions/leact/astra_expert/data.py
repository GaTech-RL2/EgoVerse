"""Immutable teacher labels, with the CFR oracle retained as an external evaluator."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

from ..compare import comparison
from ..core import kl
from ..data import file_sha256, fingerprint, write_json, write_jsonl
from .client import ENDPOINT, MODEL, PROMPT_VERSION, decode, payload


def solver_gate(teacher_metadata, gate, oracle_sha256):
    """The user's explicit 0.10 teacher threshold replaces the older dependency."""
    value = teacher_metadata.get("quality", {}).get("exploitability")
    if (teacher_metadata.get("num_infosets") != 4032
            or teacher_metadata.get("cfr_evaluation_oracle_sha256") != oracle_sha256
            or gate.get("oracle_sha256") != oracle_sha256 or gate.get("infosets") != 4032
            or gate.get("maximum") != 0.10 or gate.get("passed") is not True
            or type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 0.10
            or gate.get("exploitability") != value
            or not teacher_metadata.get("action_labels_sha256")
            or gate.get("action_labels_sha256") != teacher_metadata["action_labels_sha256"]):
        raise ValueError("Astra does not satisfy the frozen full-tree solver-quality gate")
    return gate


def leduc_gate(report, oracle_sha256):
    """A measured improvement gate, explicitly weaker than paper-number replication."""
    result = comparison(report)
    if result["oracle_sha256"] != oracle_sha256 or not result["all_planned_rounds_complete"]:
        raise ValueError("The matching Leduc comparison is not complete")
    rows = {(r["method"], r["regime"]): r for r in result["rows"]}
    measurements = {}
    for regime in ("single", "oracle_bon"):
        baseline = rows.get(("coldstart_thinking", regime))
        trained = rows.get(("leact", regime))
        if not baseline or not trained:
            raise ValueError("Leduc needs both anchor and trained measurements")
        for item in [baseline, trained]:
            if item["inference_seeds"] != [0, 1, 2] or not math.isfinite(item["mean"]):
                raise ValueError("Leduc needs finite measurements on all three inference seeds")
        if trained["mean"] >= baseline["mean"]:
            raise ValueError("Leduc has not improved on its reasoning anchor in both metrics")
        measurements[regime] = {"anchor": baseline["mean"], "leact": trained["mean"]}
    return {"passed": True, "measurements": measurements,
            "report_sha256": fingerprint(report), "oracle_sha256": oracle_sha256,
            "criterion": "All planned rounds complete; lower mean single and oracle-BoN exploitability than the shared reasoning anchor.",
            "paper_numbers_reproduced": False}


def pilot_states(states):
    selected = []
    for round_index in [0, 1]:
        for player in [0, 1]:
            group = [s for s in states if f"[Round {round_index}]" in s["info_state"] and s["player"] == player]
            if not group:
                raise ValueError("Missing round/player stratum")
            selected.append(min(group, key=lambda s: s["state_id"]))
    return selected


def read_call(path, state, metadata, *, committed=None):
    row = json.loads(Path(path).read_text())
    expected = payload(state, metadata["rules"], committed=committed)
    identity = {"state_id": state["state_id"], "oracle_sha256": metadata["sha256"],
                "prompt_version": PROMPT_VERSION, "endpoint": ENDPOINT,
                "stage": "summary" if committed is not None else "actions",
                "request_sha256": fingerprint(expected),
                "committed_decision_sha256": fingerprint(committed) if committed is not None else None}
    if any(row.get(k) != v for k, v in identity.items()) or not row.get("accepted"):
        raise ValueError("Teacher record is rejected or belongs to a different request")
    checked = decode(row["response"], state, summary=committed is not None)
    if checked != row.get("result") or row.get("request") != expected:
        raise ValueError("Teacher record content or response provenance changed")
    return row


def collect_records(states, metadata, root, *, summaries=False):
    labels = []
    for state in states:
        action = read_call(root / "actions" / (state["state_id"] + ".json"), state, metadata)
        label = {"state_id": state["state_id"], "teacher": MODEL, **action["result"],
                 "action_record_sha256": fingerprint(action)}
        if summaries:
            summary = read_call(root / "summaries" / (state["state_id"] + ".json"), state, metadata,
                                committed=action["result"])
            label.update(**summary["result"], summary_record_sha256=fingerprint(summary))
        labels.append(label)
    return labels


def export_teacher(states, metadata, root, output, *, summaries=False):
    """Export the complete Astra table for LeAct; never overwrite the CFR oracle."""
    if len(states) != 4032 or metadata["num_infosets"] != 4032:
        raise ValueError("Teacher export requires the complete 4032-state Leduc game")
    if output.exists() or output.with_suffix(".meta.json").exists():
        raise FileExistsError("Choose a fresh teacher output")
    labels = collect_records(states, metadata, root, summaries=summaries)
    rows = []
    for state, label in zip(states, labels):
        row = {k: state[k] for k in ["state_id", "info_state", "player", "legal_actions", "action_ids"]}
        row.update(expert_policy=label["policy"], teacher_action=label["action"],
                   teacher_record_sha256=label["action_record_sha256"])
        if summaries:
            row.update(teacher_decision_summary=label["decision_summary"],
                       teacher_summary_record_sha256=label["summary_record_sha256"])
        rows.append(row)
    # Directly measure the teacher's quality; never call Astra an oracle because
    # the model ID contains an expert label. Exploitability needs the full tree.
    import pyspiel
    table = {s["info_state"]: [(s["action_ids"][a], p) for a, p in label["policy"].items()]
             for s, label in zip(states, labels)}
    quality = {"exploitability": pyspiel.exploitability(
        pyspiel.load_game(metadata["game_string"]), pyspiel.TabularPolicy(table)),
        "mean_capped_kl_to_cfr": statistics.mean(kl(s["expert_policy"], l["policy"])
                                                  for s, l in zip(states, labels))}
    write_jsonl(output, rows)
    manifest = {k: metadata[k] for k in ["schema_version", "game_string", "parameters", "rules", "num_infosets"]}
    manifest.update(source="Astra teacher experiment; not a CFR oracle or paper reproduction",
                    teacher_model=MODEL, sha256=file_sha256(output),
                    cfr_evaluation_oracle_sha256=metadata["sha256"], converged=False,
                    exploitability=quality["exploitability"], quality=quality,
                    labels_sha256=fingerprint(labels), includes_public_summaries=summaries,
                    action_labels_sha256=fingerprint([{k: l[k] for k in ["state_id", "action", "policy"]}
                                                     for l in labels]))
    write_json(output.with_suffix(".meta.json"), manifest)
    return manifest


def tracking_records(states, metadata, root, *, label="Astra API preflight"):
    """Use the existing trace browser, always joining to the original CFR expert."""
    for state in states:
        path = root / "actions" / (state["state_id"] + ".json")
        if not path.exists():
            continue
        action = read_call(path, state, metadata)
        summary_path = root / "summaries" / path.name
        summary = read_call(summary_path, state, metadata, committed=action["result"]) if summary_path.exists() else None
        answer = action["result"]
        yield {"kind": "expert", "oracle_sha256": metadata["sha256"], "state_id": state["state_id"],
               "state": state, "solver": {k: metadata.get(k) for k in
                  ["exploitability", "iterations", "converged", "rules", "game"]}}
        yield {"kind": "trace", "run": "astra-expert/" + root.name, "seed": 0,
               "label": label, "scope": "API preflight; no SFT or benchmark result", "method": "astra_teacher",
               "state_id": state["state_id"], "info_state": state["info_state"],
               "oracle_sha256": metadata["sha256"], "baselines": [], "mean_baseline_policy": None,
               "baseline_kl": None, "prompts": {"forward": action["request"]["messages"][-1]["content"],
                   "backward": summary["request"]["messages"][-1]["content"] if summary else None,
                   "note": "Public explanation generated after freezing the action; not private internal reasoning."},
               "candidates": [{"candidate_id": 0, "selected": False, "complete": True,
                    "reasoning": summary["result"]["decision_summary"] if summary else "",
                    "reasoning_kind": "public_decision_summary", "predicted_policy": answer["policy"],
                    "fallback": None, "score": None, "kl": kl(state["expert_policy"], answer["policy"]),
                    "forward": {"text": action["response"]["choices"][0]["message"]["content"],
                                "finish_reason": "stop", "generated_tokens": action["response"].get("usage", {}).get("completion_tokens")},
                    "sft_admission": {"admitted_to_sft": False, "reason": "Preflight only; waiting for Leduc"}}],
               "provenance": {"action": action, "public_summary": summary}}
