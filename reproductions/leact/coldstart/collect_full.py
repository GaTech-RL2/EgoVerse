"""Collect full-coverage frontier warmstarts or evaluate the independent Astra teacher."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
import time

from ..astra_expert.client import ENDPOINT, MODEL, call, call_request
from ..astra_expert.data import export_teacher, read_call
from ..data import fingerprint, load_oracle, write_json, write_jsonl
from .data import PROMPT_VERSION, request_for, validate_candidate


def nested_states(states):
    """Fixed round-robin strata, independent of model outputs and solver scores."""
    groups = {}
    for s in states:
        text = s["info_state"]
        private = re.search(r"\[Private: ([^\]]+)\]", text).group(1)
        public = re.search(r"\[Public: ([^\]]*)\]", text).group(1)
        key = (re.search(r"\[Round (\d+)\]", text).group(1), s["player"],
               tuple(s["legal_actions"]), private[:-1], bool(public and private[:-1] == public[:-1]))
        groups.setdefault(key, []).append(s)
    groups = {k: sorted(v, key=lambda s: fingerprint(["coldstart-order-v1", s["state_id"]]))
              for k, v in sorted(groups.items())}
    return [v[i] for i in range(max(map(len, groups.values()))) for v in groups.values() if i < len(v)]


def budget_rejection(record):
    """A provider budget denial contains no model completion to select from."""
    if record.get("accepted") or record.get("http_status") != 422:
        return False
    try:
        return json.loads(record.get("provider_error", "{}"))["error"]["type"] == "budget_exceeded"
    except (ValueError, KeyError, TypeError):
        return False


def collect_stage(stage, states, metadata, root, *, workers=8, recover_budget=False):
    stop = threading.Event()

    def one(state):
        if stop.is_set():
            return None
        destination = root / stage / (state["state_id"] + ".json")
        committed = (read_call(root / "actions" / (state["state_id"] + ".json"), state, metadata)["result"]
                     if stage == "summaries" else None)
        if stage == "frontier":
            request = request_for(state, metadata["rules"])
            identity = {"state_id": state["state_id"], "oracle_sha256": metadata["sha256"],
                        "prompt_version": PROMPT_VERSION, "endpoint": ENDPOINT,
                        "stage": "frontier_coldstart_full", "request_sha256": fingerprint(request)}

        def invoke(path):
            if stage == "frontier":
                return call_request(request, path, identity=identity,
                                    validate=lambda env: validate_candidate(env, state))
            return call(state, metadata["rules"], path, oracle_sha256=metadata["sha256"], committed=committed)

        if destination.exists():
            return invoke(destination)
        attempts = root / "attempts" / stage / state["state_id"]
        # A new, explicitly authorized credential may replace budget-rejected
        # calls. Retain their original records and never reset the budget for
        # rejected model outputs or other failures.
        budget_denials = sum(budget_rejection(json.loads(p.read_text()))
                            for p in attempts.glob("attempt-*.json")) if recover_budget else 0
        attempt_limit = 5 + budget_denials
        for attempt in range(1, attempt_limit + 1):
            if stop.is_set():
                return None
            path = attempts / f"attempt-{attempt:02d}.json"
            if path.exists() and not json.loads(path.read_text()).get("accepted"):
                continue
            try:
                value = invoke(path)
                value["accepted_attempt"] = attempt
                write_json(destination, value)
                return value
            except ValueError:
                evidence = json.loads(path.read_text()) if path.exists() else {}
                if (budget_rejection(evidence) or evidence.get("http_status") in (400, 401, 403, 404)
                        or attempt == attempt_limit):
                    stop.set()
                    raise
                # Bounded format/transport retries are retained, never repaired
                # or selected by agreement with the CFR policy.
                time.sleep(min(10 * 2**(attempt - 1), 60))
        stop.set()
        raise ValueError("Recorded attempt budget exhausted for " + state["state_id"])

    completed, tokens = 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, s) for s in states]
        try:
            for future in as_completed(futures):
                row = future.result()
                if row is None:
                    continue
                completed += 1
                tokens += row["response"].get("usage", {}).get("total_tokens", 0)
                progress = {"stage": stage, "completed": completed, "total": len(states),
                            "total_tokens": tokens, "updated_utc": datetime.now(timezone.utc).isoformat()}
                write_json(root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        except Exception:
            stop.set()
            for future in futures:
                future.cancel()
            raise
    if completed != len(states):
        raise ValueError("Collection is incomplete")
    write_json(root / (stage + "-complete.json"), {"states": completed, "total_tokens": tokens,
               "oracle_sha256": metadata["sha256"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["frontier", "astra"])
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=8)
    parser.add_argument("--limit", type=int, help="API smoke check only; never exports a full teacher")
    parser.add_argument("--recover-budget", action="store_true",
                        help="Replace prior provider budget denials after authorized credential/budget restoration; retain every prior attempt")
    args = parser.parse_args()
    states, metadata = load_oracle(args.oracle)
    states = nested_states(states)
    spec = {"model": MODEL, "oracle_sha256": metadata["sha256"], "mode": args.mode,
            "state_order": [s["state_id"] for s in states], "workers": args.workers,
            "astra_gate": {"metric": "full-tree exploitability", "maximum": 0.1,
                           "authority": "User explicitly selected <=0.10 on 2026-10-21"},
            "retry_policy": "At most five recorded attempts per state; no CFR-based response selection",
            "paper_reconstruction": "Full-coverage joint corpus, one forward frontier candidate per infoset; CFR replaces the action target. Author 6r2s CoT corpus size and prompts unavailable."}
    path = args.root / "collection.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("Collection specification changed")
    write_json(path, spec)
    if args.recover_budget:
        write_json(args.root / "budget-recovery.json", {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "policy": "Prior HTTP 422 budget_exceeded denials may be replaced. Five non-budget attempts maximum per state; accepted responses reused unchanged.",
            "original_attempts_preserved": True,
        })
    selected = states if args.limit is None else states[:args.limit]
    collect_stage("frontier" if args.mode == "frontier" else "actions", selected, metadata, args.root,
                  workers=args.workers, recover_budget=args.recover_budget)
    if args.limit is not None:
        return
    if args.mode == "frontier":
        rows = [{"state_id": s["state_id"], **json.loads((args.root / "frontier" / (s["state_id"] + ".json")).read_text())["result"]}
                for s in states]
        write_jsonl(args.root / "frontier.jsonl", rows)
        write_json(args.root / "complete.json", {"states": len(rows), "oracle_sha256": metadata["sha256"],
                   "rows_sha256": fingerprint(rows), "action_target": "CFR, not the generated frontier policy"})
    else:
        teacher = args.root / "teacher-actions.jsonl"
        quality = (json.loads(teacher.with_suffix(".meta.json").read_text()) if teacher.exists()
                   else export_teacher(states, metadata, args.root, teacher))
        gate = {"passed": quality["quality"]["exploitability"] <= 0.10,
                "exploitability": quality["quality"]["exploitability"], "maximum": 0.10,
                "oracle_sha256": metadata["sha256"], "teacher_sha256": quality["sha256"],
                "action_labels_sha256": quality["action_labels_sha256"],
                "infosets": len(states), "criterion": "User-approved full-tree exploitability <= 0.10"}
        write_json(args.root / "gate.json", gate)
        print(json.dumps({"astra_quality_gate": gate}), flush=True)
        if gate["passed"]:
            collect_stage("summaries", states, metadata, args.root, workers=args.workers,
                          recover_budget=args.recover_budget)
            full_teacher = args.root / "teacher.jsonl"
            if not full_teacher.exists():
                export_teacher(states, metadata, args.root, full_teacher, summaries=True)
        write_json(args.root / "complete.json", {"gate": gate,
                   "followups": "eligible" if gate["passed"] else "skipped: Astra does not meet the solver threshold"})


if __name__ == "__main__":
    main()
