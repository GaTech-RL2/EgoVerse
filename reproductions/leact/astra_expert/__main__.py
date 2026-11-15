"""Collect actions first, then explanations; full collection requires the Leduc gate."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from ..data import fingerprint, load_oracle, write_json
from .client import MODEL, PROMPT_VERSION, SETTINGS, call
from .data import collect_records, export_teacher, leduc_gate, pilot_states, read_call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["actions", "summaries", "export"])
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="Exactly four fixed round/player states, no training export")
    parser.add_argument("--leduc-results", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-summaries", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be 1 to 4")
    states, metadata = load_oracle(args.oracle)
    if args.preflight:
        if args.stage == "export":
            parser.error("Preflight records cannot be exported as a full teacher")
        states, gate = pilot_states(states), {"passed": False, "scope": "API preflight only"}
    else:
        if args.leduc_results is None:
            parser.error("Full collection requires --leduc-results after the current run works")
        gate = leduc_gate(json.loads(args.leduc_results.read_text()), metadata["sha256"])
    protocol = {"scope": "four-state API preflight" if args.preflight else "full Leduc Astra teacher",
                "oracle_sha256": metadata["sha256"], "state_ids": [s["state_id"] for s in states],
                "model": MODEL, "prompt_version": PROMPT_VERSION, "settings": SETTINGS,
                "gate": gate}
    spec = args.root / "collection.json"
    if spec.exists() and json.loads(spec.read_text()) != protocol:
        raise ValueError("Collection protocol changed; choose a new directory")
    write_json(spec, protocol)
    if args.stage == "export":
        if args.output is None:
            parser.error("export requires --output")
        print(json.dumps(export_teacher(states, metadata, args.root, args.output, summaries=args.include_summaries)))
        return
    # Finish and validate the action stage across every requested state before
    # allowing a single explanation call. No decisions are regenerated later.
    if args.stage == "summaries":
        collect_records(states, metadata, args.root)

    def generate(state):
        committed = None
        if args.stage == "summaries":
            committed = read_call(args.root / "actions" / (state["state_id"] + ".json"), state, metadata)["result"]
        record = call(state, metadata["rules"], args.root / args.stage / (state["state_id"] + ".json"),
                      oracle_sha256=metadata["sha256"], committed=committed)
        return {"state_id": state["state_id"], "accepted": record["accepted"],
                "response_model": record["response"]["model"], "seconds": record["latency_seconds"],
                "usage": record["response"].get("usage", {}), "record_sha256": fingerprint(record)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(generate, states):
            print(json.dumps(result), flush=True)
    labels = collect_records(states, metadata, args.root, summaries=args.stage == "summaries")
    write_json(args.root / (args.stage + "-complete.json"),
               {"states": len(labels), "labels_sha256": fingerprint(labels), "collection_sha256": fingerprint(protocol)})


if __name__ == "__main__":
    main()
