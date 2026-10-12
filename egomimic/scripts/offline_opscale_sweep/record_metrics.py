"""Append one completed checkpoint's metrics to its rung log, safely."""

import argparse
import fcntl
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=int, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.metrics.read_text())
    merged = {}
    for result in source["metrics"]:
        for key, value in result.items():
            if key in merged and merged[key] != value:
                raise ValueError(f"Conflicting duplicate validation metric: {key}")
            merged[key] = value
    required = (
        "seen_op_valid/Valid/Action_MSE",
        "unseen_op_valid/Valid/Action_MSE",
        "flagship_unseen_op_valid/Valid/Action_MSE",
    )
    for prefix in required:
        for key in (prefix, *(f"{prefix}/{part}" for part in ("early", "mid", "late"))):
            if key not in merged:
                raise ValueError(f"Missing required validation metric: {key}")
    for group in ("seen_op_valid", "unseen_op_valid", "flagship_unseen_op_valid"):
        key = f"{group}/Valid/open_loop_sim/Paired_MSE"
        if key not in merged:
            raise ValueError(f"Missing open-loop video metric: {key}")

    row = {
        "op": args.op,
        "epoch": args.epoch,
        "display_epoch": args.epoch + 1,
        "checkpoint": source["ckpt_path"],
        "metrics": merged,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a+", encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.seek(0)
        if any(
            json.loads(line)["epoch"] == args.epoch for line in file if line.strip()
        ):
            return
        file.seek(0, 2)
        file.write(json.dumps(row, sort_keys=True) + "\n")
        file.flush()
    print(
        json.dumps(
            {
                "op": args.op,
                "epoch": args.epoch,
                "seen_mse": merged[required[0]],
                "unseen_mse": merged[required[1]],
                "flagship_mse": merged[required[2]],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
