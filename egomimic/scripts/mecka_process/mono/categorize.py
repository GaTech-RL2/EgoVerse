"""Assign each episode in the Mecka mono master JSON to one of four tasks.

Mecka collected ~5 h per operator per task, so the category comes from the
operator's dominant task family; operators with two folding quotas (~10 h) are
split per episode by whether the caption mentions t-shirts. Episodes whose own
task_id disagrees with the operator's category are kept in the manifest but
marked off_task.

  python -m egomimic.scripts.mecka_process.mono.categorize <master.json>
"""

import collections
import csv
import json
import re
import sys

from egomimic.scripts.mecka_process.mono.private_ids import private_dir

OUT = private_dir() / "manifest.csv"

TSHIRT = re.compile(r"t[- ]?shirt|\btee\b", re.I)


def family(task_id: str) -> str:
    if re.search(r"cabl|wire|cord|electr|coil|adapter|remote", task_id):
        return "organize_cables"
    if re.search(r"hang", task_id):
        return "hang_clothes"
    if re.search(r"fold|shirt", task_id):
        return "fold"
    return "other"


def main() -> None:
    episodes = json.load(open(sys.argv[1]))["episodes"]
    hours = collections.defaultdict(collections.Counter)
    for e in episodes:
        hours[e["user_id"]][family(e["task_id"])] += e["duration"] / 3600

    op_family = {}
    op_tshirt_rate = {}
    for uid, c in hours.items():
        known = {k: v for k, v in c.items() if k != "other"}
        op_family[uid] = max(known, key=known.get) if known else "other"
        eps = [
            e
            for e in episodes
            if e["user_id"] == uid and family(e["task_id"]) == "fold"
        ]
        if eps:
            op_tshirt_rate[uid] = sum(
                bool(TSHIRT.search(e["task_description"])) for e in eps
            ) / len(eps)

    rows = []
    for e in episodes:
        uid, fam = e["user_id"], family(e["task_id"])
        cat = op_family[uid]
        if cat == "fold":
            rate = op_tshirt_rate.get(uid, 0.0)
            if rate > 0.95:
                cat = "fold_tshirt"
            elif rate < 0.05:
                cat = "fold_no_tshirt"
            else:
                cat = (
                    "fold_tshirt"
                    if TSHIRT.search(e["task_description"])
                    else "fold_no_tshirt"
                )
        off_task = fam != op_family[uid]
        rows.append(
            {
                "episode_id": e["episode_id"],
                "category": cat,
                "off_task": int(off_task),
                "task_id": e["task_id"],
                "user_id": uid,
                "duration": e["duration"],
                "task_description": e["task_description"],
            }
        )

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = collections.defaultdict(lambda: [0, 0.0, 0, 0.0, set()])
    for r in rows:
        s = summary[r["category"]]
        i = 2 if r["off_task"] else 0
        s[i] += 1
        s[i + 1] += r["duration"] / 3600
        s[4].add(r["user_id"])
    print(
        f"{'category':16s} {'eps':>5s} {'hours':>6s} {'ops':>4s} | off_task eps/hours"
    )
    for cat, (n, h, no, ho, ops) in sorted(summary.items()):
        print(f"{cat:16s} {n:5d} {h:6.1f} {len(ops):4d} | {no:4d} {ho:5.1f}")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
