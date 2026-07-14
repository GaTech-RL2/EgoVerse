"""Count annotated frames per dataset, WITHOUT double counting.

Augmentation writes many entries (paraphrases) over the SAME or overlapping
[start_idx, end_idx) spans; per episode and per annotation key the spans are
UNIONED (merged intervals) so each frame counts once.

Episodes are bucketed by structure (matches what training sees):
  - sort:      annotations_subtask present and != annotations_task (real split)
  - pp (copy): annotations_subtask present and == annotations_task (eva regime)
  - pp:        task key only (aria pick_place)

Usage: python count_annotated_frames.py --dataset-dir D [--dataset-dir D2 ...]
"""

import argparse
import json
import os
from collections import defaultdict


def merge_intervals(ivals):
    """[(s,e), ...) half-open -> disjoint sorted union (no double counting)."""
    if not ivals:
        return []
    ivals = sorted(ivals)
    out = [list(ivals[0])]
    for s, e in ivals[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def spans_of(g, key):
    if key not in set(g.array_keys()):
        return None
    out = []
    texts = set()
    for x in g[key][:]:
        try:
            d = json.loads(x.decode() if isinstance(x, (bytes, bytearray)) else str(x))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        s, e = int(d.get("start_idx", -1)), int(d.get("end_idx", -1))
        if 0 <= s < e:
            out.append((s, e))
            texts.add(d.get("text", ""))
    return out, texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", action="append", required=True)
    args = ap.parse_args()

    import zarr

    agg = defaultdict(lambda: defaultdict(int))  # (dataset, bucket) -> counters
    for root in args.dataset_dir:
        name = os.path.basename(root.rstrip("/"))
        for ep in sorted(os.listdir(root)):
            path = os.path.join(root, ep)
            if not os.path.isdir(path):
                continue
            g = zarr.open(path, mode="r")
            total = int(g.attrs.get("total_frames") or 0)
            task = spans_of(g, "annotations_task")
            sub = spans_of(g, "annotations_subtask")
            if sub is not None and task is not None and sub[1] != task[1]:
                bucket = "sort"
            elif sub is not None:
                bucket = "pp_evacopy"
            elif task is not None:
                bucket = "pp"
            else:
                bucket = "unannotated"
            a = agg[(name, bucket)]
            a["episodes"] += 1
            a["total_frames"] += total
            for label, sp in (("task", task), ("subtask", sub)):
                if sp is None:
                    continue
                spans, _texts = sp
                merged = merge_intervals([(s, min(e, total or e)) for s, e in spans])
                a[f"{label}_frames"] += sum(e - s for s, e in merged)
                a[f"{label}_raw_spans"] += len(spans)

    print(f"{'dataset':26} {'bucket':11} {'eps':>4} {'total_frm':>10} "
          f"{'task_frm':>9} {'task%':>6} {'subtask_frm':>11} {'subtask%':>8}")
    for (name, bucket), a in sorted(agg.items()):
        tf = a["total_frames"] or 1
        print(f"{name:26} {bucket:11} {a['episodes']:>4} {a['total_frames']:>10,} "
              f"{a['task_frames']:>9,} {100*a['task_frames']/tf:>5.1f}% "
              f"{a['subtask_frames']:>11,} {100*a['subtask_frames']/tf:>7.1f}%")


if __name__ == "__main__":
    main()
