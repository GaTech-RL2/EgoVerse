"""Count annotated frames per dataset, WITHOUT double counting.

Augmentation writes many entries (paraphrases) over the SAME or overlapping
[start_idx, end_idx) spans; per episode and per annotation key the spans are
UNIONED (merged intervals) so each frame counts once.

Episodes are bucketed by structure (matches what training sees):
  - sort:      annotations_subtask present and != annotations_task (real split)
  - pp (copy): annotations_subtask present and == annotations_task (eva regime)
  - pp:        task key only (aria pick_place)

Two input modes:
  --data-config NAME   hydra data yaml (e.g. sort_pp_subtask_6d); instantiates
                       the configured datasets — filters included — so the
                       summary covers exactly the episodes training/val sees.
                       --split train|valid|both picks the groups (default both).
  --dataset-dir D      raw zarr dataset dir(s); every episode in the dir,
                       regardless of any config filters.

Usage:
    python -m egomimic.scripts.language_process.count_annotated_frames \
        --data-config sort_pp_subtask_6d [--split train]
    python -m egomimic.scripts.language_process.count_annotated_frames \
        --dataset-dir /path/ds1 [--dataset-dir /path/ds2 ...]
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


def count_episode(agg, name, path, unannotated=None):
    """Accumulate one episode zarr at ``path`` into agg[(name, bucket)]."""
    import zarr

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
        if unannotated is not None:
            unannotated.append(os.path.basename(path.rstrip("/")))
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", help="hydra data yaml name (e.g. sort_pp_subtask_6d)")
    ap.add_argument("--split", choices=("train", "valid", "both"), default="both",
                    help="which dataset groups of --data-config to count")
    ap.add_argument("--dataset-dir", action="append", default=[],
                    help="raw zarr dataset dir (repeatable); ignores config filters")
    args = ap.parse_args()
    if not args.data_config and not args.dataset_dir:
        ap.error("need --data-config and/or --dataset-dir")

    agg = defaultdict(lambda: defaultdict(int))  # (dataset, bucket) -> counters
    unannotated = defaultdict(list)  # dataset -> [episode hash]

    for root in args.dataset_dir:
        name = os.path.basename(root.rstrip("/"))
        for ep in sorted(os.listdir(root)):
            path = os.path.join(root, ep)
            if os.path.isdir(path):
                count_episode(agg, name, path, unannotated[name])

    if args.data_config:
        import hydra

        from egomimic.utils.hydra_utils import load_config

        cfg = load_config(f"data/{args.data_config}")
        splits = ("train", "valid") if args.split == "both" else (args.split,)
        for split in splits:
            groups = cfg.get(f"{split}_datasets") or {}
            for emb in groups:
                ds = hydra.utils.instantiate(groups[emb])
                name = f"{split}/{emb}"
                for leaf in ds.datasets.values():
                    count_episode(agg, name, str(leaf.episode_path), unannotated[name])

    print(f"{'dataset':26} {'bucket':11} {'eps':>4} {'total_frm':>10} "
          f"{'task_frm':>9} {'task%':>6} {'subtask_frm':>11} {'subtask%':>8}")
    for (name, bucket), a in sorted(agg.items()):
        tf = a["total_frames"] or 1
        print(f"{name:26} {bucket:11} {a['episodes']:>4} {a['total_frames']:>10,} "
              f"{a['task_frames']:>9,} {100*a['task_frames']/tf:>5.1f}% "
              f"{a['subtask_frames']:>11,} {100*a['subtask_frames']/tf:>7.1f}%")
    for name, eps in sorted(unannotated.items()):
        if eps:
            print(f"\n{name}: {len(eps)} unannotated episodes:")
            for ep in eps:
                print(f"  {ep}")


if __name__ == "__main__":
    main()
