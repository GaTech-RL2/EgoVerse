"""Inject each episode's SQL ``task`` string as its ``annotations_task``.

For episodes with no dense-language annotations (e.g. the robot sort
collections), the registry's task name ("sort the eating utensils into
containers") is itself a usable task-level instruction. Writes ONE full-span
entry under ``annotations_task`` per episode; no ``annotations_subtask`` (no
decomposition GT exists — subtask decode is evaluated qualitatively).

Usage: python inject_task_annotations.py --dataset-dir <root> [--overwrite]
"""

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import zarr
    from sqlalchemy import text

    from egomimic.rldb.zarr.zarr_writer import ZarrWriter
    from egomimic.utils.aws.aws_data_utils import load_env
    from egomimic.utils.aws.aws_sql import create_default_engine

    load_env()
    eng = create_default_engine()
    eps = sorted(
        d for d in os.listdir(args.dataset_dir)
        if os.path.isdir(os.path.join(args.dataset_dir, d))
    )
    hashes = [e[:-5] if e.endswith(".zarr") else e for e in eps]
    task_of = {}
    with eng.connect() as c:
        q = text(
            "SELECT episode_hash, task FROM app.episodes "
            "WHERE episode_hash = ANY(:h)"
        )
        for r in c.execute(q, {"h": hashes}):
            task_of[r[0]] = r[1] or ""

    n_ok = n_skip = n_miss = 0
    for ep, h in zip(eps, hashes):
        path = os.path.join(args.dataset_dir, ep)
        task = task_of.get(h, "").strip()
        if not task:
            print(f"[MISS] {ep} — no task in registry")
            n_miss += 1
            continue
        g = zarr.open(path, mode="r")
        if "annotations_task" in set(g.array_keys()) and not args.overwrite:
            print(f"[SKIP] {ep}")
            n_skip += 1
            continue
        total = int(g.attrs.get("total_frames") or 0)
        writer = ZarrWriter(episode_path=path, verbose=False)
        writer.append_annotations(
            annotation_key="annotations_task",
            annotations=[(task, 0, max(total, 1))],
            mode="w",
        )
        print(f"[OK] {ep} <- {task!r} span [0,{total})")
        n_ok += 1
    print(f"[DONE] {n_ok} injected, {n_skip} skipped, {n_miss} missing")


if __name__ == "__main__":
    main()
