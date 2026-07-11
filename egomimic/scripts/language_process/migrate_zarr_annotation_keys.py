"""Migrate IN-ZARR level-tagged annotations to the role-keyed scheme.

Zarr-side counterpart of ``correct_annotation_keys.py`` (which migrates the
S3 JSONs): for every episode zarr under ``--dataset-dir``, read the legacy
``annotations`` array and inject role-named arrays WITHOUT re-running any
LLM generation:

    annotations_task      <- level == "high" entries (sort goals), or ALL
                             entries when the episode has none (pick_place:
                             the instruction IS the task conditioning)
    annotations_subtask   <- level != "high" entries for sort episodes; for
                             eva pick_place episodes an identical copy of the
                             task list; absent for human pick_place

Entries are written as plain ``{text, start_idx, end_idx}`` (level dropped —
role lives in the key name). The legacy ``annotations`` array is left
untouched for older configs. Eva-vs-human comes from the episode's own
``embodiment`` attr — no SQL needed.

Usage:
    python migrate_zarr_annotation_keys.py --dataset-dir <root> \
        [--episode-hash H | --limit N] [--dry-run] [--overwrite]
"""

import argparse
import json
import os

from egomimic.scripts.language_process.correct_annotation_keys import split_payload


def _decode(entry):
    if isinstance(entry, (bytes, bytearray)):
        return json.loads(entry.decode("utf-8"))
    return json.loads(str(entry))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--episode-hash", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite role keys even when annotations_task already exists.",
    )
    args = parser.parse_args()

    import zarr

    from egomimic.rldb.zarr.zarr_writer import ZarrWriter

    root = args.dataset_dir
    eps = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    if args.episode_hash is not None:
        eps = [e for e in eps if e.startswith(args.episode_hash)]
    if args.limit is not None:
        eps = eps[: args.limit]
    print(f"[INFO] {len(eps)} episodes under {root}")

    n_sort = n_pp = n_eva_copy = n_skip = n_noann = 0
    for ep in eps:
        path = os.path.join(root, ep)
        g = zarr.open(path, mode="r")
        arrays = set(g.array_keys())
        if "annotations" not in arrays:
            print(f"[NOANN] {ep} — no legacy annotations array")
            n_noann += 1
            continue
        if "annotations_task" in arrays and not args.overwrite:
            print(f"[SKIP] {ep} — annotations_task already present")
            n_skip += 1
            continue

        payload = []
        for x in g["annotations"][:]:
            try:
                d = _decode(x)
            except Exception:
                continue
            if isinstance(d, dict):
                payload.append(d)
        emb = str(g.attrs.get("embodiment", ""))
        subtask_copy = emb.startswith("eva")
        keyed = split_payload(payload, subtask_copy)
        is_sort = any(e.get("level") == "high" for e in payload)
        n_sort += is_sort
        n_pp += not is_sort
        n_eva_copy += (not is_sort) and subtask_copy

        counts = {k.rsplit("_", 1)[-1]: len(v) for k, v in keyed.items()}
        tag = "sort" if is_sort else ("pp+evacopy" if subtask_copy else "pp")
        if args.dry_run:
            print(f"[DRY] {ep} ({tag}, emb={emb}) -> {counts}")
            continue

        writer = ZarrWriter(episode_path=path, verbose=False)
        for ann_key, entries in keyed.items():
            writer.append_annotations(
                annotation_key=ann_key,
                annotations=[
                    (e["text"], int(e["start_idx"]), int(e["end_idx"]))
                    for e in entries
                ],
                mode="w",
            )
        print(f"[OK] {ep} ({tag}, emb={emb}) -> {counts}")

    print(
        f"\n[DONE] {len(eps)} episodes: {n_sort} sort, {n_pp} pick_place "
        f"({n_eva_copy} eva subtask-copies), {n_skip} skipped, {n_noann} no-annotation"
    )


if __name__ == "__main__":
    main()
