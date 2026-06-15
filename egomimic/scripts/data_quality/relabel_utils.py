"""Pure-Python glue for the segment-relabeling workflow (NO LLM calls).

The actual relabeling is done by a Claude Code **workflow** whose agents read the
chunk files this script writes and return one canonical task label per segment.
This module only does deterministic file work — workflow scripts have no
filesystem access, so this is the on-disk half of the pipeline:

  split     segments.json         -> seg_chunk_000.json, seg_chunk_001.json, ...
  merge     relabel_chunk_*.json  -> relabel_map.json   (validated vs taxonomy)
  validate  relabel_map.json      -> exit non-zero unless it matches segments+taxonomy

``relabel_map.json`` is the sidecar consumed by ``ZarrDataset`` via
``annotation_override_path`` on the data config:
    {"<episode_hash>": [{"start_idx", "end_idx", "text": <label>, "orig_text"}, ...]}

The workflow agents emit per-chunk **label files** mapping hash -> ordered labels
(one per segment, in the segment order from the chunk):
    {"<episode_hash>": ["fold_tshirt", "smooth_tshirt", ...]}
An {"episodes": [{"episode_hash", "labels": [...]}]} shape is also accepted.

Usage:
    python -m egomimic.scripts.data_quality.relabel_utils split \
        --segments segments.json --out-dir relabel_work/ --chunk-size 25
    # (run the relabel workflow over relabel_work/seg_chunk_*.json -> relabel_work/relabel_chunk_*.json)
    python -m egomimic.scripts.data_quality.relabel_utils merge \
        --segments segments.json --labels-dir relabel_work/ \
        --taxonomy taxonomy.json --out relabel_map.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_taxonomy_labels(path: str | Path) -> set[str]:
    """Set of allowed label strings from a taxonomy file.

    Accepts ``["label", ...]``, ``[{"label", "description"}, ...]`` or
    ``{"labels": [...]}``.
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("labels", [])
    labels: set[str] = set()
    for item in data:
        if isinstance(item, str):
            labels.add(item)
        elif isinstance(item, dict) and item.get("label"):
            labels.add(item["label"])
    if not labels:
        raise ValueError(f"No labels found in taxonomy: {path}")
    return labels


def _load_label_chunks(
    labels_dir: str | Path,
) -> tuple[dict[str, list[str]], list[str]]:
    """Merge every ``relabel_chunk_*.json`` into one ``{hash: [labels]}`` map."""
    files = sorted(glob.glob(str(Path(labels_dir) / "relabel_chunk_*.json")))
    if not files:
        raise SystemExit(f"No relabel_chunk_*.json found in {labels_dir}")
    out: dict[str, list[str]] = {}
    for fp in files:
        data = json.loads(Path(fp).read_text())
        if isinstance(data, dict) and "episodes" in data:
            for e in data["episodes"]:
                out[e["episode_hash"]] = list(e["labels"])
        elif isinstance(data, dict):
            for h, labels in data.items():
                out[h] = list(labels)
        else:
            raise ValueError(f"Unexpected chunk shape in {fp} (want dict)")
    return out, files


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def split(segments_path: str | Path, out_dir: str | Path, chunk_size: int) -> list[str]:
    """Chunk ``segments.json`` into ``seg_chunk_XXX.json`` files for the workflow."""
    records = json.loads(Path(segments_path).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i in range(0, len(records), chunk_size):
        chunk = records[i : i + chunk_size]
        cp = out_dir / f"seg_chunk_{len(paths):03d}.json"
        cp.write_text(json.dumps(chunk, indent=2))
        paths.append(str(cp.resolve()))
    manifest = out_dir / "chunks_manifest.json"
    manifest.write_text(json.dumps(paths, indent=2))
    print(
        f"Wrote {len(paths)} chunks ({len(records)} episodes, ~{chunk_size}/chunk) "
        f"-> {out_dir}"
    )
    print(f"Manifest (pass as workflow args): {manifest}")
    return paths


def merge(
    segments_path: str | Path,
    labels_dir: str | Path,
    taxonomy_path: str | Path,
    out_path: str | Path,
    strict: bool = False,
) -> dict[str, list[dict]]:
    """Assemble + validate ``relabel_map.json`` from the per-chunk label files.

    Each output span keeps the original ``start_idx``/``end_idx`` (boundaries are
    authoritative from the source episode), sets ``text`` to the canonical label,
    and preserves ``orig_text``. On any per-episode problem (missing labels,
    count mismatch, or a label outside the taxonomy) the episode falls back to its
    original annotation text (``fallback: true``) unless ``strict``.
    """
    records = json.loads(Path(segments_path).read_text())
    taxonomy = load_taxonomy_labels(taxonomy_path)
    label_map, files = _load_label_chunks(labels_dir)

    relabel: dict[str, list[dict]] = {}
    n_seg = n_relabeled = n_fallback = 0
    fallbacks: list[str] = []
    for rec in records:
        h = rec["episode_hash"]
        segs = rec["segments"]
        n_seg += len(segs)
        labels = label_map.get(h)

        if labels is None:
            reason = "no labels produced"
        elif len(labels) != len(segs):
            reason = f"label count {len(labels)} != {len(segs)} segments"
        else:
            unknown = sorted({lbl for lbl in labels if lbl not in taxonomy})
            reason = f"labels not in taxonomy: {unknown}" if unknown else ""

        if reason:
            if strict:
                raise SystemExit(f"{h}: {reason}")
            fallbacks.append(f"{h}: {reason}")
            spans = [
                {
                    "start_idx": s["start_idx"],
                    "end_idx": s["end_idx"],
                    "text": s["text"],
                    "orig_text": s["text"],
                    "fallback": True,
                }
                for s in segs
            ]
            n_fallback += len(spans)
        else:
            spans = [
                {
                    "start_idx": s["start_idx"],
                    "end_idx": s["end_idx"],
                    "text": lbl,
                    "orig_text": s["text"],
                }
                for s, lbl in zip(segs, labels, strict=True)
            ]
            n_relabeled += len(spans)

        if spans:
            relabel[h] = spans

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(relabel, indent=2))

    print(f"Merged {len(files)} chunk file(s) -> {out_path}")
    print(
        f"Episodes: {len(relabel)} | segments: {n_seg} | "
        f"relabeled: {n_relabeled} | fallback: {n_fallback}"
    )
    for line in fallbacks:
        print("  FALLBACK", line)
    return relabel


def validate(
    relabel_path: str | Path,
    segments_path: str | Path,
    taxonomy_path: str | Path | None = None,
) -> None:
    """Exit non-zero unless ``relabel_map.json`` is consistent with the source.

    Checks: every episode is in ``segments.json``, span count == segment count,
    span boundaries are unchanged, labels are non-empty, and (if a taxonomy is
    given) every non-fallback label is in the taxonomy.
    """
    relabel = json.loads(Path(relabel_path).read_text())
    records = {
        r["episode_hash"]: r for r in json.loads(Path(segments_path).read_text())
    }
    taxonomy = load_taxonomy_labels(taxonomy_path) if taxonomy_path else None

    problems: list[str] = []
    for h, spans in relabel.items():
        rec = records.get(h)
        if rec is None:
            problems.append(f"{h}: not present in segments file")
            continue
        segs = rec["segments"]
        if len(spans) != len(segs):
            problems.append(f"{h}: {len(spans)} spans != {len(segs)} segments")
            continue
        for sp, s in zip(spans, segs, strict=True):
            if sp["start_idx"] != s["start_idx"] or sp["end_idx"] != s["end_idx"]:
                problems.append(f"{h}: span boundary changed")
            if not str(sp.get("text", "")).strip():
                problems.append(f"{h}: empty label")
            if (
                taxonomy is not None
                and not sp.get("fallback")
                and sp["text"] not in taxonomy
            ):
                problems.append(f"{h}: label {sp['text']!r} not in taxonomy")

    if problems:
        for p in problems[:50]:
            print("INVALID:", p)
        raise SystemExit(f"{len(problems)} validation problem(s)")
    print(
        f"OK: {len(relabel)} episodes valid against {segments_path}"
        + (" + taxonomy" if taxonomy is not None else "")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split", help="Chunk segments.json for the relabel workflow.")
    sp.add_argument("--segments", type=Path, required=True)
    sp.add_argument("--out-dir", type=Path, required=True)
    sp.add_argument("--chunk-size", type=int, default=25)

    mp = sub.add_parser("merge", help="Assemble + validate relabel_map.json.")
    mp.add_argument("--segments", type=Path, required=True)
    mp.add_argument("--labels-dir", type=Path, required=True)
    mp.add_argument("--taxonomy", type=Path, required=True)
    mp.add_argument("--out", type=Path, required=True)
    mp.add_argument(
        "--strict",
        action="store_true",
        help="Error on any problem instead of falling back to original text.",
    )

    vp = sub.add_parser("validate", help="Check an existing relabel_map.json.")
    vp.add_argument("--relabel-map", type=Path, required=True)
    vp.add_argument("--segments", type=Path, required=True)
    vp.add_argument("--taxonomy", type=Path, default=None)

    a = p.parse_args()
    if a.cmd == "split":
        split(a.segments, a.out_dir, a.chunk_size)
    elif a.cmd == "merge":
        merge(a.segments, a.labels_dir, a.taxonomy, a.out, strict=a.strict)
    elif a.cmd == "validate":
        validate(a.relabel_map, a.segments, a.taxonomy)


if __name__ == "__main__":
    main()
