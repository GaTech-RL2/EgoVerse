"""results/*.jsonl -> markdown tables (medians over repeats)."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

FORMATS = [
    "zarr",
    "zarr_fast",
    "lance",
    "lerobot",
    "lerobot_stream",
    "lerobot_unpacked",
]
PATTERNS = ["chunk1", "numeric", "frames8"]


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Runs before the packed layout wrote format=lerobot for the one-column-per-key layout.
        # The fake-length runs (rco3_fblen) are packed under a plain lerobot/ dir.
        if (
            r.get("format") == "lerobot"
            and "root" in r
            and not any(s in r["root"] for s in ("lerobot_packed", "rco3_fblen"))
        ):
            r["format"] = "lerobot_unpacked"
        rows.append(r)
    return rows


def fmt(x, nd=1):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return f"{x:,}"


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def throughput(rows: list[dict], title: str) -> str:
    rows = [r for r in rows if r.get("label") == "tput"]
    if not rows:
        return f"## {title}\n\n(no results)\n"
    by = defaultdict(list)
    for r in rows:
        by[(r["format"], r["pattern"], r["workers"])].append(r)
    workers = sorted({r["workers"] for r in rows})
    best_rows, grid_rows = [], []
    for p in PATTERNS:
        for f in FORMATS:
            cells, best = [], None
            for w in workers:
                rs = by.get((f, p, w), [])
                if not rs:
                    cells.append("–")
                    continue
                sps = median(r["samples_per_s"] for r in rs)
                cells.append(f"{sps:,.0f} (n={len(rs)})")
                if best is None or sps > best[0]:
                    best = (sps, w, rs)
            grid_rows.append([p, f, *cells])
            if best:
                sps, w, rs = best
                hit = [
                    r["decoder_hit_rate"]
                    for r in rs
                    if r.get("decoder_hit_rate") is not None
                ]
                best_rows.append(
                    [
                        p,
                        f,
                        f"{sps:,.0f}",
                        w,
                        fmt(median(r["batch_wait_p95_ms"] for r in rs), 0),
                        fmt(median(r["peak_rss_mb"] for r in rs) / 1024, 1),
                        fmt(median(hit), 2) if hit else "–",
                    ]
                )
    return (
        f"## {title}\n\nBest worker count per format (median of repeats):\n\n"
        + table(
            [
                "pattern",
                "format",
                "samples/s",
                "workers",
                "p95 batch wait ms",
                "peak RSS GB",
                "decoder hit rate",
            ],
            best_rows,
        )
        + "\n\nAll worker counts (median samples/s):\n\n"
        + table(["pattern", "format", *[f"w={w}" for w in workers]], grid_rows)
        + "\n"
    )


def nocache(rows: list[dict]) -> str:
    rows = [r for r in rows if r.get("label") == "nocache"]
    if not rows:
        return ""
    by = defaultdict(list)
    for r in rows:
        by[r["format"]].append(r)
    out = []
    for f in FORMATS:
        rs = by.get(f)
        if rs:
            out.append(
                [
                    f,
                    f"{median(r['samples_per_s'] for r in rs):,.0f}",
                    len(rs),
                    fmt(median(r["peak_rss_mb"] for r in rs) / 1024, 1),
                ]
            )
    return (
        "## Local, decoder cache size 1 (chunk1, 12 workers)\n\n"
        "Approximates a 100k-episode run, where a worker almost never reuses an open video decoder.\n\n"
        + table(["format", "samples/s", "n", "peak RSS GB"], out)
        + "\n"
    )


def startup(rows: list[dict], title: str) -> str:
    if not rows:
        return f"## {title}\n\n(no results)\n"
    by = defaultdict(list)
    for r in rows:
        variant = r["format"] + (
            ""
            if not r["format"].startswith("lerobot")
            else (" cold" if r.get("cold") else " warm")
        )
        by[(r["label"], r["location"], variant, r["episodes"])].append(r)
    out = []
    for (label, loc, variant, _), rs in sorted(by.items()):
        m = {
            k: median(r[k] for r in rs)
            for k in (
                "resolve_s",
                "construct_s",
                "worker_spawn_s",
                "startup_total_s",
                "peak_rss_mb",
            )
        }
        m["first_only_s"] = median(r["first_batch_s"] - r["worker_spawn_s"] for r in rs)
        out.append(
            [
                label.replace("startup_", "").replace("_nfs", ""),
                loc,
                variant,
                fmt(rs[0]["episodes"]),
                fmt(m["resolve_s"], 2),
                fmt(m["construct_s"], 2),
                fmt(m["worker_spawn_s"], 2),
                fmt(m["first_only_s"], 2),
                fmt(m["startup_total_s"], 2),
                fmt(m["peak_rss_mb"] / 1024, 1),
                len(rs),
            ]
        )
    return (
        f"## {title}\n\n"
        "resolve = finding episodes and building per-episode state; construct = MultiDataset / HF Arrow load; "
        "spawn = DataLoader start until the last worker is ready; first batch = after workers are ready; "
        "total = resolve + construct + time to first batch (each column is its own median). "
        "Peak RSS sums RSS over the process tree, so pages shared between workers count once per worker.\n\n"
        + table(
            [
                "N fake",
                "location",
                "format",
                "episodes",
                "resolve s",
                "construct s",
                "spawn s",
                "first batch s",
                "total s",
                "peak RSS GB",
                "n",
            ],
            out,
        )
        + "\n"
    )


def by_label(rows: list[dict], title: str, note: str = "") -> str:
    """Median samples/s per (label, format, pattern, workers) for ablation-style runs."""
    if not rows:
        return ""
    by = defaultdict(list)
    for r in rows:
        by[(r["label"], r["format"], r["pattern"], r["workers"])].append(r)
    out = []
    for (label, f, p, w), rs in sorted(
        by.items(), key=lambda kv: (PATTERNS.index(kv[0][2]), kv[0])
    ):
        hit = [
            r["decoder_hit_rate"] for r in rs if r.get("decoder_hit_rate") is not None
        ]
        out.append(
            [
                p,
                label,
                f,
                w,
                f"{median(r['samples_per_s'] for r in rs):,.0f}",
                fmt(median(hit), 2) if hit else "–",
                len(rs),
            ]
        )
    return (
        f"## {title}\n\n{note}"
        + table(
            [
                "pattern",
                "variant",
                "format",
                "workers",
                "samples/s",
                "decoder hit rate",
                "n",
            ],
            out,
        )
        + "\n"
    )


def storage(rows: list[dict]) -> str:
    if not rows:
        return "## Storage\n\n(no results)\n"
    out = [
        [
            r["format"],
            r["variant"],
            fmt(r["frames"]),
            fmt(r["numeric_mb"]),
            fmt(r["video_mb"]),
            fmt(r["metadata_mb"]),
            fmt(r["total_mb"]),
            fmt(r["numeric_bytes_per_frame"], 0),
            fmt(r["video_bytes_per_frame"], 0),
            fmt(r["total_bytes_per_frame"], 0),
        ]
        for r in rows
    ]
    return (
        "## Storage (48 pilot recordings)\n\n"
        + table(
            [
                "format",
                "variant",
                "frames",
                "numeric MB",
                "video/image MB",
                "metadata MB",
                "total MB",
                "numeric B/frame",
                "video B/frame",
                "total B/frame",
            ],
            out,
        )
        + "\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    r = args.results
    local = load(r / "local_throughput.jsonl")
    s3 = load(r / "s3_throughput.jsonl")
    parts = [
        "# LeRobot v3.0 vs zarr vs Lance\n",
        storage(load(r / "storage.jsonl")),
        throughput(local, "Throughput, local SATA SSD (walle)"),
        nocache(local),
        throughput(
            s3, "Throughput, MinIO (S3) on clippy (SATA SSD), MinIO on 4 of the 16 CPUs"
        ),
        by_label(
            [
                x
                for x in s3
                if x["format"] in ("lance", "zarr_fast")
                and x["workers"] == 8
                and x["label"] in ("tput", "per_sample")
            ],
            "Batched fetch vs per-sample reads over MinIO (8 workers)",
            "per_sample removes MultiDataset.__getitems__, so the loader reads one sample at a time like zarr and LeRobot.\n\n",
        ),
        by_label(
            [x for x in local if x["label"] == "per_sample"]
            + [
                x
                for x in local
                if x["format"] == "zarr_fast"
                and x["workers"] == 12
                and x["label"] == "tput"
                and x["pattern"] != "frames8"
            ],
            "zarr_fast local: batched vs per-sample (12 workers)",
        ),
        by_label(
            load(r / "cache_ablation.jsonl"),
            "Lance decoder cache 16 vs 100 (local chunk1, 12 workers)",
        ),
        by_label(
            load(r / "scale_throughput.jsonl"),
            "Steady-state throughput over 10k episodes (cheetah local, 8 workers, samples drawn from all episodes)",
        ),
        startup(
            load(r / "startup.jsonl"),
            "Startup at scale (cheetah, SATA SSD; real + N fake episodes)",
        ),
        startup(
            load(r / "startup_nfs.jsonl"),
            "Startup at 100k from NFS (/coc/flash7/scratch, production check)",
        ),
        startup(
            load(r / "startup_len.jsonl"),
            "Startup at 10k vs fake episode length (kitt local; N fake = 10000_f<frames per fake>)",
        ),
    ]
    args.out.write_text("\n".join(parts))
    print(args.out.read_text())


if __name__ == "__main__":
    main()
