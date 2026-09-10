"""Audit every uploaded episode's search-quality manifest using tar range reads."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .articulation_status import client


def read_manifest(c, record, cache):
    key = record["archive"].removeprefix("s3://rldb/")
    relative = Path(*Path(key).parts[-3:]).with_suffix("")
    cached = cache / relative / (record["archive_sha256"] + ".json")
    if cached.exists():
        return json.loads(cached.read_text())
    name = (str(relative) + "/episodes.jsonl").encode()
    size = record["archive_bytes"]
    for length in (262144, 1048576, 4194304):
        start = max(0, size - length)
        start -= start % 512
        response = c.get_object(Bucket="rldb", Key=key, Range=f"bytes={start}-{size-1}")
        payload = response["Body"].read()
        for offset in range(0, len(payload) - 511, 512):
            block = payload[offset : offset + 512]
            if block[:100].split(b"\0", 1)[0] != name:
                continue
            header = tarfile.TarInfo.frombuf(block, "utf-8", "strict")
            if not header.isfile() or offset + 512 + header.size > len(payload):
                continue
            raw = payload[offset + 512 : offset + 512 + header.size]
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(rows, separators=(",", ":")) + "\n")
            return rows
    raise ValueError("Cannot locate complete episode manifest in " + key)


def check_shard(c, record, cache):
    rows = read_manifest(c, record, cache)
    assert len(rows) == record["kept"], (
        "manifest episode count",
        record["archive"],
        len(rows),
    )
    assert len({r["episode"] for r in rows}) == len(rows)
    for row in rows:
        assert row["success"] and row["failure"] is None
        assert row["embodiment"] == record["embodiment"] and row["gap"] == record["gap"]
        assert record["seed0"] <= row["seed"] < record["seed0"] + record["max_attempts"]
        assert row["final_coverage"] >= 0.95 and row["peak_engaged_coverage"] >= 0.95
        assert (
            row["engaged"]
            and row["engaged_steps"] >= 20
            and row["carry_distance"] >= 30
        )
        assert (
            row["jerk_speed"] < 0.5
            and row["max_position_step"] < 6.0
            and row["max_angle_step"] < 0.1
        )
        assert row["angle_travel"] > 0.01
        if row["max_grip"] is not None:
            assert row["max_grip"] > 0
        if "mechanism_work_steps" in row:
            assert row["mechanism_work_steps"] >= 5
        if row["embodiment"] in ("triangle", "scoop"):
            assert row["contact_angle_travel"] >= 0.1
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    snapshot = json.loads(a.snapshot.read_text())
    c = client()
    cells = defaultdict(list)
    seeds = set()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for rows in pool.map(lambda r: check_shard(c, r, a.cache), snapshot["shards"]):
            for row in rows:
                assert row["seed"] not in seeds, ("duplicate reset seed", row["seed"])
                seeds.add(row["seed"])
                cells[(row["embodiment"], row["gap"])].append(row)
    assert len(seeds) == snapshot["durable_episodes"]
    summary = []
    for (emb, gap), rows in sorted(cells.items()):
        steps = [row["steps"] for row in rows]
        summary.append(
            dict(
                embodiment=emb,
                gap=gap,
                episodes=len(rows),
                min_final_coverage=min(r["final_coverage"] for r in rows),
                max_jerk_speed=max(r["jerk_speed"] for r in rows),
                min_carry_distance=min(r["carry_distance"] for r in rows),
                min_interaction_frames=min(r["engaged_steps"] for r in rows),
                median_frames=float(np.median(steps)),
                total_frames=sum(steps),
                fast_search_episodes=sum(
                    r.get("search_query_shortcuts", False) for r in rows
                ),
            )
        )
    digest = hashlib.sha256(
        json.dumps(sorted(seeds), separators=(",", ":")).encode()
    ).hexdigest()
    report = dict(
        passed=True,
        episodes_checked=len(seeds),
        shards_checked=len(snapshot["shards"]),
        all_reset_seeds_unique=True,
        sorted_seeds_sha256=digest,
        total_frames=sum(c["total_frames"] for c in summary),
        snapshot_checked_at=snapshot.get("checked_at"),
        cells=summary,
    )
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cells"}, indent=2))


if __name__ == "__main__":
    main()
