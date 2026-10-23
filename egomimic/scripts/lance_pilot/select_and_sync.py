"""Pick the Aria pilot recordings and sync their raw captures and current zarrs from R2."""

import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from egomimic.utils.aws.aws_sql import (
    create_default_engine,
    episode_hash_to_timestamp_ms,
)

RAW_PREFIX = "s3://rldb/raw_v2/aria"
# Semidense points and the open-loop trajectory are ~70% of MPS bytes and unused.
MPS_FILES = [
    "hand_tracking/*",
    "eye_gaze/*",
    "slam/closed_loop_trajectory.csv",
    "slam/online_calibration.jsonl",
    "slam/summary.json",
    "vrs_health_check*.json",
]
ALWAYS = [
    "2025-09-20-17-47-54-000000",
    "2025-10-29-22-06-00-411000",
    "2025-11-24-20-24-32-080000",
]
SKIP_TASKS = {"debug", "interview"}


def load_rows() -> pd.DataFrame:
    q = text(
        """select episode_hash, operator, lab, embodiment, scene, task, num_frames,
                  zarr_processed_path
           from app.episodes
           where zarr_processed_path like :p and not is_deleted
           order by episode_hash"""
    )
    with create_default_engine().connect() as c:
        df = pd.read_sql(q, c, params={"p": "%/aria/%"})
    df["epoch_ms"] = df.episode_hash.map(episode_hash_to_timestamp_ms)
    return df


def select(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Round-robin over (lab, task, embodiment) strata, preferring typical lengths."""
    lo, hi = df.num_frames.quantile([0.1, 0.9])
    pool = df[
        df.num_frames.between(lo, hi)
        & ~df.episode_hash.isin(ALWAYS)
        & ~df.task.isin(SKIP_TASKS)
    ]
    pool = pool.sample(frac=1.0, random_state=seed)
    groups = [g for _, g in pool.groupby(["lab", "task", "embodiment"])]
    groups.sort(key=len, reverse=True)
    picked = list(df[df.episode_hash.isin(ALWAYS)].index)
    i = 0
    while len(picked) < n and any(len(g) > i for g in groups):
        for g in groups:
            if len(picked) >= n:
                break
            if len(g) > i:
                picked.append(g.index[i])
        i += 1
    return df.loc[picked].sort_values("episode_hash").reset_index(drop=True)


def s5cmd_lines(sel: pd.DataFrame, out: Path) -> list[str]:
    lines = []
    for r in sel.itertuples():
        ms = r.epoch_ms
        raw = out / "raw"
        for name in (f"{ms}.vrs", f"{ms}.json", f"{ms}_metadata.json"):
            lines.append(f'cp -n "{RAW_PREFIX}/{name}" "{raw}/{name}"')
        for f in MPS_FILES:
            sub = os.path.dirname(f)
            dst = raw / f"mps_{ms}_vrs" / sub
            lines.append(f'cp -n "{RAW_PREFIX}/mps_{ms}_vrs/{f}" "{dst}/"')
        src = r.zarr_processed_path.rstrip("/")
        if not src.startswith("s3://"):
            src = f"s3://rldb/{src.lstrip('/')}"
        lines.append(f'sync "{src}/*" "{out / "zarr" / r.episode_hash}/"')
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("/coc/flash7/scratch/lance_pilot"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--numworkers", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--shard",
        type=int,
        default=0,
        help="download only recordings i with i %% nshards == shard",
    )
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    manifest = args.out / "manifest.csv"
    # The manifest is the pilot set; select only once so reruns and shards agree.
    if manifest.exists():
        sel = pd.read_csv(manifest)
    else:
        sel = select(load_rows(), args.n, args.seed)
        args.out.mkdir(parents=True, exist_ok=True)
        sel.to_csv(manifest, index=False)
    print(sel.groupby(["lab", "embodiment"]).size().to_string())
    print(
        f"{len(sel)} recordings, {sel.num_frames.sum():,} zarr frames, {sel.task.nunique()} tasks"
    )

    batch = args.out / f"s5cmd_batch_{args.shard}_of_{args.nshards}.txt"
    shard = sel.iloc[args.shard :: args.nshards]
    batch.write_text("\n".join(s5cmd_lines(shard, args.out)) + "\n")
    if args.dry_run:
        return
    env = os.environ.copy()
    env.update(
        AWS_ACCESS_KEY_ID=env["R2_ACCESS_KEY_ID"],
        AWS_SECRET_ACCESS_KEY=env["R2_SECRET_ACCESS_KEY"],
        AWS_REGION="auto",
        AWS_DEFAULT_REGION="auto",
    )
    # R2 rejects the session token that ~/.egoverse_env also exports.
    env.pop("AWS_SESSION_TOKEN", None)
    cmd = [
        "s5cmd",
        "--endpoint-url",
        env["R2_ENDPOINT_URL"],
        "--numworkers",
        str(args.numworkers),
        "run",
        str(batch),
    ]
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
