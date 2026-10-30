"""Dry-run planner for the zarr/raw -> Lance migration (docs/zarr_to_lance_migration.md).

Reads every non-deleted episode from SQL, decides for each whether it can be converted
and from what, and writes, under --out:

    manifest.parquet   one row per convertible episode: route (raw or zarr), the raw
                       URIs and converter or the source zarr URI, recording_id, source
                       namespace, target steps table, shard and work unit
    rejected.parquet   one row per blocked episode with every reason that blocks it
    inventory.csv      episodes / frames / estimated bytes per lab x embodiment x end_effector
    r2_sample.parquet  with --check-r2 N: N listed episodes per (lab, embodiment)
    summary.json       totals, rejection counts, R2 estimate

Read-only: SQL selects, R2 list calls and (with --lance) a read of recordings.lance.

    python -m egomimic.scripts.lance_pilot.plan_zarr_migration \\
        --out /coc/flash7/scratch/lance_pilot/migration_plan --check-r2 40

Aria and EVA episodes migrate from raw only: with --raw-listing, those whose raw capture is
complete on R2 are routed to the raw converters (aria_to_lance, eva_to_lance) and the rest
are rejected (no_raw, incomplete_raw). Every other episode goes through its zarr
(zarr_to_lance).
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from egomimic.rldb.embodiment.end_effector import END_EFFECTOR
from egomimic.rldb.lance import commit
from egomimic.rldb.lance.metadata import placeholder
from egomimic.rldb.lance.schema import recording_id, steps_path
from egomimic.scripts.zarr_to_lance import STAGING_SOURCE
from egomimic.utils.aws.aws_sql import (
    create_default_engine,
    episode_hash_to_timestamp_ms,
)
from egomimic.utils.env import load_env

COLUMNS = (
    "episode_hash, operator, lab, embodiment, rig_name, scene, task, task_description,"
    " objects, license, num_frames, zarr_processed_path, zarr_mp4_path,"
    " zarr_processing_error"
)

# Embodiment (SQL value) -> (end_effector, steps schema in egomimic/rldb/lance/schema.py).
EMBODIMENTS = {
    "human_bimanual": (END_EFFECTOR.HUMAN_HAND, "HUMAN_STEPS"),
    "human_left_arm": (END_EFFECTOR.HUMAN_HAND, "HUMAN_STEPS"),
    "human_right_arm": (END_EFFECTOR.HUMAN_HAND, "HUMAN_STEPS"),
    "eva_bimanual": (END_EFFECTOR.PARALLEL_JAW_GRIPPER, "EVA_STEPS"),
    "eva_left_arm": (END_EFFECTOR.PARALLEL_JAW_GRIPPER, "EVA_STEPS"),
    "eva_right_arm": (END_EFFECTOR.PARALLEL_JAW_GRIPPER, "EVA_STEPS"),
    # YAM stores exactly EVA's per-arm arrays.
    "yam_bimanual": (END_EFFECTOR.PARALLEL_JAW_GRIPPER, "EVA_STEPS"),
    "dexmate_bimanual": (END_EFFECTOR.DEX_HAND, "DEXMATE_STEPS"),
}

# Zarr path segment -> recording_id namespace of the raw converter for that rig. Rows in
# these namespaces migrate from raw only; every other row gets ("zarr", episode_hash).
RAW_NAMESPACES = {"aria": "aria", "eva": "eva"}

RAW_PREFIXES = {"aria": "raw_v2/aria/", "eva": "raw_v2/eva/"}
CONVERTERS = {
    "aria": "egomimic.scripts.aria_process.aria_to_lance",
    "eva": "egomimic.scripts.eva_process.eva_to_lance",
}
# MPS outputs aria_to_lance reads; semidense points and the open-loop trajectory are
# most of the MPS bytes and unused. eye_gaze is optional.
ARIA_MPS_REQUIRED = (
    "hand_tracking/hand_tracking_results.csv",
    "slam/closed_loop_trajectory.csv",
)
ARIA_MPS_OPTIONAL = ("eye_gaze/general_eye_gaze.csv",)


def blank(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or not str(v).strip()


def load_rows(snapshot: Path | None) -> pd.DataFrame:
    """Non-deleted app.episodes rows, from `snapshot` when it exists (written by an
    earlier run from the same query)."""
    if snapshot and snapshot.exists():
        return pd.read_parquet(snapshot)
    q = text(f"select {COLUMNS} from app.episodes where not is_deleted")
    with create_default_engine().connect() as c:
        df = pd.read_sql(q, c)
    if snapshot:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(snapshot)
    return df


def _epoch_ms(episode_hash) -> str | None:
    """The epoch ms of a timestamp episode hash; None for others (Mongo ids, UUIDs)."""
    try:
        return str(episode_hash_to_timestamp_ms(episode_hash))
    except ValueError:
        return None


def vendor_id(row, raw: pd.DataFrame | None = None) -> tuple[str, str]:
    """(source namespace, vendor_recording_id) for recording_id(); see RAW_NAMESPACES.

    The namespace comes from the zarr path, or for a row without one from its raw
    capture: an eva_* row with an EVA HDF5 or a human_* row with an Aria VRS.
    """
    seg = str(row.zarr_processed_path or "").split("/")
    ns = next((RAW_NAMESPACES[s] for s in seg if s in RAW_NAMESPACES), None)
    ms = _epoch_ms(row.episode_hash)
    if ns is None and ms and raw is not None and blank(row.zarr_processed_path):
        emb = str(row.embodiment or "")
        for src, prefix in (("eva", "eva_"), ("aria", "human_")):
            if emb.startswith(prefix) and (src, ms) in raw.index:
                ns = src
    if ns and ms:
        return ns, ms
    # Not keyed by lab, so fixing a row's lab in SQL never changes its recording_id.
    return STAGING_SOURCE, str(row.episode_hash)


def raw_index(listing: pd.DataFrame, bucket: str = "rldb") -> pd.DataFrame:
    """One row per raw capture in a listing of RAW_PREFIXES (columns key, size).

    Indexed by (source, vendor_recording_id); `raw_complete` means every file the
    converter needs is present, `raw_uris` are the files the worker downloads.
    """
    out = []
    for src, prefix in RAW_PREFIXES.items():
        ls = listing[listing.key.str.startswith(prefix)]
        files = dict(zip(ls.key.str.slice(len(prefix)), ls["size"]))
        ext = ".hdf5" if src == "eva" else ".vrs"
        for ms in sorted(
            f[: -len(ext)] for f in files if f.endswith(ext) and "/" not in f
        ):
            # Both take lab/operator/task from SQL; eva_to_lance still reads the robot
            # (embodiment) from the capture's _metadata.json.
            need = (
                [f"{ms}{ext}"]
                if src == "aria"
                else [f"{ms}{ext}", f"{ms}_metadata.json"]
            )
            opt, mps = [], []
            if src == "aria":
                mps = [f"mps_{ms}_vrs/{f}" for f in ARIA_MPS_REQUIRED]
                opt = [f"mps_{ms}_vrs/{f}" for f in ARIA_MPS_OPTIONAL]
            got = [f for f in [*need, *mps, *opt] if f in files]
            missing = [f for f in [*need, *mps] if f not in files]
            out.append(
                dict(
                    source=src,
                    vendor_recording_id=ms,
                    raw_complete=not missing,
                    raw_missing=";".join(missing),
                    mps_present=bool(mps) and all(f in files for f in mps),
                    raw_uris=[f"s3://{bucket}/{prefix}{f}" for f in got],
                    raw_bytes=int(sum(files[f] for f in got)),
                )
            )
    cols = ["source", "vendor_recording_id", "raw_complete", "raw_missing"]
    cols += ["mps_present", "raw_uris", "raw_bytes"]
    return pd.DataFrame(out, columns=cols).set_index(["source", "vendor_recording_id"])


def zarr_uri(path: str) -> str:
    path = str(path).rstrip("/")
    return path if path.startswith("s3://") else f"s3://rldb/{path.lstrip('/')}"


def plan(
    df: pd.DataFrame, batch_size: int, raw: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(manifest, rejected). Aria/eva rows whose raw capture is complete in `raw` (a
    raw_index) get route "raw" and the rest of them are rejected; other rows go through
    their zarr."""
    df = df.copy()
    for c in ("lab", "embodiment"):
        df[c] = df[c].map(lambda v: None if blank(v) else str(v).strip().lower())
    reasons = [[] for _ in range(len(df))]
    n = len(df)
    rec = raw.to_dict("index") if raw is not None else {}
    ee, schema, ns, vid, rid, table, uri = ([None] * n for _ in range(7))
    route = ["zarr"] * n
    for i, r in enumerate(df.itertuples(index=False)):
        why = reasons[i]
        for col in ("lab", "task", "operator"):
            if placeholder(getattr(r, col)):
                why.append(f"missing_{col}")
        if blank(r.embodiment):
            why.append("missing_embodiment")
        elif r.embodiment not in EMBODIMENTS:
            why.append(f"unknown_embodiment:{r.embodiment}")
        else:
            e, s = EMBODIMENTS[r.embodiment]
            ee[i], schema[i] = e.value, s
        if not blank(r.zarr_processed_path):
            uri[i] = zarr_uri(r.zarr_processed_path)
        if not blank(r.lab) and (uri[i] or raw is not None):
            ns[i], vid[i] = vendor_id(r, raw)
            if ns[i] == STAGING_SOURCE and not uri[i]:
                ns[i] = vid[i] = None
            else:
                rid[i] = recording_id(ns[i], vid[i])
        if ns[i] in CONVERTERS:
            hit = rec.get((ns[i], vid[i]))
            if hit and hit["raw_complete"]:
                route[i] = "raw"
            else:
                why.append("no_raw" if not hit else "incomplete_raw")
        if route[i] == "zarr":
            if not uri[i]:
                why.append("no_zarr_path")
            if not blank(r.zarr_processing_error):
                why.append("zarr_processing_error")
            # A raw conversion counts its own frames; SQL num_frames is the zarr's.
            if pd.isna(r.num_frames) or r.num_frames <= 0:
                why.append("no_frames")
        if ee[i] and not blank(r.lab):
            table[i] = steps_path(r.lab, r.embodiment, ee[i])
    df["end_effector"], df["steps_schema"] = ee, schema
    df["source"], df["vendor_recording_id"], df["recording_id"] = ns, vid, rid
    df["steps_table"], df["zarr_uri"], df["route"] = table, uri, route
    df["converter"] = [
        CONVERTERS[s] if r == "raw" else "zarr_to_lance" for s, r in zip(ns, route)
    ]
    if raw is not None:
        for c in ("raw_uris", "raw_bytes", "raw_missing", "mps_present"):
            df[c] = [rec.get(k, {}).get(c) for k in zip(ns, vid)]

    for col, tag in (
        ("zarr_uri", "duplicate_zarr_uri"),
        ("recording_id", "duplicate_recording_id"),
    ):
        dup = df[col].notna() & df[col].duplicated(keep=False)
        if col == "zarr_uri":
            dup &= df.route == "zarr"
        for i in np.flatnonzero(dup.to_numpy()):
            reasons[i].append(tag)
    df["reasons"] = [";".join(w) for w in reasons]

    ok = df[df.reasons == ""].drop(columns="reasons")
    ok = ok.sort_values(["steps_table", "route", "episode_hash"]).reset_index(drop=True)
    ok["shard"] = ok.steps_table.str.removeprefix("steps/").str.removesuffix(".lance")
    # Work units never span steps tables or routes: each stages into one table and is
    # run by one driver.
    ok["work_unit"] = (
        ok.shard
        + "/"
        + ok.route
        + "/"
        + (ok.groupby(["shard", "route"]).cumcount() // batch_size).map("{:05d}".format)
    )
    bad = df[df.reasons != ""].reset_index(drop=True)
    return ok, bad


def _list(uri: str) -> tuple[int, int]:
    files = commit.list_files(uri)
    return len(files), sum(size for size, _ in files.values())


def check_r2(
    rows: pd.DataFrame, per_stratum: int, seed: int, threads: int
) -> pd.DataFrame:
    """List the zarr prefix of `per_stratum` random rows of each (lab, embodiment)."""
    rows = rows[rows.zarr_uri.notna()]
    rng = random.Random(seed)
    picks = []
    for _, g in rows.groupby(["lab", "embodiment"], dropna=False):
        idx = list(g.index)
        picks += rng.sample(idx, min(per_stratum, len(idx)))
    sample = rows.loc[
        picks, ["episode_hash", "lab", "embodiment", "num_frames", "zarr_uri"]
    ]
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(_list, sample.zarr_uri))
    sample = sample.assign(n_objects=[n for n, _ in res], bytes=[b for _, b in res])
    sample["exists"] = sample.n_objects > 0
    return sample.reset_index(drop=True)


def inventory(
    ok: pd.DataFrame, bad: pd.DataFrame, sample: pd.DataFrame | None
) -> pd.DataFrame:
    allrows = pd.concat([ok.assign(convertible=True), bad.assign(convertible=False)])
    allrows["convertible_frames"] = allrows.num_frames.where(allrows.convertible, 0)
    allrows["kind"] = np.where(
        allrows.embodiment.fillna("").str.startswith("human"), "human", "robot"
    )
    keys = ["lab", "embodiment", "end_effector"]
    allrows[keys] = allrows[keys].fillna("NULL")
    inv = allrows.groupby(["kind", *keys]).agg(
        episodes=("episode_hash", "size"),
        convertible=("convertible", "sum"),
        frames=("num_frames", "sum"),
        convertible_frames=("convertible_frames", "sum"),
    )
    if sample is not None and len(sample):
        s = sample[sample.exists & (sample.num_frames > 0)]
        bpf = (
            s.groupby(["lab", "embodiment"]).bytes.sum()
            / s.groupby(["lab", "embodiment"]).num_frames.sum()
        )
        overall = s.bytes.sum() / max(s.num_frames.sum(), 1)
        miss = sample.groupby(["lab", "embodiment"]).exists.agg(lambda e: 1 - e.mean())
        idx = inv.index.to_frame(index=False)
        inv["bytes_per_frame"] = [
            bpf.get((lab, emb), overall) for lab, emb in zip(idx.lab, idx.embodiment)
        ]
        inv["est_zarr_gb"] = inv.frames * inv.bytes_per_frame / 1e9
        inv["sample_missing_frac"] = [
            miss.get((lab, emb), np.nan) for lab, emb in zip(idx.lab, idx.embodiment)
        ]
    return inv.reset_index().sort_values("frames", ascending=False)


def list_raw(threads: int, bucket: str = "rldb") -> pd.DataFrame:
    """Every object under RAW_PREFIXES (key, size), listed in parallel by first character."""
    from egomimic.rldb.lance.dataset import _s3_client

    client = _s3_client()
    chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-."

    def one(prefix):
        pages = client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        )
        return [(o["Key"], o["Size"]) for p in pages for o in p.get("Contents", [])]

    subs = [p + c for p in RAW_PREFIXES.values() for c in chars]
    with ThreadPoolExecutor(threads) as ex:
        rows = [r for rs in ex.map(one, subs) for r in rs]
    return pd.DataFrame(rows, columns=["key", "size"])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--snapshot",
        type=Path,
        help="parquet cache of the SQL rows; read if it exists, else written",
    )
    ap.add_argument("--batch-size", type=int, default=64, help="episodes per work unit")
    ap.add_argument(
        "--check-r2",
        type=int,
        default=0,
        metavar="N",
        help="list N random zarrs per (lab, embodiment)",
    )
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--lance", help="Lance root; recordings already in it are marked committed"
    )
    ap.add_argument(
        "--raw-listing",
        type=Path,
        help="parquet listing (key, size) of the raw prefixes; read if it exists, else"
        " listed from R2 and written. Routes aria/eva rows with complete raw to the raw"
        " converters",
    )
    args = ap.parse_args()
    load_env()

    df = load_rows(args.snapshot)
    raw = None
    if args.raw_listing:
        if args.raw_listing.exists():
            listing = pd.read_parquet(args.raw_listing)
        else:
            listing = list_raw(args.threads)
            args.raw_listing.parent.mkdir(parents=True, exist_ok=True)
            listing.to_parquet(args.raw_listing, index=False)
        raw = raw_index(listing)
    ok, bad = plan(df, args.batch_size, raw)
    if args.lance:
        ok["committed"] = ok.recording_id.isin(commit.committed_ids(args.lance))

    sample = None
    if args.check_r2:
        both = pd.concat([ok, bad], ignore_index=True)
        both = both.assign(
            lab=both.lab.fillna("NULL"), embodiment=both.embodiment.fillna("NULL")
        )
        sample = check_r2(both, args.check_r2, args.seed, args.threads)
        missing = set(sample.loc[~sample.exists, "episode_hash"])
        ok["sampled_missing_on_r2"] = ok.episode_hash.isin(missing)

    args.out.mkdir(parents=True, exist_ok=True)
    keep = [
        "episode_hash",
        "lab",
        "embodiment",
        "end_effector",
        "steps_schema",
        "rig_name",
        "operator",
        "task",
        "task_description",
        "scene",
        "objects",
        "license",
        "num_frames",
        "zarr_uri",
        "zarr_mp4_path",
        "source",
        "vendor_recording_id",
        "recording_id",
        "steps_table",
        "shard",
        "work_unit",
        "route",
        "converter",
    ]
    opt = ("raw_uris", "raw_bytes", "raw_missing", "mps_present")
    keep += [c for c in (*opt, "committed", "sampled_missing_on_r2") if c in ok]
    ok[keep].to_parquet(args.out / "manifest.parquet", index=False)
    bad[
        [c for c in [*keep, "zarr_processing_error", "reasons"] if c in bad]
    ].to_parquet(args.out / "rejected.parquet", index=False)
    inv = inventory(ok, bad, sample)
    inv.to_csv(args.out / "inventory.csv", index=False)
    if sample is not None:
        sample.to_parquet(args.out / "r2_sample.parquet", index=False)
    if raw is not None:
        raw.reset_index().to_parquet(args.out / "raw_index.parquet", index=False)

    reason_counts = (
        bad.reasons.str.split(";").explode().str.split(":").str[0].value_counts()
    )
    summary = {
        "episodes": len(df),
        "convertible": len(ok),
        "rejected": len(bad),
        "frames_total": int(df.num_frames.fillna(0).sum()),
        "frames_convertible": int(ok.num_frames.sum()),
        "steps_tables": int(ok.steps_table.nunique()),
        "work_units": int(ok.work_unit.nunique()),
        "routes": ok.route.value_counts().to_dict(),
        "frames_by_route": ok.groupby("route").num_frames.sum().astype(int).to_dict(),
        "rejection_reasons": reason_counts.to_dict(),
        "missing_required_any": int(
            bad.reasons.str.contains("missing_(?:lab|task|operator)").sum()
        ),
    }
    if sample is not None:
        summary["r2_sampled"] = len(sample)
        summary["r2_sampled_missing"] = int((~sample.exists).sum())
        summary["est_zarr_tb"] = float(inv.est_zarr_gb.sum() / 1e3)
    if raw is not None:
        is_raw = ok.route == "raw"
        summary["raw_tb"] = float(ok.raw_bytes[is_raw].sum() / 1e12)
        summary["raw_by_source"] = ok[is_raw].source.value_counts().to_dict()
        ms_seen = set(df.episode_hash.map(_epoch_ms))
        summary["raw_without_sql_row"] = (
            raw[[v not in ms_seen for _, v in raw.index]]
            .groupby(level=0)
            .size()
            .to_dict()
        )
    if args.lance:
        summary["already_committed"] = int(ok.committed.sum())
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    with pd.option_context(
        "display.width",
        250,
        "display.max_rows",
        200,
        "display.float_format",
        "{:,.2f}".format,
    ):
        print(inv.to_string(index=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
