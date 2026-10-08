"""Migration report: what each Aria delivery keeps, drops and changes relative to today's zarr."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def column(t, side, field=None):
    c = t.column(side).combine_chunks()
    if field is not None:
        c = c.field(field)
    return np.asarray(c.flatten()).reshape(t.num_rows, -1)


def quat_angle_deg(a, b):
    dot = np.abs((a * b).sum(1)).clip(0, 1)
    return np.degrees(2 * np.arccos(dot))


def compare(d: Path, zdir: Path, emb: str) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    t = pq.read_table(d / "steps.parquet")
    z = zarr.open(str(zdir), mode="r")
    nz = int(z.attrs["total_frames"])
    ts = t.column("timestamp_ns").to_numpy()
    zts = np.asarray(z["obs_rgb_timestamps_ns"][:nz])
    idx = np.searchsorted(ts, zts).clip(0, len(ts) - 1)
    found = ts[idx] == zts

    lv, rv = t.column("left_valid").to_numpy(), t.column("right_valid").to_numpy()
    hv = t.column("head_pose_valid").to_numpy()
    sides = {
        "human_bimanual": ("left", "right"),
        "human_left_arm": ("left",),
        "human_right_arm": ("right",),
    }[emb]
    any_hand = np.zeros(len(ts), bool)
    for s in sides:
        any_hand |= {"left": lv, "right": rv}[s]
    in_zarr = np.zeros(len(ts), bool)
    in_zarr[idx[found]] = True

    r = dict(
        vendor_recording_id=meta["vendor_recording_id"],
        embodiment=emb,
        zarr_frames=nz,
        zarr_frames_missing_in_lance=int((~found).sum()),
        lance_frames=meta["num_frames"],
        filler_frames=meta["filler_frames"],
        gaps=meta["gaps"],
        graph_changes=meta["graph_changes"],
        left_valid=float(lv.mean()),
        right_valid=float(rv.mean()),
        head_valid=float(hv.mean()),
        any_hand_valid=float(any_hand.mean()),
        valid_not_in_zarr=int((any_hand & ~in_zarr).sum()),
        zarr_kept_now_invalid=int((in_zarr & ~any_hand).sum()),
        bytes_zarr=dir_bytes(zdir),
        bytes_mp4_360=(d / "front_1.mp4").stat().st_size,
        bytes_mp4_480=(d / "front_1_480.mp4").stat().st_size,
        bytes_parquet=(d / "steps.parquet").stat().st_size,
        seconds=meta["seconds_total"],
    )
    zi = idx[found]
    zrow = np.flatnonzero(found)
    head = column(t, "head_pose")[zi]
    ok = hv[zi]
    r["head_pose_maxdiff"] = float(
        np.abs(head[ok] - np.asarray(z["obs_head_pose"][:nz])[zrow][ok]).max(initial=0)
    )
    for s in sides:
        v = {"left": lv, "right": rv}[s][zi]
        if not v.any():
            continue
        zr = lambda k: np.asarray(z[f"{s}.{k}"][:nz])[zrow][v]  # noqa: E731
        r[f"{s}_wrist_maxdiff"] = float(
            np.abs(column(t, s, "wrist_pose")[zi][v] - zr("obs_wrist_pose")).max()
        )
        r[f"{s}_aria_kp_maxdiff"] = float(
            np.abs(
                column(t, s, "aria_keypoints")[zi][v] - zr("obs_aria_keypoints")
            ).max()
        )
        kp_err = np.linalg.norm(
            (column(t, s, "keypoints")[zi][v] - zr("obs_keypoints")).reshape(-1, 21, 3),
            axis=-1,
        )
        r[f"{s}_mano_kp_p99_mm"] = float(np.percentile(kp_err, 99) * 1e3)
        ee, zee = column(t, s, "ee_pose")[zi][v], zr("obs_ee_pose")
        r[f"{s}_ee_pos_p99_mm"] = float(
            np.percentile(np.linalg.norm(ee[:, :3] - zee[:, :3], axis=1), 99) * 1e3
        )
        r[f"{s}_ee_rot_p99_deg"] = float(
            np.percentile(quat_angle_deg(ee[:, 3:], zee[:, 3:]), 99)
        )
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root", type=Path, default=Path("/coc/flash7/scratch/lance_pilot")
    )
    args = ap.parse_args()
    m = pd.read_csv(args.root / "manifest.csv", dtype={"epoch_ms": str})
    rows, missing = [], []
    for rec in m.itertuples():
        d = args.root / "deliveries" / rec.epoch_ms
        if not (d / "meta.json").exists():
            missing.append(rec.epoch_ms)
            continue
        rows.append(
            compare(d, args.root / "zarr" / rec.episode_hash, rec.embodiment)
            | {"site": rec.lab, "task": rec.task}
        )
    df = pd.DataFrame(rows)
    df.to_csv(args.root / "migration_report.csv", index=False)

    tot = df[
        [
            "zarr_frames",
            "lance_frames",
            "filler_frames",
            "valid_not_in_zarr",
            "zarr_kept_now_invalid",
        ]
    ].sum()
    any_valid = (df.any_hand_valid * df.lance_frames).sum()
    print(f"recordings {len(df)} (not converted: {missing or 'none'})")
    print(
        f"frames: zarr {tot.zarr_frames:,}  lance {tot.lance_frames:,} ({tot.lance_frames / tot.zarr_frames:.2f}x)  filler {tot.filler_frames:,}"
    )
    print(
        f"frames with a valid hand: {any_valid:,.0f} ({any_valid / tot.zarr_frames:.2f}x zarr)"
    )
    print(
        f"valid hand frames the zarr dropped: {tot.valid_not_in_zarr:,}; zarr frames now invalid: {tot.zarr_kept_now_invalid:,}"
    )
    print(
        f"recordings with gaps: {(df.gaps > 0).sum()}, with SLAM graph changes: {(df.graph_changes > 0).sum()}"
    )
    print(f"zarr frames missing from lance: {df.zarr_frames_missing_in_lance.sum()}")
    agree = [c for c in df.columns if c.endswith(("maxdiff", "_p99_mm", "_p99_deg"))]
    print("agreement (max over recordings):")
    print(df[agree].max().to_string())
    zb = df.bytes_zarr.sum() / tot.zarr_frames
    lf = tot.lance_frames
    print(
        f"bytes/frame: zarr {zb / 1e3:.1f} KB | mp4 360 {df.bytes_mp4_360.sum() / lf / 1e3:.1f} KB"
        f" | mp4 480 {df.bytes_mp4_480.sum() / lf / 1e3:.1f} KB | parquet {df.bytes_parquet.sum() / lf / 1e3:.2f} KB"
    )
    print(
        f"conversion: median {df.seconds.median():.0f} s/recording, total {df.seconds.sum() / 3600:.1f} GPU-h"
    )
    print(
        df.groupby("embodiment")[
            ["left_valid", "right_valid", "head_valid", "any_hand_valid"]
        ]
        .mean()
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
