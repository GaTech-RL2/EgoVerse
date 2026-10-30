import argparse
import json

import pandas as pd
import pytest

from egomimic.rldb.lance.schema import recording_id
from egomimic.scripts.lance_pilot.plan_zarr_migration import (
    load_rows,
    plan,
    raw_index,
)
from egomimic.utils.aws.aws_sql import episode_hash_to_timestamp_ms


def row(h, **kw):
    base = dict(
        episode_hash=h,
        operator="op",
        lab="rl2",
        embodiment="eva_bimanual",
        rig_name="eva",
        scene=None,
        task="fold",
        task_description=None,
        objects=None,
        license=None,
        num_frames=100,
        zarr_processed_path=f"s3://rldb/processed_v3/eva/{h}.zarr",
        zarr_mp4_path=None,
        zarr_processing_error="",
    )
    return {**base, **kw}


def _vendor(h, lab, embodiment):
    return row(
        h,
        lab=lab,
        embodiment=embodiment,
        rig_name=None,
        zarr_processed_path=f"s3://rldb/processed_v3/{lab}/{h}",
    )


def test_plan_ids_tables_and_rejections():
    eva = "2025-11-26-21-07-35-274000"
    mecka = "69b4971fd211e71b4719984b"
    yam = "37757ec4-c681-43cc-bfe4-7be3fc520f8a"
    dex = "2026-05-29-03-16-12-000000"
    df = pd.DataFrame(
        [
            row(eva),
            _vendor(mecka, "mecka", "human_bimanual"),
            _vendor(yam, "abc", "yam_bimanual"),
            _vendor(dex, "sharpa_dex", "dexmate_bimanual"),
            row("2025-01-01-00-00-00-000000", operator="Unknown"),
            row(
                "2025-01-01-00-00-02-000000",
                lab="mecka",
                zarr_processed_path=None,
                num_frames=-1,
            ),
        ]
    )
    ok, bad = plan(df, batch_size=1)

    assert sorted(ok.episode_hash) == sorted([mecka, yam, dex])
    r = ok.set_index("episode_hash")
    m = r.loc[mecka]
    assert m.recording_id == recording_id("zarr", mecka)
    assert m.shard == "mecka/human_bimanual/human_hand" and m.route == "zarr"
    assert r.loc[yam, "steps_table"] == (
        "steps/abc/yam_bimanual/parallel_jaw_gripper.lance"
    )
    assert r.loc[yam, "steps_schema"] == "EVA_STEPS"
    assert r.loc[dex, "end_effector"] == "dex_hand"
    assert r.loc[dex, "steps_schema"] == "DEXMATE_STEPS"
    assert r.loc[dex, "recording_id"] == recording_id("zarr", dex)

    why = dict(zip(bad.episode_hash, bad.reasons))
    # EVA migrates from raw only, and no raw listing was given.
    assert why[eva] == "no_raw"
    assert why["2025-01-01-00-00-00-000000"] == "missing_operator;no_raw"
    assert why["2025-01-01-00-00-02-000000"] == "no_zarr_path;no_frames"


def test_raw_index_and_routing():
    aria, aria_no_mps, eva_raw_only = (
        "2025-11-24-20-24-32-080000",
        "2025-11-24-20-24-33-000000",
        "2025-11-26-21-07-35-274000",
    )
    a, b, e = (
        str(episode_hash_to_timestamp_ms(h)) for h in (aria, aria_no_mps, eva_raw_only)
    )
    keys = {
        f"raw_v2/aria/{a}.vrs": 100,
        f"raw_v2/aria/{a}_metadata.json": 1,
        f"raw_v2/aria/mps_{a}_vrs/hand_tracking/hand_tracking_results.csv": 10,
        f"raw_v2/aria/mps_{a}_vrs/slam/closed_loop_trajectory.csv": 20,
        f"raw_v2/aria/mps_{a}_vrs/slam/semidense_points.csv.gz": 999,
        f"raw_v2/aria/{b}.vrs": 100,
        f"raw_v2/aria/mps_{b}_vrs/slam/closed_loop_trajectory.csv": 20,
        f"raw_v2/eva/{e}.hdf5": 300,
        f"raw_v2/eva/{e}_metadata.json": 2,
        "raw_v2/eva/corrupted/123.hdf5": 5,
    }
    raw = raw_index(pd.DataFrame({"key": list(keys), "size": list(keys.values())}))
    assert raw.loc[("aria", a), "raw_complete"] and raw.loc[("aria", a), "mps_present"]
    assert raw.loc[("aria", a), "raw_bytes"] == 130
    assert not raw.loc[("aria", b), "raw_complete"]
    assert raw.loc[("eva", e), "raw_complete"] and len(raw) == 3

    df = pd.DataFrame(
        [
            row(
                aria,
                embodiment="human_bimanual",
                rig_name="aria",
                zarr_processed_path=f"s3://rldb/processed_v3/aria/{aria}.zarr",
            ),
            row(
                aria_no_mps,
                embodiment="human_bimanual",
                rig_name="aria",
                zarr_processed_path=f"s3://rldb/processed_v3/aria/{aria_no_mps}.zarr",
            ),
            row(eva_raw_only, zarr_processed_path=None, num_frames=None),
            row("2025-01-01-00-00-03-000000", zarr_processed_path=None),
        ]
    )
    ok, bad = plan(df, batch_size=8, raw=raw)
    r = ok.set_index("episode_hash")
    assert set(ok.route) == {"raw"}
    assert r.loc[aria, "converter"] == "egomimic.scripts.aria_process.aria_to_lance"
    assert r.loc[aria, "recording_id"] == recording_id("aria", a)
    assert not any("semidense" in u for u in r.loc[aria, "raw_uris"])
    assert r.loc[eva_raw_only, "recording_id"] == recording_id("eva", e)
    assert r.loc[eva_raw_only, "work_unit"] == (
        "rl2/eva_bimanual/parallel_jaw_gripper/raw/00000"
    )
    assert dict(zip(bad.episode_hash, bad.reasons)) == {
        aria_no_mps: "incomplete_raw",
        "2025-01-01-00-00-03-000000": "no_zarr_path",
    }


def test_snapshot_is_read_back(tmp_path):
    snap = tmp_path / "snapshot.parquet"
    pd.DataFrame([row("2025-01-01-00-00-00-000000")]).to_parquet(snap)
    assert list(load_rows(snap).episode_hash) == ["2025-01-01-00-00-00-000000"]


def test_driver_redoes_half_staged_episodes(tmp_path):
    pytest.importorskip("ray")
    from egomimic.scripts.zarr_to_lance_ray import select

    manifest = tmp_path / "manifest.parquet"
    hashes = ["a", "b", "c"]
    pd.DataFrame(
        dict(
            episode_hash=hashes,
            vendor_recording_id=hashes,
            route="zarr",
            shard="s",
            work_unit="u",
        )
    ).to_parquet(manifest)
    staging = tmp_path / "staging" / "zarr"
    (staging / "a").mkdir(parents=True)
    (staging / "a" / "meta.json").write_text("{}")
    (staging / "b").mkdir()
    (staging / "b" / "steps.parquet").write_text("")
    report = tmp_path / "report.jsonl"
    report.touch()
    args = argparse.Namespace(
        manifest=manifest,
        shard=None,
        work_unit=None,
        lance=None,
        stage_to=str(tmp_path / "staging"),
        report=report,
        retry_failed=False,
        limit=None,
    )
    assert list(select(args).episode_hash) == ["b", "c"]
    report.write_text(json.dumps({"episode_hash": "c", "status": "failed"}) + "\n")
    assert list(select(args).episode_hash) == ["b"]
