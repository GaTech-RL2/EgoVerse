"""Launch and preflight must bind the corrected data view to exact frame receipts."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def proof(tmp_path, monkeypatch):
    source = (
        Path(__file__).resolve().parents[2] / "experiments/fold_clothes_moving_opscale"
    )
    monkeypatch.syspath_prepend(str(source))
    launch = importlib.import_module("launch_pace")
    root = tmp_path / "private"
    for folder in ("inputs", "recovery2", "canonical_zarr", "splits"):
        (root / folder).mkdir(parents=True)
    launch.write_json(root / "moving_opscale_design.json", {"fixture": True})
    (root / "inputs/main_episodes.csv").write_text("episode_id,frames\noriginal,100\n")
    launch.write_json(
        root / "recovery2/report.json",
        {"episodes": [{"episode_id": "recovered", "frames": 150, "validated": True}]},
    )
    index = {
        eid: {
            "episode_id": eid,
            "total_frames": count,
            "moving_opscale_canonicalization": {
                "source_convention": convention,
                "target_convention": "post_fix",
                "frames": count,
            },
        }
        for eid, count, convention in [
            ("original", 100, "pre_fix"),
            ("recovered", 150, "post_fix"),
        ]
    }
    launch.write_json(root / "canonical_zarr/_episode_index.json", index)
    for b, n in launch.MATRIX:
        launch.write_json(
            root / "splits" / f"fold_moving_b{b}_op{n}.json",
            {
                "train": ["original"],
                "seen_matched": ["recovered"],
                "seen_perlevel": [],
                "unseen": [],
            },
        )
    ready = {
        "schema_version": 1,
        "status": "passed",
        "checks": {key: True for key in launch.DATA_CHECKS},
        "design_sha256": launch.digest(root / "moving_opscale_design.json"),
        "canonical_root": str(root / "canonical_zarr"),
        "canonical_index_sha256": launch.digest(
            root / "canonical_zarr/_episode_index.json"
        ),
        "converted_frame_index": {
            "main_episodes_sha256": launch.digest(root / "inputs/main_episodes.csv"),
            "recovery_report_sha256": launch.digest(root / "recovery2/report.json"),
            "selected_frames": 250,
        },
    }
    return launch, root, ready, index


def test_verified_canonical_index_matches_source_and_recovery(proof):
    launch, root, ready, _ = proof
    assert launch.validate_data_proof(root, ready) == ready["canonical_index_sha256"]


def test_raw_resolver_root_cannot_replace_canonical_view(proof):
    launch, root, ready, _ = proof
    ready["canonical_root"] = str(root / "zarr")
    with pytest.raises(RuntimeError, match="Canonical resolver root differs"):
        launch.validate_data_proof(root, ready)


def test_source_index_change_invalidates_proof(proof):
    launch, root, ready, _ = proof
    (root / "inputs/main_episodes.csv").write_text("episode_id,frames\noriginal,99\n")
    with pytest.raises(RuntimeError, match="source frame index changed"):
        launch.validate_data_proof(root, ready)


def test_equal_total_does_not_hide_swapped_episode_frame_counts(proof):
    launch, root, ready, index = proof
    index["original"]["total_frames"] = 150
    index["recovered"]["total_frames"] = 100
    launch.write_json(root / "canonical_zarr/_episode_index.json", index)
    ready["canonical_index_sha256"] = launch.digest(
        root / "canonical_zarr/_episode_index.json"
    )
    with pytest.raises(RuntimeError, match="frame count differs"):
        launch.validate_data_proof(root, ready)


def test_pre_fix_target_cannot_pass_canonical_proof(proof):
    launch, root, ready, index = proof
    index["original"]["moving_opscale_canonicalization"]["target_convention"] = (
        "pre_fix"
    )
    launch.write_json(root / "canonical_zarr/_episode_index.json", index)
    ready["canonical_index_sha256"] = launch.digest(
        root / "canonical_zarr/_episode_index.json"
    )
    with pytest.raises(RuntimeError, match="wrist convention is unverified"):
        launch.validate_data_proof(root, ready)


def test_norm_cache_is_bound_to_canonical_index(proof):
    launch, private, ready, _ = proof
    root = private.parent
    norm = root / "norm/b24_op4"
    (norm / "norm_stats").mkdir(parents=True)
    stats = norm / "norm_stats/norm_stats.json"
    stats.write_text("{}")
    entry = {"config_sha256": "config", "split_sha256": "split"}
    report = {
        "status": "passed",
        "commit": "a" * 40,
        "design_sha256": "design",
        "budget_hours": 24,
        "operator_count": 4,
        **entry,
        "seed": 42026,
        "sample_frac": 1.0,
        "max_samples": 200000,
        "pool_horizon": False,
        "canonical_index_sha256": ready["canonical_index_sha256"],
        "stats_path": str(stats),
        "stats_sha256": launch.digest(stats),
    }
    launch.write_json(norm / "norm_complete.json", report)
    assert launch.verified_norm(root, 24, 4, "a" * 40, "design", entry)[0] == stats
    report["canonical_index_sha256"] = "stale"
    launch.write_json(norm / "norm_complete.json", report)
    with pytest.raises(RuntimeError, match="Normalization provenance mismatch"):
        launch.verified_norm(root, 24, 4, "a" * 40, "design", entry)
