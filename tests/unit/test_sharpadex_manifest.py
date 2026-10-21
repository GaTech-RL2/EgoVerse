"""Plan a tiny synthetic SharpaDex release: dedup, holds, clocks and names."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from egomimic.rldb.embodiment.registry import load_end_effectors
from egomimic.scripts.sharpa_process import sharpadex_manifest as manifest
from egomimic.scripts.sharpa_process.sharpa_to_zarr import episode_hash

FPS = 30
SEASON = "season_POC99001_2026_01_02_03_04_05_train"
OTHER_RIG = "season_POC99002_2026_01_02_03_04_05_train"
#: 2026-01-02 03:04:05 at +08:00.
SEASON_START_NS = int(datetime(2026, 1, 1, 19, 4, 5, tzinfo=timezone.utc).timestamp())
SEASON_START_NS *= 10**9


def _state(excess: float = 0.0, seed: int = 0) -> dict:
    """Joint statistics inside the registry hand limits, optionally past one."""
    specs = load_end_effectors()
    middle = np.zeros(65)
    for side, offset in manifest.SIDES.items():
        limits = np.asarray(specs[f"sharpa_wave_v1_{side}"].joint_limits)
        middle[offset : offset + 22] = limits.mean(axis=1)
    low, high = middle - 0.01, middle + 0.01 + 1e-3 * seed
    if excess:
        right = specs["sharpa_wave_v1_left"].joint_limits[8][1]  # left_index_DIP
        high[7 + 8] = right + excess
    return {"min": low.tolist(), "max": high.tolist(), "mean": middle.tolist()}


def _stats(seed: int = 0, excess: float = 0.0, flat_thumb: bool = False) -> dict:
    tactile_low = np.zeros(60)
    tactile_high = np.ones(60)
    if flat_thumb:
        tactile_high[30:36] = 0.0  # right thumb
    return {
        "observation.state": _state(excess, seed),
        "action": _state(0.0, seed),
        "observation.tactile": {
            "min": tactile_low.tolist(),
            "max": tactile_high.tolist(),
        },
    }


def _season(root: Path, task: str, season: str, episodes: list[dict]) -> None:
    """Write one season's v2.1 metadata; each episode dict sets its fields."""
    meta = root / task / season / "lerobot_v2.1" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"codebase_version": "v2.1", "fps": FPS})
    )
    with (meta / "episodes.jsonl").open("w") as f:
        for i, e in enumerate(episodes):
            f.write(json.dumps({"episode_index": i, "length": e["length"]}) + "\n")
    with (meta / "episodes_stats.jsonl").open("w") as f:
        for i, e in enumerate(episodes):
            f.write(json.dumps({"episode_index": i, "stats": e["stats"]}) + "\n")
    with (meta / "annotations.jsonl").open("w") as f:
        for i, e in enumerate(episodes):
            row = {
                "episode_index": i,
                "source_uid": e["uid"],
                "episode_id": e.get("episode_id", ""),
                "tags": {"task_name": e.get("tag", task), "score": "S"},
            }
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def release(tmp_path) -> Path:
    root = tmp_path / "release"
    _season(
        root,
        "empty_dustpan_into_bin",
        SEASON,
        [
            # Starts with the season; its recording name agrees.
            {
                "length": 300,
                "stats": _stats(0),
                "uid": "u0",
                "episode_id": "POC99001_2026_01_02_03_04_05_train",
            },
            # Recorded a minute later; mis-filed copy below.
            {
                "length": 300,
                "stats": _stats(1),
                "uid": "u1",
                "episode_id": "POC99001_2026_01_02_03_05_05_train",
            },
            # No per-episode name: the season clock applies.
            {"length": 90, "stats": _stats(2), "uid": "u2"},
            {
                "length": 60,
                "stats": _stats(3, excess=0.4),
                "uid": "u3",
                "episode_id": "POC99001_2026_01_02_03_07_00_train",
            },
            {
                "length": 60,
                "stats": _stats(4, flat_thumb=True),
                "uid": "u4",
                "episode_id": "POC99001_2026_01_02_03_08_00_train",
            },
        ],
    )
    _season(
        root,
        "sweep_waste_into_dustpan",
        SEASON,
        [
            # An identical copy of u1, tagged with the other task.
            {
                "length": 300,
                "stats": _stats(1),
                "uid": "u1",
                "tag": "Empty_Dustpan_Into_Bin",
                "episode_id": "POC99001_2026_01_02_03_05_05_train",
            },
            # Shares u2 with different data.
            {"length": 90, "stats": _stats(9), "uid": "u2"},
        ],
    )
    # Another rig that started recording in the same second.
    _season(
        root,
        "fold_towel",
        OTHER_RIG,
        [
            {
                "length": 30,
                "stats": _stats(5),
                "uid": "u5",
                "episode_id": "POC99002_2026_01_02_03_04_05_train",
            }
        ],
    )
    return root


def _plan(release, tmp_path, *extra):
    out = tmp_path / "manifest.jsonl"
    assert (
        manifest.main(["--release-root", str(release), "--output", str(out), *extra])
        == 0
    )
    header, records = manifest.read_manifest(out)
    return header, records


def test_duplicates_collisions_and_holds_are_decided(release, tmp_path):
    header, records = _plan(release, tmp_path)
    by = {r: rec for r, rec in records.items()}
    dustpan = f"empty_dustpan_into_bin/{SEASON}"
    sweep = f"sweep_waste_into_dustpan/{SEASON}"

    # The copy filed under the matching task folder is the one kept.
    assert by[f"{dustpan}/1"].decision == "keep"
    assert by[f"{sweep}/0"].decision == "drop_duplicate"
    assert by[f"{sweep}/0"].duplicate_of == f"{dustpan}/1"
    # One source_uid naming different data holds both records.
    assert (
        by[f"{dustpan}/2"].decision == by[f"{sweep}/1"].decision == "hold_uid_collision"
    )
    # Observed hand joints 0.4 rad past a limit are held.
    assert by[f"{dustpan}/3"].decision == "hold_gross_excursion"
    assert "obs_hand_excursion_over_5deg:left" in by[f"{dustpan}/3"].flags
    assert "tactile_flat:right_thumb" in by[f"{dustpan}/4"].flags
    assert header["summary"]["decisions"]["keep"] == 4


def test_clocks_come_from_recording_names_then_the_season(release, tmp_path):
    _, records = _plan(release, tmp_path)
    dustpan = f"empty_dustpan_into_bin/{SEASON}"
    first, second = records[f"{dustpan}/0"], records[f"{dustpan}/1"]
    assert first.clock == second.clock == manifest.CLOCK_VENDOR
    assert first.start_source == "episode_id"
    assert second.start_ns == SEASON_START_NS + 60 * 10**9
    # No per-episode name: season start plus the 600 frames before it.
    third = records[f"{dustpan}/2"]
    assert third.clock == manifest.CLOCK_SYNTHESIZED
    assert third.start_ns == SEASON_START_NS + (600 * 10**9) // FPS


def test_names_are_unique_and_avoid_existing_hashes(release, tmp_path):
    _, records = _plan(release, tmp_path)
    kept = [r for r in records.values() if r.decision == "keep"]
    names = [r.episode_hash for r in kept]
    assert len(set(names)) == len(names)
    # Two rigs started in the same second: the later in visiting order moves 1 ms.
    towel = records[f"fold_towel/{OTHER_RIG}/0"]
    first = records[f"empty_dustpan_into_bin/{SEASON}/0"]
    assert first.hash_offset_ms == 0 and towel.hash_offset_ms == 1
    assert towel.episode_hash == episode_hash(SEASON_START_NS + 10**6)

    taken = tmp_path / "taken.txt"
    taken.write_text(episode_hash(SEASON_START_NS) + "\n")
    _, again = _plan(release, tmp_path, "--existing-hashes", str(taken))
    assert again[f"empty_dustpan_into_bin/{SEASON}/0"].hash_offset_ms == 1
    assert again[f"fold_towel/{OTHER_RIG}/0"].hash_offset_ms == 2


def test_exclusions_must_name_real_episodes(release, tmp_path):
    good = tmp_path / "exclude.yaml"
    good.write_text(
        yaml.safe_dump(
            {
                "exclude": [
                    {
                        "task": "fold_towel",
                        "season": OTHER_RIG,
                        "episode_index": 0,
                        "reason": "corrupt frame",
                    }
                ]
            }
        )
    )
    _, records = _plan(release, tmp_path, "--exclude-file", str(good))
    towel = records[f"fold_towel/{OTHER_RIG}/0"]
    assert towel.decision == "exclude" and towel.episode_hash is None
    # With the towel gone nothing shares the first name.
    assert records[f"empty_dustpan_into_bin/{SEASON}/0"].hash_offset_ms == 0

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "exclude": [
                    {"task": "x", "season": "y", "episode_index": 0, "reason": "z"}
                ]
            }
        )
    )
    with pytest.raises(SystemExit, match="unknown episodes"):
        _plan(release, tmp_path, "--exclude-file", str(bad))


def test_the_shipped_v1_exclusions_parse():
    path = Path(manifest.__file__).with_name("sharpadex_v1_exclusions.yaml")
    exclusions = manifest.load_exclusions(path)
    assert len(exclusions) == 3
    assert all(reason for reason in exclusions.values())
