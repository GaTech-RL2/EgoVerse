"""Aria uploads through ingest on real recordings: prepare, stage and commit.

    EGOVERSE_TEST_ARIA_RAW_DIR=<dir laid out like raw_v2/aria/: <id>.vrs,
        <id>_metadata.json and mps_<id>_vrs/ per recording>
    pytest --integration tests/integration/test_ingest_aria.py

MANO is off (it needs a GPU and the license-gated model); each recording is converted,
checked by the committer before review, staged and committed to a scratch Lance root.
"""

import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("projectaria_tools")
lance = pytest.importorskip("lance")

from egomimic.ingest import aria  # noqa: E402
from egomimic.ingest.formats import prepare_delivery  # noqa: E402
from egomimic.ingest.stage import REVIEW_ONLY, read_pin  # noqa: E402
from egomimic.rldb.lance.schema import recording_id  # noqa: E402
from egomimic.scripts.lance_commit import commit_staged  # noqa: E402


def _recordings(root: Path, limit: int = 2) -> list[str]:
    ids = sorted(
        p.stem
        for p in root.glob("*.vrs")
        if (root / f"mps_{p.stem}_vrs").is_dir()
        and (root / f"{p.stem}_metadata.json").exists()
    )
    return ids[:limit]


def test_real_aria_recordings_stage_and_commit(tmp_path):
    raw = os.environ.get("EGOVERSE_TEST_ARIA_RAW_DIR")
    if not raw:
        pytest.skip("set EGOVERSE_TEST_ARIA_RAW_DIR to a raw_v2/aria-style directory")
    ids = _recordings(Path(raw))
    assert ids, f"no <id>.vrs + mps_<id>_vrs + <id>_metadata.json under {raw}"
    for vid in ids:
        d = tmp_path / "deliveries" / vid
        d.mkdir(parents=True)
        for name in (f"{vid}.vrs", f"{vid}_metadata.json"):
            (d / name).symlink_to(Path(raw) / name)
        mps = f"mps_{vid}_vrs"
        for rel in (
            aria.HAND_TRACKING,
            aria.TRAJECTORY,
            "eye_gaze/general_eye_gaze.csv",
        ):
            if (Path(raw) / mps / rel).exists():
                (d / mps / rel).parent.mkdir(parents=True, exist_ok=True)
                (d / mps / rel).symlink_to(Path(raw) / mps / rel)
        p = prepare_delivery(
            d, tmp_path / "staged" / vid, tmp_path / "lance", fingerprint="f" * 64,
            vendor="test", mano=False, default_license="test-license",
        )  # fmt: skip
        assert p.status == "staged", (vid, p.problems, p.error)
        assert p.recording_id == recording_id("aria", vid)
        assert p.report["valid_frac"]["head"] > 0.5, p.report
        pin = read_pin(p.staged)
        dst = tmp_path / "lance_staging" / pin["source"] / pin["recording_id"]
        shutil.copytree(p.staged, dst, ignore=shutil.ignore_patterns(*REVIEW_ONLY))
    report = commit_staged(
        str(tmp_path / "lance_staging"), "aria", str(tmp_path / "lance"), tmp_path / "w"
    )
    assert report["rejected"] == {}
    recs = lance.dataset(tmp_path / "lance" / "recordings.lance").to_table().to_pylist()
    assert sorted(r["vendor_recording_id"] for r in recs) == ids
    assert {r["embodiment"] for r in recs} <= {
        "human_bimanual", "human_left_arm", "human_right_arm",
    }  # fmt: skip
