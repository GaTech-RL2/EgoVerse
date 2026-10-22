"""MCAP -> Lance on real spec deliveries, checked against Ryan's aria_to_lance.

    EGOVERSE_TEST_MCAP_DIR=<dir of episode_*/>
    EGOVERSE_TEST_ARIA_LANCE_DIR=<dir of aria_to_lance deliveries, one per vendor id>  # optional
    pytest --integration tests/integration/test_mcap_to_lance_aria.py

Every episode converts, stages and commits. Where aria_to_lance output for the same recording is
given (run with --no-mano, as here), steps rows line up one to one on the same clock, and
the head pose matches wherever both picked the same pose sample: aria_to_lance uses
projectaria's search, which lands one sample off on about half the frames, while this
converter takes the true nearest sample.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

lance = pytest.importorskip("lance")
pq = pytest.importorskip("pyarrow.parquet")

from egomimic.scripts.lance_commit import commit_staged  # noqa: E402
from egomimic.scripts.mcap_process.mcap_to_lance import convert_and_stage  # noqa: E402


def _codec() -> str:
    out = subprocess.run(
        [shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
    ).stdout.decode()
    for c in ("libsvtav1", "libx264"):
        if c in out:
            return c
    pytest.skip("ffmpeg has neither libsvtav1 nor libx264")


def test_real_episodes_convert_and_match_aria_to_lance(tmp_path):
    root = os.environ.get("EGOVERSE_TEST_MCAP_DIR")
    if not root:
        pytest.skip("set EGOVERSE_TEST_MCAP_DIR to a directory of episode_*/ dirs")
    eps = sorted(p for p in Path(root).glob("episode_*") if p.is_dir())
    ryan_root = os.environ.get("EGOVERSE_TEST_ARIA_LANCE_DIR")
    codec = _codec()
    for ep in eps:
        r = convert_and_stage(
            ep, tmp_path / "staging", tmp_path / "lance", codec=codec, mano=False
        )
        assert r.status == "staged", r
        commit_staged(
            str(tmp_path / "staging"), r.lab, str(tmp_path / "lance"), tmp_path / "work"
        )
        assert r.valid_frac["head"] > 0.9, r
        if not ryan_root:
            continue
        ryan = Path(ryan_root) / ep.name.removeprefix("episode_")
        if not (ryan / "steps.parquet").exists():
            continue
        ours = (
            lance.dataset(tmp_path / "lance" / "steps" / f"{r.lab}.lance")
            .to_table(filter=f"recording_id = '{r.recording_id}'")
            .to_pydict()
        )
        theirs = pq.read_table(ryan / "steps.parquet").to_pydict()
        assert len(ours["timestamp_ns"]) == len(theirs["timestamp_ns"])
        offset = np.array(ours["timestamp_ns"]) - np.array(theirs["timestamp_ns"])
        assert np.ptp(offset) == 0  # same frames; ours Unix ns, theirs device ns
        both = np.array(ours["head_pose_valid"]) & np.array(theirs["head_pose_valid"])
        a = np.array(ours["head_pose"])[both]
        b = np.array(theirs["head_pose"])[both]
        dp = np.linalg.norm(a[:, :3] - b[:, :3], axis=1)
        # Same pose sample on about half the frames (exact), the neighbor on the rest
        # (head moves < 1 mm between 1 kHz samples).
        assert (dp < 1e-6).mean() > 0.4
        assert dp.max() < 2e-3
