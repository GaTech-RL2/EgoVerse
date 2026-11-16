"""Aria Gen 2 conversion end to end on a real recording.

    EGOVERSE_TEST_ARIA_GEN2_VRS=<dir>/<ts>.vrs   pytest --integration tests/integration/test_aria_gen2_extract.py

<dir>/mps_<ts>_vrs/{hand_tracking,slam} must sit next to the VRS for the MPS
case (the on-device case only needs the VRS). Recordings are ~1-5 GB, so
there is no checked-in fixture.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import zarr

from egomimic.scripts.aria_process.aria_to_zarr import DatasetConverter
from egomimic.utils.aws.aws_sql import timestamp_ms_to_episode_hash

META = dict(task_name="t", operator="o", lab="l", scene="s", objects=["x"])


def _vrs() -> Path:
    path = os.environ.get("EGOVERSE_TEST_ARIA_GEN2_VRS")
    if not path:
        pytest.skip("set EGOVERSE_TEST_ARIA_GEN2_VRS to an Aria Gen 2 <ts>.vrs")
    return Path(path)


@pytest.mark.parametrize("source", ["mps", "on_device"])
def test_gen2_recording_converts_to_valid_demo_episodes(tmp_path, source):
    vrs = _vrs()
    if source == "mps" and not (vrs.parent / f"mps_{vrs.stem}_vrs").exists():
        pytest.skip("no mps_<ts>_vrs/ next to the VRS")
    conv = DatasetConverter(
        raw_path=vrs.parent, fps=30, arm="both", convert_mano=False, gen2_source=source
    )
    parent = timestamp_ms_to_episode_hash(int(vrs.stem))
    try:
        results = conv.extract_episode(
            vrs, required_metadata=META, output_dir=tmp_path, dataset_name=parent
        )
    except ValueError as e:  # e.g. on-device VIO that never initialized
        pytest.skip(f"{source}: {e}")

    hashes = [r["episode_hash"] for r in results]
    assert results and len(set(hashes)) == len(hashes) and parent not in hashes
    for r in results:
        store = zarr.open(str(r["zarr_path"]), mode="r")
        attrs = json.loads((r["zarr_path"] / "zarr.json").read_text())["attributes"]
        n = attrs["total_frames"]
        assert n == r["kept_frames"] > 0
        assert attrs["device"] == "aria_gen2" and attrs["mp_source"] == source
        assert attrs["embodiment"] == "aria_gen2_bimanual"
        assert attrs["parent_episode_hash"] == parent
        assert "left.obs_keypoints" not in store  # MANO-only key
        for key in ("left.obs_ee_pose", "right.obs_ee_pose", "obs_head_pose"):
            pose = np.asarray(store[key][:n])
            assert np.isfinite(pose).all() and (np.abs(pose) < 1e8).all(), key
            np.testing.assert_allclose(
                np.linalg.norm(pose[:, 3:7], axis=1), 1, atol=1e-6
            )
        ts = np.asarray(store["obs_rgb_timestamps_ns"][:n])
        assert (np.diff(ts) > 0).all()
