"""Preserve failed reset evidence without accepting a mismatched comparison."""

import copy
import io
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from astra_reversal.hardware_interface.robocasa_worker import (
    record_reset_comparison,
    record_reset_evidence,
)
from experiments.liberomax_hardware_interface.tools.prepare_plus_assets import (
    asset_members,
)


def test_capture_precedes_parity_rejection(tmp_path):
    env = SimpleNamespace(
        reset_receipt={"state_sha256": "original", "initial_success": False},
        reset_model_xml="<mujoco/>",
        env=SimpleNamespace(
            model=SimpleNamespace(get_xml=lambda: "<source/>"),
            sim=SimpleNamespace(get_state=lambda: np.array([1.0, 2.0])),
        ),
    )
    proxy = SimpleNamespace(
        native_keys={"sensor"},
        observe=lambda keys: {"observations": {"sensor": [3.0]}},
    )
    reset = record_reset_evidence(env, proxy, tmp_path)
    assert (tmp_path / "reset-model.xml").read_text() == "<mujoco/>"
    assert np.array_equal(np.load(tmp_path / "reset-state.npz")["state"], [1, 2])
    assert record_reset_comparison(reset, copy.deepcopy(reset), tmp_path)
    changed = copy.deepcopy(reset)
    changed["receipt"]["state_sha256"] = "different"
    changed["observations_sha256"] = "different-image"
    assert not record_reset_comparison(reset, changed, tmp_path)
    receipt = json.loads((tmp_path / "reset-mismatch.json").read_text())
    assert receipt["different_receipt_fields"] == ["state_sha256"]
    assert not receipt["observations_equal"]
    assert receipt["reference"] == reset and receipt["actual"] == changed


@pytest.mark.parametrize("name", ["../source.py", "/tmp/scene.xml", "libero/source.py"])
def test_asset_archive_cannot_replace_benchmark_source(name):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, "content")
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive:
        with pytest.raises(ValueError, match="outside_assets"):
            asset_members(archive)


def test_official_asset_tree_is_accepted():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("assets/scenes/kitchen/table.xml", "<mujoco/>")
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive:
        assert len(asset_members(archive)) == 1
