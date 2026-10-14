"""Limit the native amendment to deterministic ordering of unchanged geometries."""

import hashlib
import xml.etree.ElementTree as ET

import pytest

from astra_reversal.hardware_interface import robocasa
from astra_reversal.hardware_interface import robocasa_compat as compat


@pytest.fixture
def source(tmp_path, monkeypatch):
    before = b"# test native sampler\n" + compat.BEFORE + b"\n"
    after = before.replace(compat.BEFORE, compat.AFTER)
    monkeypatch.setattr(compat, "BEFORE_SHA256", hashlib.sha256(before).hexdigest())
    monkeypatch.setattr(compat, "AFTER_SHA256", hashlib.sha256(after).hexdigest())
    path = tmp_path / compat.COUNTER_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(before)
    return tmp_path, path, before, after


def test_patch_is_exact_and_idempotent(source):
    root, path, before, after = source
    receipt = compat.apply_fixture_order_fix(root)
    assert path.read_bytes() == after
    assert receipt["upstream_sha256"] == hashlib.sha256(before).hexdigest()
    assert compat.apply_fixture_order_fix(root) == receipt
    assert compat.verify_fixture_order_fix(root) == receipt


def test_unrecognized_upstream_content_is_never_overwritten(source):
    root, path, before, _ = source
    changed = before + b"# independent change\n"
    path.write_bytes(changed)
    with pytest.raises(ValueError, match="counter_source_not_pinned"):
        compat.apply_fixture_order_fix(root)
    assert path.read_bytes() == changed
    with pytest.raises(ValueError, match="fix_missing_or_changed"):
        compat.verify_fixture_order_fix(root)


def test_stable_deduplication_preserves_xml_identity_and_candidate_set():
    left = ET.Element("geom", name="left", pos="-1 0 0")
    right = ET.Element("geom", name="right", pos="1 0 0")
    candidates = [right, left, right, left]
    state = {"valid_geoms": candidates}
    exec(compat.AFTER.decode(), {}, state)
    result = state["valid_geoms"]
    assert result == [right, left]
    assert set(result) == set(candidates)
    assert [ET.tostring(g) for g in result] == [ET.tostring(right), ET.tostring(left)]


@pytest.mark.parametrize("extra_change", [False, True])
def test_source_check_accepts_only_the_hash_pinned_amendment(
    source, monkeypatch, extra_change
):
    root, path, _, _ = source
    compat.apply_fixture_order_fix(root)

    def git(args, **kwargs):
        if "rev-parse" in args:
            return robocasa.ROBOCASA_COMMIT + "\n"
        return (
            " M "
            + compat.COUNTER_PATH
            + "\n"
            + (" M another_file.py\n" if extra_change else "")
        )

    monkeypatch.setattr(robocasa.subprocess, "check_output", git)
    if extra_change:
        with pytest.raises(ValueError, match="unexpected_robocasa_source_changes"):
            robocasa.check_source(root, robocasa.ROBOCASA_COMMIT)
    else:
        robocasa.check_source(root, robocasa.ROBOCASA_COMMIT)
        path.write_bytes(path.read_bytes() + b"# changed\n")
        with pytest.raises(ValueError, match="fix_missing_or_changed"):
            robocasa.check_source(root, robocasa.ROBOCASA_COMMIT)
