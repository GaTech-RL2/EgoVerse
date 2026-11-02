"""Reviewed operations install preserves runtime control, incidents and artifacts."""

import fcntl
import json
import tempfile
from pathlib import Path

import pytest

from experiments.fold_clothes_moving_opscale.operations import (
    install_operations as install,
)


def test_install_requires_reviewed_digest_and_preserves_runtime_files():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "run"
        ops = root / "operations"
        ops.mkdir(parents=True)
        (root / "production_control.json").write_text('{"allow_training": false}')
        state = {"incidents": {"example": {"state": "NEEDS_REVIEW", "attempts": 3}}}
        (ops / "recovery_state.json").write_text(json.dumps(state))
        (ops / "private.log").write_text("private runtime contents")
        (ops / "agent_dispatcher.py").write_text("old reviewed code")
        package = install.inventory()
        with pytest.raises(RuntimeError, match="checksum"):
            install.install(root, "wrong")
        result = install.install(root, package["package_sha256"])
        assert result["state"] == "INSTALLED"
        assert json.loads((ops / "recovery_state.json").read_text()) == state
        assert (ops / "private.log").read_text() == "private runtime contents"
        assert json.loads((root / "production_control.json").read_text()) == {
            "allow_training": False
        }
        assert (
            ops / "code_history" / package["package_sha256"] / "agent_dispatcher.py"
        ).read_text() == "old reviewed code"
        assert (
            install.file_hash(ops / "agent_dispatcher.py")
            == package["files"]["agent_dispatcher.py"]
        )


def test_install_refuses_enabled_agents_active_claims_or_incomplete_production_scope():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ops = root / "operations"
        ops.mkdir()
        (root / "production_control.json").write_text('{"allow_training": false}')
        (ops / "dispatcher_control.json").write_text('{"enabled": true}')
        with pytest.raises(RuntimeError, match="Disable dispatch"):
            install.require_quiescent(root)
        (ops / "dispatcher_control.json").write_text('{"enabled": false}')
        (ops / "recovery_state.json").write_text(
            '{"incidents":{"x":{"state":"CLAIMED"}}}'
        )
        with pytest.raises(RuntimeError, match="claims"):
            install.require_quiescent(root)
        (ops / "recovery_state.json").unlink()
        (root / "production_control.json").write_text('{"allow_training": true}')
        with pytest.raises(RuntimeError, match="production control"):
            install.require_quiescent(root)


@pytest.mark.parametrize(
    "daemon_lock", ["agent_dispatcher.lock", "recovery_watch.lock", "supervision.lock"]
)
def test_install_holds_daemon_locks_even_without_status_receipts(tmp_path, daemon_lock):
    ops = tmp_path / "operations"
    ops.mkdir()
    (tmp_path / "production_control.json").write_text('{"allow_training": false}')
    package = install.inventory()
    with (ops / daemon_lock).open("a+") as running_service:
        fcntl.flock(running_service, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            install.install(tmp_path, package["package_sha256"])
    assert not (ops / "operations_install.json").exists()
    assert not (ops / "agent_dispatcher.py").exists()
