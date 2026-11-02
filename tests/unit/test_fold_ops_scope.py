"""Durable authorization governs repairs without changing experiment provenance."""

import json
import subprocess
import sys
from unittest.mock import patch

import pytest

from experiments.fold_clothes_moving_opscale.operations import (
    authorization_scope as auth,
)
from experiments.fold_clothes_moving_opscale.operations import (
    install_operations as installer,
)
from experiments.fold_clothes_moving_opscale.operations import (
    recovery_controller as recovery,
)

PRODUCTION = {
    "schema_version": 1,
    "allow_training": True,
    "scope": "production",
    "user_instruction": "Run all 36 existing experiments and monitor them.",
}


def write_control(root, control):
    (root / "production_control.json").write_text(json.dumps(control))


@pytest.mark.parametrize(
    "control",
    [
        {"allow_training": False},
        {"schema_version": 1, "allow_training": False, "scope": "preflight_only"},
        PRODUCTION,
    ],
)
def test_explicit_scope_and_recovery_policy_follow_current_control(tmp_path, control):
    write_control(tmp_path, control)
    snapshot = auth.require_scope(tmp_path)
    assert snapshot["allow_training"] is control["allow_training"]
    policy = recovery.recovery_policy(tmp_path)
    assert policy["authorization_scope"] == snapshot
    assert policy["production_source_migration"] == "blocked"
    assert policy["protected_invariants"] == recovery.POLICY["protected_invariants"]
    assert auth.require_scope(tmp_path, snapshot) == snapshot


@pytest.mark.parametrize(
    "control",
    [
        {},
        [],
        None,
        {"allow_training": True},
        {**PRODUCTION, "allow_training": "true"},
        {**PRODUCTION, "allow_training": 1},
        {**PRODUCTION, "schema_version": True},
        {**PRODUCTION, "schema_version": 2},
        {k: v for k, v in PRODUCTION.items() if k != "schema_version"},
        {**PRODUCTION, "user_instruction": " "},
        {**PRODUCTION, "user_instruction": ["yes"]},
        {**PRODUCTION, "scope": "preflight_only"},
        {**PRODUCTION, "allow_training": False},
        {"allow_training": False, "scope": "unknown"},
        {"authorization_record": "/any/path"},
    ],
)
def test_malformed_or_incomplete_controls_fail_closed(tmp_path, control):
    write_control(tmp_path, control)
    snapshot = auth.read_scope(tmp_path)
    assert snapshot["scope"] == "preflight_only"
    assert snapshot["allow_training"] is False
    assert snapshot["valid"] is False
    with pytest.raises(RuntimeError, match="Invalid production control"):
        auth.require_scope(tmp_path)
    assert recovery.recovery_policy(tmp_path)["allow_production_training"] is False


@pytest.mark.parametrize(
    "contents", [None, "{broken", '{"allow_training":false,"allow_training":true}']
)
def test_missing_corrupt_and_duplicate_controls_cannot_grant_authority(
    tmp_path, contents
):
    if contents is not None:
        (tmp_path / "production_control.json").write_text(contents)
    assert auth.read_scope(tmp_path)["allow_training"] is False
    with pytest.raises(RuntimeError):
        auth.require_scope(tmp_path)


def test_every_authorization_change_invalidates_attempt_but_new_attempt_can_read(
    tmp_path,
):
    write_control(tmp_path, {"allow_training": False})
    preflight = auth.require_scope(tmp_path)
    write_control(tmp_path, PRODUCTION)
    with pytest.raises(RuntimeError, match="authorization changed"):
        auth.require_scope(tmp_path, preflight)
    production = auth.require_scope(tmp_path)
    assert production["allow_training"] is True
    write_control(tmp_path, {**PRODUCTION, "authorization_record": "/new/audit/record"})
    with pytest.raises(RuntimeError, match="authorization changed"):
        auth.require_scope(tmp_path, production)
    write_control(tmp_path, {"allow_training": False})
    with pytest.raises(RuntimeError, match="authorization changed"):
        auth.require_scope(tmp_path, production)
    assert auth.require_scope(tmp_path)["allow_training"] is False


def test_reviewed_production_install_preserves_training_identity_and_imports(tmp_path):
    ops = tmp_path / "operations"
    ops.mkdir()
    write_control(tmp_path, PRODUCTION)
    (ops / "dispatcher_control.json").write_text('{"enabled":false}')
    pinned = {"repo": "/pinned/source", "commit": "d" * 40}
    (ops / "recovery_config.json").write_text(json.dumps(pinned))
    norms = tmp_path / "norm_complete.json"
    norms.write_text('{"commit":"unchanged-provenance"}')
    prior_norm = norms.read_bytes()
    package = installer.inventory()
    assert "authorization_scope.py" in package["files"]
    receipt = installer.install(tmp_path, package["package_sha256"])
    assert receipt["authorization_scope"] == auth.require_scope(tmp_path)
    assert json.loads((ops / "recovery_config.json").read_text()) == pinned
    assert json.loads((tmp_path / "production_control.json").read_text()) == PRODUCTION
    assert norms.read_bytes() == prior_norm
    for script in (
        "agent_dispatcher.py",
        "recovery_controller.py",
        "install_operations.py",
    ):
        result = subprocess.run(
            [sys.executable, str(ops / script), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "control,allowed",
    [
        (PRODUCTION, True),
        ({"allow_training": False}, True),
        ({"allow_training": True}, False),
    ],
)
def test_controller_refreshes_scope_and_blocks_restart_for_invalid_control(
    tmp_path, control, allowed
):
    ops = tmp_path / "operations"
    ops.mkdir()
    write_control(tmp_path, control)
    (ops / "recovery_config.json").write_text('{"configured":true}')
    with patch.object(recovery, "inspect_local", return_value={}) as inspect:
        recovery.reconcile(tmp_path, apply=True)
    assert inspect.call_args.args[-1] is allowed
    policy = json.loads((ops / "recovery_policy.json").read_text())
    assert policy["authorization_scope"] == auth.read_scope(tmp_path)
