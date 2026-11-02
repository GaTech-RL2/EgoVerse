"""Read durable user authorization without granting authority from incident text."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate control field")
        result[key] = value
    return result


def read_scope(root):
    """Default to no production permission; invalid control also blocks repairs.

    Legacy explicit false controls remain supported. Production permission needs
    every field in the versioned authorization contract, with exact JSON types.
    The digest binds all phases to one immutable authorization snapshot.
    """
    snapshot = {
        "scope": "preflight_only",
        "allow_training": False,
        "valid": False,
        "control_sha256": None,
        "reason": "Missing production control",
    }
    try:
        data = (Path(root) / "production_control.json").read_bytes()
        snapshot["control_sha256"] = hashlib.sha256(data).hexdigest()
        control = json.loads(data, object_pairs_hook=_unique_object)
        if not isinstance(control, dict):
            raise ValueError("Production control must be an object")
        version = control.get("schema_version", 1)
        if type(version) is not int or version != 1:
            raise ValueError("Unsupported production control schema")
        allowed = control.get("allow_training")
        scope = control.get("scope")
        if allowed is False and scope in (None, "preflight_only"):
            snapshot.update(valid=True, reason="Explicit preflight-only control")
        elif (
            allowed is True
            and scope == "production"
            and control.get("schema_version") == 1
            and isinstance(control.get("user_instruction"), str)
            and control["user_instruction"].strip()
        ):
            snapshot.update(
                valid=True,
                scope="production",
                allow_training=True,
                reason="Explicit durable production authorization",
            )
        else:
            raise ValueError("Incomplete or contradictory production authorization")
    except (OSError, ValueError, UnicodeError) as error:
        snapshot["reason"] = str(error)
    return snapshot


def require_scope(root, expected=None):
    scope = read_scope(root)
    if not scope["valid"]:
        raise RuntimeError("Invalid production control: " + scope["reason"])
    if expected is not None and scope != expected:
        raise RuntimeError(
            "Production authorization changed; stop and reassess the incident"
        )
    return scope


def scope_instructions(scope):
    if not scope.get("valid"):
        raise RuntimeError("Cannot start an agent without valid production control")
    if scope["allow_training"]:
        return (
            "CURRENT AUTHORIZED SCOPE: production. Incident repair may cover the "
            "existing 36-run experiment, subject to every unchanged source, data, "
            "checkpoint, review and launch gate. The existing pipeline owns initial "
            "production submission; do not submit initial runs directly. Production "
            "source migration is blocked. No blind reruns."
        )
    return (
        "CURRENT AUTHORIZED SCOPE: preflight only. Complete data verification, "
        "normalization and required short GPU evaluation/checkpoint-resume smoke "
        "checks. Never launch, resume or relaunch production training."
    )
