"""Kinematic models of dexmate_bimanual for the SharpaDex converter.

``sharpa_dexmate.yaml`` is the spec: which URDF each side uses (with its
sha256), the head link, the hand mount, the aux joint order, every MANO slot's
hand link and the tactile layout. ``load_chain`` reads a URDF named by the
spec and refuses one whose bytes differ from the recorded sha256.

These files ship as package data; nothing here reads the checkout.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path

import yaml

from egomimic.scripts.sharpa_process.kinematics.urdf import (
    UrdfChain,
    UrdfError,
    load_urdf,
)

SPEC_FILE = "sharpa_dexmate.yaml"
SIDES = ("left", "right")
#: MANO-21 slots; every one is filled on the Sharpa Wave hands.
KEYPOINT_SLOTS = 21


def asset_path(name: str) -> Path:
    """Return a file shipped beside this module."""
    return Path(__file__).resolve().parent / name


@functools.cache
def load_spec() -> dict:
    """Return the parsed spec after checking its shape.

    Raises:
        UrdfError: If a required field is missing or a hand does not map all 21
            MANO slots.
    """
    spec = yaml.safe_load(asset_path(SPEC_FILE).read_text())
    platform = spec.get("platform") or {}
    for key in ("urdf", "urdf_sha256", "head_link", "hand_mount_link"):
        if not isinstance(platform.get(key), str):
            raise UrdfError(f"{SPEC_FILE}: platform.{key} must be a string")
    if not isinstance(platform.get("aux_joint_names"), list):
        raise UrdfError(f"{SPEC_FILE}: platform.aux_joint_names must be a list")
    hands = spec.get("hands") or {}
    for side in SIDES:
        hand = hands.get(side) or {}
        for key in ("urdf", "urdf_sha256", "ee_pose_link"):
            if not isinstance(hand.get(key), str):
                raise UrdfError(f"{SPEC_FILE}: hands.{side}.{key} must be a string")
        slots = hand.get("keypoint_links") or {}
        if sorted(slots) != list(range(KEYPOINT_SLOTS)):
            raise UrdfError(
                f"{SPEC_FILE}: hands.{side}.keypoint_links must map slots 0..20"
            )
        tactile = hand.get("tactile") or {}
        if len(tactile.get("channel_names") or ()) != len(
            tactile.get("channel_units") or ()
        ) or len(tactile.get("fingers") or ()) != len(tactile.get("taxel_links") or ()):
            raise UrdfError(f"{SPEC_FILE}: hands.{side}.tactile is inconsistent")
    return spec


def verified_urdf(name: str, sha256: str) -> bytes:
    """Return a shipped URDF's bytes after checking their sha256.

    Raises:
        UrdfError: If the file is missing or its bytes differ from ``sha256``.
    """
    path = asset_path(name)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise UrdfError(f"cannot read {path}: {exc}") from exc
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise UrdfError(f"{path}: sha256 {digest} != {sha256} recorded in {SPEC_FILE}")
    return data


@functools.cache
def load_chain(name: str, sha256: str) -> UrdfChain:
    """Load and cache a shipped URDF named by the spec."""
    return load_urdf(verified_urdf(name, sha256), name)


__all__ = [
    "KEYPOINT_SLOTS",
    "SIDES",
    "UrdfChain",
    "UrdfError",
    "asset_path",
    "load_chain",
    "load_spec",
    "load_urdf",
    "verified_urdf",
]
