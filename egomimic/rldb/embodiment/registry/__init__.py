"""Declarative embodiment registry.

The YAML files beside this module are data, not code: adding an embodiment, a
platform or an end-effector should be a diffable block here rather than an edit
spread across an enum, two class dicts and a hydra config.

The registry is factorized along the two axes that are actually combinatorial —
``platforms.yaml`` (arm chain, aux chain, camera set, reference frame) and
``end_effectors.yaml`` (DOF, keypoint topology, action space) — so a new hand on
a known robot is one block in one file. Calibration is deliberately absent: it
is per-episode data, not a specification, and centralizing it is the failure
mode the schema exists to avoid.

This module imports nothing from ``egomimic.rldb.embodiment``. It is a leaf, so
the parent package (and ``embodiment.py`` itself) can read the registry without
an import cycle. Anything needing the ``EMBODIMENT`` enum belongs one level up.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import yaml

REGISTRY_DIR = Path(__file__).parent

PLATFORM_KINDS = frozenset({"robot", "human"})
END_EFFECTOR_CLASSES = frozenset({"parallel_jaw", "human_hand", "dexterous_hand"})
ACTION_SPACES = frozenset({"cartesian", "keypoints"})
#: Keypoint topology -> number of slots. A topology is a superset; an
#: end-effector declares which of its slots are valid.
TOPOLOGY_SLOTS = {"mano21": 21}

_PLATFORM_FIELDS = frozenset(
    {
        "kind",
        "embodiment_prefix",
        "arity",
        "arm_dof",
        "aux",
        "cameras",
        "reference_frame",
        "default_end_effector",
        "embodiment_class",
    }
)
_END_EFFECTOR_FIELDS = frozenset(
    {
        "class",
        "dof",
        "action_space",
        "keypoints",
        "joint_names",
        "joint_limits",
        "urdf",
        "dead_dims",
    }
)
_AUX_FIELDS = frozenset({"dof", "joint_names"})
_KEYPOINT_FIELDS = frozenset({"topology", "valid"})


class RegistryError(ValueError):
    """A registry file is malformed.

    Raised at load, not at use: a stale or typo'd registry must break the
    embodiment that depends on it immediately, otherwise the YAML becomes a
    fifth place to drift instead of the one place that replaces the other four.
    """


@dataclass(frozen=True)
class KeypointSpec:
    """Which slots of a keypoint topology an end-effector actually has."""

    topology: str
    valid: tuple[int, ...]

    @property
    def n_slots(self) -> int:
        """Total slots in the topology, valid or not."""
        return TOPOLOGY_SLOTS[self.topology]

    @property
    def is_complete(self) -> bool:
        return len(self.valid) == self.n_slots


@dataclass(frozen=True)
class AuxChainSpec:
    """A platform's non-arm joint chain (torso / lift / head)."""

    dof: int
    joint_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class EndEffectorSpec:
    name: str
    ee_class: str
    dof: int | None
    action_space: str
    keypoints: KeypointSpec
    joint_names: tuple[str, ...] = ()
    joint_limits: tuple[tuple[float, float], ...] = ()
    urdf: str | None = None
    dead_dims: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlatformSpec:
    name: str
    kind: str
    embodiment_prefix: str
    arity: tuple[str, ...]
    arm_dof: int | None
    aux: AuxChainSpec | None
    cameras: tuple[str, ...]
    reference_frame: str
    default_end_effector: str
    embodiment_class: str | None = None

    @property
    def embodiments(self) -> tuple[str, ...]:
        """The EMBODIMENT names this platform owns, e.g. ``eva_bimanual``."""
        return tuple(f"{self.embodiment_prefix}_{a}" for a in self.arity)


def _load_yaml(file_name: str) -> dict:
    with (REGISTRY_DIR / file_name).open("r") as f:
        return yaml.safe_load(f) or {}


def _check_fields(block: dict, allowed: frozenset[str], where: str) -> None:
    """Reject unknown keys — a typo'd field would otherwise be silently ignored."""
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise RegistryError(
            f"{where}: unknown field(s) {unknown}; expected some of {sorted(allowed)}"
        )


def _require(block: dict, field: str, where: str):
    if field not in block:
        raise RegistryError(f"{where}: missing required field {field!r}")
    return block[field]


def _parse_keypoints(raw, where: str) -> KeypointSpec:
    if not isinstance(raw, dict):
        raise RegistryError(f"{where}: `keypoints` must be a mapping, got {raw!r}")
    _check_fields(raw, _KEYPOINT_FIELDS, f"{where}.keypoints")
    topology = _require(raw, "topology", f"{where}.keypoints")
    if topology not in TOPOLOGY_SLOTS:
        raise RegistryError(
            f"{where}.keypoints: unknown topology {topology!r}; "
            f"expected one of {sorted(TOPOLOGY_SLOTS)}"
        )
    n_slots = TOPOLOGY_SLOTS[topology]
    valid = _require(raw, "valid", f"{where}.keypoints")
    if valid == "all":
        return KeypointSpec(topology=topology, valid=tuple(range(n_slots)))
    if not isinstance(valid, list) or not valid:
        raise RegistryError(
            f"{where}.keypoints: `valid` must be 'all' or a non-empty list of "
            f"slot indices, got {valid!r}"
        )
    out = []
    for slot in valid:
        if not isinstance(slot, int) or isinstance(slot, bool):
            raise RegistryError(f"{where}.keypoints: slot {slot!r} is not an int")
        if not 0 <= slot < n_slots:
            raise RegistryError(
                f"{where}.keypoints: slot {slot} out of range for {topology} "
                f"(0..{n_slots - 1})"
            )
        out.append(slot)
    if len(set(out)) != len(out):
        raise RegistryError(f"{where}.keypoints: duplicate slots in {valid!r}")
    return KeypointSpec(topology=topology, valid=tuple(sorted(out)))


def _parse_aux(raw, where: str) -> AuxChainSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RegistryError(f"{where}: `aux` must be null or a mapping, got {raw!r}")
    _check_fields(raw, _AUX_FIELDS, f"{where}.aux")
    dof = _require(raw, "dof", f"{where}.aux")
    if not isinstance(dof, int) or dof <= 0:
        raise RegistryError(f"{where}.aux: `dof` must be a positive int, got {dof!r}")
    joint_names = tuple(raw.get("joint_names") or ())
    if joint_names and len(joint_names) != dof:
        raise RegistryError(
            f"{where}.aux: {len(joint_names)} joint_names for dof {dof}"
        )
    return AuxChainSpec(dof=dof, joint_names=joint_names)


def _parse_end_effector(name: str, block: dict) -> EndEffectorSpec:
    where = f"end_effectors.yaml[{name}]"
    if not isinstance(block, dict):
        raise RegistryError(f"{where}: must be a mapping, got {block!r}")
    _check_fields(block, _END_EFFECTOR_FIELDS, where)

    ee_class = _require(block, "class", where)
    if ee_class not in END_EFFECTOR_CLASSES:
        raise RegistryError(
            f"{where}: unknown class {ee_class!r}; "
            f"expected one of {sorted(END_EFFECTOR_CLASSES)}"
        )
    action_space = _require(block, "action_space", where)
    if action_space not in ACTION_SPACES:
        raise RegistryError(
            f"{where}: unknown action_space {action_space!r}; "
            f"expected one of {sorted(ACTION_SPACES)}"
        )

    dof = _require(block, "dof", where)
    if dof is not None and (not isinstance(dof, int) or dof <= 0):
        raise RegistryError(
            f"{where}: `dof` must be null or a positive int, got {dof!r}"
        )

    joint_names = tuple(block.get("joint_names") or ())
    if joint_names and dof is not None and len(joint_names) != dof:
        raise RegistryError(f"{where}: {len(joint_names)} joint_names for dof {dof}")

    raw_limits = block.get("joint_limits") or ()
    limits: list[tuple[float, float]] = []
    for limit in raw_limits:
        if not isinstance(limit, (list, tuple)) or len(limit) != 2:
            raise RegistryError(
                f"{where}: joint_limits entry {limit!r} is not [lo, hi]"
            )
        lo, hi = float(limit[0]), float(limit[1])
        if lo > hi:
            raise RegistryError(f"{where}: joint limit [{lo}, {hi}] has lo > hi")
        limits.append((lo, hi))
    if limits and dof is not None and len(limits) != dof:
        raise RegistryError(f"{where}: {len(limits)} joint_limits for dof {dof}")

    dead_dims = tuple(block.get("dead_dims") or ())
    for dim in dead_dims:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise RegistryError(f"{where}: dead_dims entry {dim!r} is not an int")
        if dof is None or not 0 <= dim < dof:
            raise RegistryError(f"{where}: dead dim {dim} out of range for dof {dof!r}")

    return EndEffectorSpec(
        name=name,
        ee_class=ee_class,
        dof=dof,
        action_space=action_space,
        keypoints=_parse_keypoints(_require(block, "keypoints", where), where),
        joint_names=joint_names,
        joint_limits=tuple(limits),
        urdf=block.get("urdf"),
        dead_dims=dead_dims,
    )


def _parse_platform(name: str, block: dict, end_effectors: dict[str, EndEffectorSpec]):
    where = f"platforms.yaml[{name}]"
    if not isinstance(block, dict):
        raise RegistryError(f"{where}: must be a mapping, got {block!r}")
    _check_fields(block, _PLATFORM_FIELDS, where)

    kind = _require(block, "kind", where)
    if kind not in PLATFORM_KINDS:
        raise RegistryError(
            f"{where}: unknown kind {kind!r}; expected one of {sorted(PLATFORM_KINDS)}"
        )

    arity = tuple(_require(block, "arity", where) or ())
    if not arity:
        raise RegistryError(f"{where}: `arity` must be a non-empty list")
    if len(set(arity)) != len(arity):
        raise RegistryError(f"{where}: duplicate entries in arity {list(arity)}")

    arm_dof = _require(block, "arm_dof", where)
    if arm_dof is not None and (not isinstance(arm_dof, int) or arm_dof <= 0):
        raise RegistryError(
            f"{where}: `arm_dof` must be null or a positive int, got {arm_dof!r}"
        )

    cameras = tuple(_require(block, "cameras", where) or ())
    if not cameras:
        raise RegistryError(f"{where}: `cameras` must be a non-empty list")
    if len(set(cameras)) != len(cameras):
        raise RegistryError(f"{where}: duplicate entries in cameras {list(cameras)}")

    reference_frame = _require(block, "reference_frame", where)
    if reference_frame.startswith("camera:"):
        camera = reference_frame.split(":", 1)[1]
        if camera not in cameras:
            raise RegistryError(
                f"{where}: reference_frame {reference_frame!r} names a camera that "
                f"is not declared; cameras are {list(cameras)}"
            )
    elif reference_frame not in ("robot_base", "slam_world"):
        raise RegistryError(
            f"{where}: unknown reference_frame {reference_frame!r}; expected "
            "'robot_base', 'slam_world' or 'camera:<declared camera>'"
        )

    default_end_effector = _require(block, "default_end_effector", where)
    if default_end_effector not in end_effectors:
        raise RegistryError(
            f"{where}: default_end_effector {default_end_effector!r} is not in "
            f"end_effectors.yaml; known: {sorted(end_effectors)}"
        )

    return PlatformSpec(
        name=name,
        kind=kind,
        embodiment_prefix=_require(block, "embodiment_prefix", where),
        arity=arity,
        arm_dof=arm_dof,
        aux=_parse_aux(block.get("aux"), where),
        cameras=cameras,
        reference_frame=reference_frame,
        default_end_effector=default_end_effector,
        embodiment_class=block.get("embodiment_class"),
    )


@functools.lru_cache(maxsize=None)
def load_aliases() -> dict[str, str]:
    """Deprecated embodiment name -> current name, lowercased both sides.

    See ``aliases.yaml``: the table is append-only, and it is what keeps a
    rename from being a fleet-wide re-download.
    """
    raw = _load_yaml("aliases.yaml")
    return {str(old).lower(): str(new).lower() for old, new in raw.items()}


@functools.lru_cache(maxsize=None)
def load_end_effectors() -> dict[str, EndEffectorSpec]:
    """End-effector name -> spec, from ``end_effectors.yaml``."""
    return {
        name: _parse_end_effector(name, block)
        for name, block in _load_yaml("end_effectors.yaml").items()
    }


@functools.lru_cache(maxsize=None)
def load_platforms() -> dict[str, PlatformSpec]:
    """Platform name -> spec, from ``platforms.yaml``."""
    end_effectors = load_end_effectors()
    platforms = {
        name: _parse_platform(name, block, end_effectors)
        for name, block in _load_yaml("platforms.yaml").items()
    }
    seen: dict[str, str] = {}
    for platform in platforms.values():
        for embodiment in platform.embodiments:
            if embodiment in seen:
                raise RegistryError(
                    f"platforms.yaml: embodiment {embodiment!r} is claimed by both "
                    f"{seen[embodiment]!r} and {platform.name!r}"
                )
            seen[embodiment] = platform.name
    return platforms


@functools.lru_cache(maxsize=None)
def load_embodiment_platforms() -> dict[str, PlatformSpec]:
    """Embodiment name -> the platform that owns it, aliases included."""
    by_embodiment = {
        embodiment: platform
        for platform in load_platforms().values()
        for embodiment in platform.embodiments
    }
    for old, new in load_aliases().items():
        if new in by_embodiment:
            by_embodiment.setdefault(old, by_embodiment[new])
    return by_embodiment


__all__ = [
    "ACTION_SPACES",
    "END_EFFECTOR_CLASSES",
    "PLATFORM_KINDS",
    "REGISTRY_DIR",
    "TOPOLOGY_SLOTS",
    "AuxChainSpec",
    "EndEffectorSpec",
    "KeypointSpec",
    "PlatformSpec",
    "RegistryError",
    "load_aliases",
    "load_embodiment_platforms",
    "load_end_effectors",
    "load_platforms",
]
