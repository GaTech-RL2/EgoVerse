"""Load and validate the embodiment registry YAML files.

``platforms.yaml`` defines arm chains, auxiliary chains, cameras, and reference
frames. ``end_effectors.yaml`` defines joints, keypoint slots, and action
spaces. ``aliases.yaml`` maps deprecated embodiment names to current names.
Episode metadata stores calibration, so this registry does not define
calibration values.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import yaml

# Do not import the parent embodiment package here. The parent package imports
# this module when it builds ``EMBODIMENT_CLASSES``.
REGISTRY_DIR = Path(__file__).parent

PLATFORM_KINDS = frozenset({"robot", "human"})
END_EFFECTOR_CLASSES = frozenset({"parallel_jaw", "human_hand", "dexterous_hand"})
ACTION_SPACES = frozenset({"cartesian", "keypoints"})
#: Number of keypoint slots in each supported topology.
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
    """Report invalid data in a registry YAML file."""


@dataclass(frozen=True)
class KeypointSpec:
    """Identify a keypoint topology and the end-effector's valid slots.

    Attributes:
        topology: A key in ``TOPOLOGY_SLOTS``.
        valid: Zero-based indices of valid slots in the topology. The registry
            loader sorts these indices.
    """

    topology: str
    valid: tuple[int, ...]

    @property
    def n_slots(self) -> int:
        """Return the total number of slots in the topology."""
        return TOPOLOGY_SLOTS[self.topology]

    @property
    def is_complete(self) -> bool:
        return len(self.valid) == self.n_slots


@dataclass(frozen=True)
class AuxChainSpec:
    """Describe the non-arm joints of a platform.

    Attributes:
        dof: The number of joints in the auxiliary chain.
        joint_names: The joint names in chain order. This tuple can be empty.
    """

    dof: int
    joint_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class EndEffectorSpec:
    """Store one validated entry from ``end_effectors.yaml``.

    ``keypoints`` defines the common topology and the slots that this
    end-effector supports. The optional fields store joint metadata and a URDF
    path.
    """

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
    """Store one validated entry from ``platforms.yaml``.

    The entry defines the arm configurations, joint counts, image streams,
    reference frame, and default end-effector for a platform.
    """

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
        """Return ``<embodiment_prefix>_<arity>`` for each supported arity."""
        return tuple(f"{self.embodiment_prefix}_{a}" for a in self.arity)


def _load_yaml(file_name: str) -> dict:
    with (REGISTRY_DIR / file_name).open("r") as f:
        return yaml.safe_load(f) or {}


def _check_fields(block: dict, allowed: frozenset[str], where: str) -> None:
    """Raise ``RegistryError`` if a mapping contains an unknown field."""
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
    """Load the deprecated-to-current embodiment name mapping.

    Returns:
        A dictionary from lowercase deprecated names to lowercase current
        names.
    """
    raw = _load_yaml("aliases.yaml")
    return {str(old).lower(): str(new).lower() for old, new in raw.items()}


@functools.lru_cache(maxsize=None)
def load_end_effectors() -> dict[str, EndEffectorSpec]:
    """Load and validate ``end_effectors.yaml``.

    Returns:
        A dictionary from end-effector names to validated specifications.

    Raises:
        RegistryError: If an entry has an unknown field or an invalid value.
    """
    return {
        name: _parse_end_effector(name, block)
        for name, block in _load_yaml("end_effectors.yaml").items()
    }


@functools.lru_cache(maxsize=None)
def load_platforms() -> dict[str, PlatformSpec]:
    """Load and validate ``platforms.yaml``.

    Returns:
        A dictionary from platform names to validated specifications.

    Raises:
        RegistryError: If an entry is invalid or two platforms produce the
            same embodiment name.
    """
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
    """Map current and deprecated embodiment names to platform specifications.

    The result includes an alias only if its current target identifies a
    platform.
    """
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
