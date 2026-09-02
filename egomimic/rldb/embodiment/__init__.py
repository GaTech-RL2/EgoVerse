"""Expose embodiment identifiers, classes, and registry lookup functions.

``EMBODIMENT_CLASSES`` maps each canonical embodiment name to its configured
``Embodiment`` subclass. The module builds this mapping from
``registry/platforms.yaml``. It omits platforms that do not specify an
``embodiment_class``.
"""

from egomimic.rldb.embodiment.embodiment import (
    EMBODIMENT,
    EMBODIMENT_ID_TO_KEY,
    Embodiment,
    ResolvedEmbodiment,
    _import_embodiment_class,
    canonical_embodiment_name,
    get_embodiment,
    get_embodiment_id,
)
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.embodiment.registry import load_platforms


def _build_embodiment_classes() -> dict[str, type[Embodiment]]:
    """Build the embodiment class mapping from the platform registry.

    Returns:
        A mapping from each canonical embodiment name to an ``Embodiment``
        subclass. The mapping excludes platforms without an
        ``embodiment_class`` value.
    """
    classes: dict[str, type[Embodiment]] = {}
    for platform in load_platforms().values():
        if platform.embodiment_class is None:
            continue
        cls = _import_embodiment_class(platform.embodiment_class)
        for embodiment in platform.embodiments:
            classes[embodiment] = cls
    return classes


EMBODIMENT_CLASSES: dict[str, type[Embodiment]] = _build_embodiment_classes()


def get_embodiment_class(
    embodiment_name: str | None, default: type[Embodiment] | None = None
) -> type[Embodiment] | None:
    """Return the class for a current or deprecated embodiment name.

    Args:
        embodiment_name: An embodiment name. The lookup ignores letter case and
            applies aliases from ``registry/aliases.yaml``.
        default: The value to return if the name is empty or unknown.

    Returns:
        The configured ``Embodiment`` subclass, or ``default`` if no class
        matches the name.
    """
    if not embodiment_name:
        return default
    return EMBODIMENT_CLASSES.get(canonical_embodiment_name(embodiment_name), default)


__all__ = [
    "EMBODIMENT",
    "EMBODIMENT_CLASSES",
    "EMBODIMENT_ID_TO_KEY",
    "Embodiment",
    "Eva",
    "Human",
    "ResolvedEmbodiment",
    "canonical_embodiment_name",
    "get_embodiment",
    "get_embodiment_class",
    "get_embodiment_id",
]
