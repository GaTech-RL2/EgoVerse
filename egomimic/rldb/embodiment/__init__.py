"""Embodiment package — the one place that says which embodiments exist.

``EMBODIMENT_CLASSES`` maps an episode's ``embodiment`` attribute to the class
that owns its keymap, transform pipeline and overlays. The same six-entry dict
used to be pasted into ``egomimic/scripts/viz_language.py`` and into
``egomimic/scripts/data_visualization/inspector_lib/dataset_view.py``, the
second copy carrying a comment asking a future editor to keep it in sync with
the first. Both callers now import from here, and the dict itself is *derived*
from ``registry/platforms.yaml`` rather than typed out: a registry the code does
not actually read would be a fifth source of truth beside the four it replaces,
so the registry is the only place an embodiment can be declared.
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
    """Embodiment name -> class, from each platform's ``embodiment_class:``.

    A platform that declares none is skipped: it has no hand-written pipeline
    yet, and `Embodiment.resolve` raises a pointed error rather than handing
    back a class that does not fit.
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
    """Resolve an embodiment name to its class, case-insensitively.

    Deprecated names resolve through the alias table, so an episode written
    before a rename still gets its real overlay rather than the fallback.
    Returns ``default`` for a missing, empty or unknown name — callers read this
    name out of an episode's attrs and want a fallback, not a ``KeyError``.
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
