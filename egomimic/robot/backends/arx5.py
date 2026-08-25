"""Compatibility boundary for the optional native ARX5 controller binding.

The maintained upstream wheel exposes ``arx5_interface`` while the historical
EgoVerse image copied its extension under ``arx5.arx5_interface``.  Nothing
outside this module should depend on either packaging detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType


@dataclass(frozen=True)
class Arx5API:
    module: ModuleType
    joint_controller: type
    joint_state: type
    import_name: str


class Arx5Unavailable(ImportError):
    """Raised only when live hardware is requested without an ARX5 binding."""


def load_arx5_api() -> Arx5API:
    """Load the upstream package first, then the legacy copied extension."""

    failures = []
    for import_name in ("arx5_interface", "arx5.arx5_interface"):
        try:
            module = import_module(import_name)
            return Arx5API(
                module=module,
                joint_controller=module.Arx5JointController,
                joint_state=module.JointState,
                import_name=import_name,
            )
        except (ImportError, AttributeError) as error:
            failures.append(f"{import_name}: {type(error).__name__}: {error}")
    detail = "; ".join(failures)
    raise Arx5Unavailable(
        "Live ARX5 control needs a CPython-compatible arx5-interface wheel. "
        "Offline rollout does not require it. Tried the upstream and legacy "
        f"module layouts ({detail})."
    )


def optional_arx5_api() -> Arx5API | None:
    """Return ``None`` when the optional native binding is unavailable."""

    try:
        return load_arx5_api()
    except Arx5Unavailable:
        return None
