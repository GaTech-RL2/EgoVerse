"""Versioned PushShapes simulators used by EgoVerse.

Layout:
    Tsimulation/
        sim_v1/   frozen original simulator
        sim_v2/   current simulator (solid-pusher collision, pocket-bottom-only
                  U-socket friction, arena edge guards)

The submodules of the ACTIVE version are aliased to the top level, so the ~150
existing call sites that do ``from Tsimulation.pushshapes import X`` or
``PUSHSHAPES_SIM=Tsimulation`` keep working unchanged after the consolidation
that removed the separate ``Tsimulation`` package and its compatibility symlink.

Select a version with the ``TSIM_VERSION`` env var (``sim_v1`` / ``sim_v2``);
default is ``sim_v2``. Unlike the old symlink this is per-process and visible in
the job environment, so a run cannot silently pick up a different simulator than
it reports. Jobs should still assert the version they expect, e.g.::

    from Tsimulation.pushshapes import env
    assert env.SIM_VERSION == 2
"""
import importlib as _importlib
import os as _os
import sys as _sys

__all__ = ["sim_v1", "sim_v2", "ACTIVE"]

ACTIVE = _os.environ.get("TSIM_VERSION", "sim_v2")
if ACTIVE not in ("sim_v1", "sim_v2"):
    raise ImportError(
        f"TSIM_VERSION={ACTIVE!r} is not a supported simulator; use sim_v1 or sim_v2"
    )

_active_pkg = _importlib.import_module(f"{__name__}.{ACTIVE}")

# Alias the active version's submodules to the top level so legacy imports
# resolve. Missing submodules are skipped rather than raising: sim_v1 and
# sim_v2 do not expose an identical set.
for _sub in (
    "pushshapes",
    "collect",
    "examples",
    "tests",
    "visualize_episode",
):
    try:
        _module = _importlib.import_module(
            f"{__name__}.{ACTIVE}.{_sub}"
        )
        _sys.modules[f"{__name__}.{_sub}"] = _module
        globals()[_sub] = _module
    except ModuleNotFoundError:
        pass

# Re-export the active package's public names (e.g. get_env) at the top level.
for _name in getattr(_active_pkg, "__all__", []):
    globals()[_name] = getattr(_active_pkg, _name)
