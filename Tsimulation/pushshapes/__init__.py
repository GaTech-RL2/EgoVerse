"""PushShapes: T/U/Z pushing env with multiple pushers and obstacle levels.

VERSIONED SIM. Each version is a frozen package under this one:

    v1  original per-level obstacle geometry (no solid_pusher, no pocket friction)
    v2  rewritten obstacle generators; solid_pusher; socket grips on all faces
    v3  v2 + pocket-bottom-only socket friction (CURRENT)

Pick a version explicitly when reproducing a historical eval::

    from Tsimulation.pushshapes import get_env
    Env = get_env("v2")
    env = Env(pusher_shape="u_socket", obstacle_level=15)

`PushShapesEnv` (unversioned) resolves to CURRENT_VERSION so existing callers
keep working. ⚠️ Data does not record which sim produced it unless the writer
stamped SIM_VERSION -- see collect/zarr_writer.py.
"""

from gymnasium.envs.registration import register

from . import v1, v2, v3
from .v3 import obstacles, render, shapes  # current-version convenience aliases

CURRENT_VERSION = "v3"

_VERSIONS = {"v1": v1, "v2": v2, "v3": v3}


def available_versions():
    """Sorted list of sim versions that can be instantiated."""
    return sorted(_VERSIONS)


def get_module(version=None):
    """Return the sim package for `version` (default: CURRENT_VERSION)."""
    v = str(version or CURRENT_VERSION).lower()
    if not v.startswith("v"):
        v = "v" + v
    if v not in _VERSIONS:
        raise ValueError(
            "unknown sim version %r; available: %s" % (version, available_versions()))
    return _VERSIONS[v]


def get_env(version=None):
    """Return the PushShapesEnv CLASS for `version`."""
    return get_module(version).PushShapesEnv


PushShapesEnv = get_env(CURRENT_VERSION)

# ---------------------------------------------------------------------------
# Back-compat: the historical module paths
#     Tsimulation.pushshapes.{env,obstacles,shapes,render}
# now live under the versioned packages. Alias them to CURRENT_VERSION so every
# existing `from Tsimulation.pushshapes.env import PushShapesEnv` keeps working
# and keeps seeing the same module object (constants included).
# To pin a version explicitly use get_env("v2") / get_module("v2").
import sys as _sys

_current = _VERSIONS[CURRENT_VERSION]
for _name in ("env", "obstacles", "shapes", "render"):
    _sys.modules[__name__ + "." + _name] = getattr(_current, _name)
del _sys, _current, _name


for _v in _VERSIONS:
    register(id="PushShapes-%s" % _v,
             entry_point="Tsimulation.pushshapes.%s.env:PushShapesEnv" % _v)
register(id="PushShapes-v0",
         entry_point="Tsimulation.pushshapes.%s.env:PushShapesEnv" % CURRENT_VERSION)

__all__ = ["PushShapesEnv", "get_env", "get_module",
           "available_versions", "CURRENT_VERSION",
           "obstacles", "shapes", "render", "v1", "v2", "v3"]
