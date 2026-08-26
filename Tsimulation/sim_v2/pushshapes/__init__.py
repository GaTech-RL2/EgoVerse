"""PushShapes: T/U/Z pushing env with multiple pushers and obstacle levels.

VERSIONED SIM -- one selection mechanism, one copy of each version on disk:

    Tsimulation/sim_v1/   frozen original simulator
    Tsimulation/sim_v2/   CURRENT fixed simulator (this package)
    Tsimulation          compatibility link to Tsimulation/sim_v2

Frozen versions are separate TOP-LEVEL packages because sys.modules keys on the
package name -- two trees both called `Tsimulation` could never be imported in
one process, which is what makes side-by-side comparison possible here.

    from Tsimulation.pushshapes import get_env
    Env = get_env("v2")          # reproduce data collected under v2
    env = Env(pusher_shape="u_socket", obstacle_level=15)

`PushShapesEnv` resolves to the CURRENT version, so existing callers are
unchanged. Current collections record `sim_version` in `episode_init`; older
data must be dated by its dataset, not inferred.
"""

import importlib
import os as _os
import sys as _sys

from gymnasium.envs.registration import register

from .env import PushShapesEnv

CURRENT_VERSION = "v2"

# Resolve the physical versioned layout even when this package was imported
# through the repository-level Tsimulation compatibility symlink.
_REAL_FILE = _os.path.realpath(__file__)
_TSIMULATOR = _os.path.dirname(_os.path.dirname(_os.path.dirname(_REAL_FILE)))
_REPO_ROOT = _os.path.dirname(_TSIMULATOR)
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

# every spelling that has ever selected a version, mapped to one canonical key
# Two versions only. "v3" is accepted as a historical misnomer for the fixed
# current sim -- what shipped as v2 -- so older commands keep working.
_ALIASES = {
    "v1": "v1", "1": "v1", "tsimulation_v1": "v1", "tsimulation_legacy": "v1",
    "v2": "v2", "2": "v2", "tsimulation_v2": "v2",
    "v3": "v2", "3": "v2", "tsimulation_v3": "v2",
    "tsimulation": CURRENT_VERSION, "": CURRENT_VERSION,
}
_PACKAGES = {"v1": "Tsimulation.sim_v1", "v2": "Tsimulation"}


def available_versions():
    """Sorted sim versions that can be instantiated."""
    return ["v1", "v2"]


def normalize_version(version=None):
    """Map any historical spelling (incl. PUSHSHAPES_SIM values) to v1/v2."""
    key = str(version if version is not None else "").strip().lower()
    if key not in _ALIASES:
        raise ValueError("unknown sim version %r; available: %s (aliases: %s)"
                         % (version, available_versions(), sorted(_ALIASES)))
    return _ALIASES[key]


def package_name(version=None):
    """Top-level package for `version`, e.g. "Tsimulation.sim_v1".

    Needed by callers importing sibling modules such as `<pkg>.collect.zarr_writer`.
    """
    return _PACKAGES[normalize_version(version)]


def get_module(version=None):
    """Return the `pushshapes` module for `version`."""
    v = normalize_version(version)
    if v == CURRENT_VERSION:
        return _sys.modules[__name__]
    return importlib.import_module(_PACKAGES[v] + ".pushshapes")


def get_env(version=None):
    """Return the PushShapesEnv CLASS for `version` (default: current)."""
    return get_module(version).PushShapesEnv


def version_from_env(default=None):
    """Resolve the PUSHSHAPES_SIM env var through the same aliases."""
    return normalize_version(_os.environ.get("PUSHSHAPES_SIM", default or ""))


register(id="PushShapes-v0", entry_point="Tsimulation.pushshapes.env:PushShapesEnv")

__all__ = ["PushShapesEnv", "get_env", "get_module", "package_name",
           "available_versions",
           "normalize_version", "version_from_env", "CURRENT_VERSION"]
