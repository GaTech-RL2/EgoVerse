"""Path defaults for scripts that run outside hydra.

A standalone script (a viz tool, a probe) still wants the cluster's dataset
mirror, and a second copy of the path silently drifts from the config the
training runs use. This reads the one hydra defines.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

import egomimic.hydra_configs as _cfg_pkg

_PATHS_YAML = Path(_cfg_pkg.__file__).parent / "paths" / "default.yaml"


def default_dataset_dir() -> str:
    """``paths.dataset_dir`` from the hydra configs, env override included."""
    return str(OmegaConf.load(_PATHS_YAML).dataset_dir)
