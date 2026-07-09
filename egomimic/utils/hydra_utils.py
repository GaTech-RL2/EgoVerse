import os
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

HYDRA_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hydra_configs"
)


def load_config(
    config_name: str,
    overrides: list[str] | None = None,
    config_dir: str | None = None,
) -> DictConfig:
    """
    Load a Hydra config by name, resolving the full defaults chain.

    For config group paths (e.g. "data/cotrain_pi_lang"), the returned
    DictConfig contains the group contents directly (not nested under
    the group key).

    Args:
        config_name: Name of the config file (without .yaml extension).
            Can also be a config group path like "data/cotrain_pi_lang".
        overrides: Optional list of Hydra overrides (e.g. ["+trainer=debug"]).
        config_dir: Optional override of the Hydra config search directory.
            Defaults to the ``egomimic/hydra_configs`` dir of the *installed*
            ``egomimic`` package. Pass an explicit dir when the caller wants
            to load a config from a different checkout (see
            :func:`load_config_from_path`).

    Returns:
        Fully composed DictConfig.
    """
    config_dir = config_dir or HYDRA_CONFIG_DIR
    gh = GlobalHydra.instance()
    was_initialized = gh.is_initialized()
    if was_initialized:
        gh.clear()
    try:
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(config_name=config_name, overrides=overrides or [])
        # Config groups (e.g. "data/foo") get nested under the group key.
        # Unwrap if the config has exactly one key matching the group name.
        if "/" in config_name:
            group_key = config_name.split("/")[0]
            if list(cfg.keys()) == [group_key]:
                cfg = cfg[group_key]
        return cfg
    finally:
        if was_initialized:
            gh.clear()


def find_hydra_config_dir(config_path: str | Path) -> str:
    """
    Walk upward from a Hydra config file to find the nearest ``hydra_configs``
    ancestor, returning it as a string.

    This avoids hardcoding :data:`HYDRA_CONFIG_DIR`, which always points at the
    *installed* ``egomimic`` package — wrong when the caller runs a script from
    a sibling checkout and passes a config path inside *that* checkout.
    """
    cur = Path(config_path).resolve().parent
    for ancestor in (cur, *cur.parents):
        if ancestor.name == "hydra_configs":
            return str(ancestor)
    raise ValueError(
        f"Could not locate a 'hydra_configs/' ancestor for {config_path}. "
        "Pass a config that lives under a hydra_configs/ directory."
    )


def find_run_snapshot_path(start_path: str | Path) -> str | None:
    """Walk up from ``start_path`` looking for a sibling ``.hydra/config.yaml``.

    Hydra writes the fully-composed config snapshot for a training run into
    ``<run_dir>/.hydra/config.yaml``. Given a path inside that run (e.g. a
    checkpoint at ``<run_dir>/checkpoints/foo.ckpt``), this returns the
    snapshot path, or ``None`` if no ancestor has one.
    """
    d = os.path.dirname(os.path.abspath(str(start_path)))
    seen: set[str] = set()
    while d and d not in seen:
        seen.add(d)
        candidate = os.path.join(d, ".hydra", "config.yaml")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def load_run_snapshot(
    start_path: str | Path,
    *,
    resolve: bool = False,
) -> dict[str, Any] | None:
    """Load the fully-composed hydra config snapshot for a training run.

    Walks up from ``start_path`` (typically a checkpoint path) to find the
    nearest ``<run_dir>/.hydra/config.yaml`` and loads it as a plain dict.

    ``resolve`` defaults to ``False`` because run snapshots often contain
    interpolations like ``${data.dataset.data_schematic}`` that reference
    runtime-only nodes and would crash on resolution. Set ``resolve=True``
    only when the caller knows all interpolations are self-contained.

    Returns ``None`` if no snapshot is found in any ancestor.
    """
    snapshot = find_run_snapshot_path(start_path)
    if snapshot is None:
        return None
    return OmegaConf.to_container(OmegaConf.load(snapshot), resolve=resolve)


def load_config_from_path(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> DictConfig:
    """
    Load a Hydra config given an on-disk file path (instead of a config name).

    The Hydra search dir is derived from the nearest ``hydra_configs/``
    ancestor of ``config_path`` via :func:`find_hydra_config_dir`, and the
    config name is taken as the path relative to that dir (without the
    ``.yaml`` suffix). The full ``defaults:`` chain is resolved against that
    same directory, so configs that ``defaults:``-import siblings work the
    same way as ``load_config``.
    """
    abs_path = os.path.abspath(str(config_path))
    config_dir = find_hydra_config_dir(abs_path)
    rel_path = os.path.relpath(abs_path, config_dir)
    config_name = os.path.splitext(rel_path)[0]
    return load_config(config_name, overrides=overrides, config_dir=config_dir)
