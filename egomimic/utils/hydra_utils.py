import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

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
