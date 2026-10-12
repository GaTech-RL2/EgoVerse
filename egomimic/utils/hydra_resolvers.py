"""Project-wide OmegaConf resolvers.

Importing this module registers custom resolvers used by Hydra configs:
  - ``run_dir``: picks output dir based on train vs eval mode
  - ``model_type_from_ckpt``: extracts model type from checkpoint path
  - ``model_time_from_ckpt``: extracts timestamp from checkpoint path
  - ``mecka_private``: an ID list kept out of the public repo

Also ``apply_experiment_wandb_tag``, which trainHydra calls before building loggers.

Safe to import multiple times (registration is idempotent).
"""

from __future__ import annotations

import functools
import json
import os
import re
from pathlib import Path

from omegaconf import OmegaConf, open_dict


def model_dir_name_from_ckpt(ckpt_path):
    """Extract the model-folder name from a ckpt path like
    ``logs/pick_place/finetuned/<NAME>/0/checkpoints/...``."""
    if not ckpt_path or ckpt_path == "null":
        return None
    parts = str(ckpt_path).replace("\\", "/").split("/")
    if "finetuned" in parts:
        i = parts.index("finetuned")
        if i + 1 < len(parts):
            return parts[i + 1]
    if len(parts) >= 4:
        return parts[-4]
    return None


def model_type_from_ckpt(ckpt_path):
    """Strip trailing ``_YYYY-MM-DD_HH-MM-SS`` from the model dir name.
    Returns ``'pretrained'`` when no checkpoint is supplied."""
    name = model_dir_name_from_ckpt(ckpt_path)
    if name is None:
        return "pretrained"
    m = re.match(r"^(.+?)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$", name)
    return m.group(1) if m else name


def model_time_from_ckpt(ckpt_path):
    """Extract the trailing ``YYYY-MM-DD_HH-MM-SS`` from the model dir name.
    Returns ``'base'`` when no checkpoint is supplied."""
    name = model_dir_name_from_ckpt(ckpt_path)
    if name is None:
        return "base"
    m = re.match(r"^.+?_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$", name)
    return m.group(1) if m else "unknown"


def experiment_wandb_tag(config_name):
    """Folder directly under ``hydra_configs/experiments/`` for this config.

    ``experiments/fixed_budget_opscale/rdt_1b_opscale1`` -> ``fixed_budget_opscale``.
    Configs that do not live in that tree have no experiment tag.
    """
    parts = str(config_name or "").replace("\\", "/").split("/")
    if len(parts) >= 3 and parts[0] == "experiments" and parts[1]:
        return parts[1]
    return None


def apply_experiment_wandb_tag(cfg, config_name) -> None:
    """Add the experiments-folder name to ``logger.wandb.tags`` when present."""
    tag = experiment_wandb_tag(config_name)
    wandb_cfg = OmegaConf.select(cfg, "logger.wandb")
    if tag is None or wandb_cfg is None:
        return
    tags = [str(t) for t in (wandb_cfg.get("tags") or [])]
    if tag not in tags:
        tags.append(tag)
    with open_dict(cfg):
        cfg.logger.wandb.tags = tags


def run_dir(name, description, mode, ckpt_path, ts):
    """Pick the right Hydra output dir: ``latent_eval/<type>/<time>/...``
    for eval mode, ``finetuned/...`` for everything else."""
    if str(mode) == "eval":
        mt = model_type_from_ckpt(ckpt_path)
        mtime = model_time_from_ckpt(ckpt_path)
        return f"./logs/{name}/latent_eval/{mt}/{mtime}/{description}_{ts}"
    return f"./logs/{name}/finetuned/{description}_{ts}"


MECKA_PRIVATE_DIR_ENV = "EGOVERSE_MECKA_PRIVATE_DIR"


@functools.cache
def _mecka_private_splits(name):
    root = os.environ.get(MECKA_PRIVATE_DIR_ENV)
    if not root:
        raise RuntimeError(
            f"{MECKA_PRIVATE_DIR_ENV} is not set; config {name} needs it"
        )
    return json.loads((Path(root) / "splits" / f"{name}.json").read_text())


def mecka_private(name, key):
    """Value ``key`` of ``$EGOVERSE_MECKA_PRIVATE_DIR/splits/<name>.json``.

    The Mecka private set's episode and operator IDs may not be published, so its
    configs name them by key and the IDs live next to the data."""
    value = _mecka_private_splits(name)[key]
    # OmegaConf.resolve (hydra.utils.instantiate) rejects a plain list from a resolver
    return OmegaConf.create(list(value)) if isinstance(value, list) else value


for _name, _fn in (
    ("model_type_from_ckpt", model_type_from_ckpt),
    ("model_time_from_ckpt", model_time_from_ckpt),
    ("run_dir", run_dir),
    ("mecka_private", mecka_private),
):
    try:
        OmegaConf.register_new_resolver(_name, _fn)
    except (ValueError, AssertionError):
        pass
