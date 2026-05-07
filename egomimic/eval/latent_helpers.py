"""Helpers + OmegaConf resolvers for the latent-feature evaluator.

Importing this module registers three OmegaConf resolvers
(``model_type_from_ckpt``, ``model_time_from_ckpt``, ``run_dir``) that the
``train_zarr_cartesian`` Hydra config uses to route eval-mode runs into
``logs/<name>/latent_eval/<type>/<time>/...`` instead of the default
``finetuned/...`` layout. Pure module-side-effect; safe to import multiple
times.
"""

from __future__ import annotations

import re

from omegaconf import OmegaConf


def model_dir_name_from_ckpt(ckpt_path):
    """Extract the model-folder name (e.g. 'cotrain_partial_aug_2026-04-24_03-08-44')
    from a ckpt path like 'logs/pick_place/finetuned/<NAME>/0/checkpoints/...'."""
    if not ckpt_path or ckpt_path == "null":
        return None
    parts = str(ckpt_path).replace("\\", "/").split("/")
    if "finetuned" in parts:
        i = parts.index("finetuned")
        if i + 1 < len(parts):
            return parts[i + 1]
    # Fallback: walk up from the ckpt file (logs/.../<NAME>/0/checkpoints/file.ckpt)
    if len(parts) >= 4:
        return parts[-4]
    return None


def model_type_from_ckpt(ckpt_path):
    """Strip trailing '_YYYY-MM-DD_HH-MM-SS' from the model dir name. Returns
    'pretrained' when no checkpoint is supplied."""
    name = model_dir_name_from_ckpt(ckpt_path)
    if name is None:
        return "pretrained"
    m = re.match(r"^(.+?)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$", name)
    return m.group(1) if m else name


def model_time_from_ckpt(ckpt_path):
    """Extract the trailing 'YYYY-MM-DD_HH-MM-SS' from the model dir name.
    Returns 'base' when no checkpoint is supplied."""
    name = model_dir_name_from_ckpt(ckpt_path)
    if name is None:
        return "base"
    m = re.match(r"^.+?_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$", name)
    return m.group(1) if m else "unknown"


def run_dir(name, description, mode, ckpt_path, ts):
    """Pick the right hydra run dir: latent_eval/<type>/<time>/... for eval,
    finetuned/... for everything else."""
    if str(mode) == "eval":
        mt = model_type_from_ckpt(ckpt_path)
        mtime = model_time_from_ckpt(ckpt_path)
        return f"./logs/{name}/latent_eval/{mt}/{mtime}/{description}_{ts}"
    return f"./logs/{name}/finetuned/{description}_{ts}"


# Register resolvers idempotently: OmegaConf raises on re-register, so guard.
for _name, _fn in (
    ("model_type_from_ckpt", model_type_from_ckpt),
    ("model_time_from_ckpt", model_time_from_ckpt),
    ("run_dir", run_dir),
):
    try:
        OmegaConf.register_new_resolver(_name, _fn)
    except (ValueError, AssertionError):
        # Already registered (e.g. module imported twice in tests).
        pass
