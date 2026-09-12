"""GPU tier: the real recipes (real configs, real episodes via the S3 resolver
+ SQL filters, real Pi weights) take 5 optimizer steps and write a checkpoint.

Run on a GPU node:
  srun -p rl2-lab -A rl2-lab --gres=gpu:a40:1 -c 12 --mem=64G -t 60 \\
    python -m pytest --integration -m gpu tests/integration/test_train_step_gpu.py -q
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch
from fixtures.recipes import RECIPES
from fixtures.train_harness import (
    BatchKeySpy,
    compose_recipe,
    hermetic_env,
    pi_unavailable,
)
from omegaconf import open_dict

import egomimic
import egomimic.trainHydra as train_hydra

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

VENDOR_NAMES = ["eva", "aria", "mecka", "scale"]
# Known-good episodes: the ones the shipped per-vendor HPT configs pin
# (data/eva.yaml, data/aria.yaml). Deterministic, and a regression here means
# the recipe broke, not that the S3 resolver handed us a different episode.
# Two shipped pins are unusable (checked against app.episodes 2026-09-12):
#   data/scale.yaml pins 2026-03-16-01-22-26-448000, recorded as eva_bimanual
#   (stale after the 2026-05 scale reprocess); data/mecka.yaml pins
#   69199812208123403bbdb24f, which is not in the table at all.
# Those vendors fall back to lab + embodiment with resolver.debug=2. The S3
# resolver's order is not deterministic, and one rl2 eva episode
# (2026-03-28-19-49-52-786000) has an all-NaN gripper command, so pin
# whenever a pin exists.
PINNED_EPISODES = {
    "eva": ["2025-12-26-18-07-46-296000"],
    "aria": ["2025-09-20-17-47-54-000000"],
}
# `lab` column of app.episodes for the fallback vendors (queried 2026-09-12).
VENDOR_SQL_LAB = {"mecka": "mecka", "scale": "scale"}
STEPS = 5


def _pi_ckpt() -> Path:
    default = Path(egomimic.__file__).parent / "algo/pi_checkpoints/pi05_base_pytorch"
    return Path(os.environ.get("EGOVERSE_PI05_CKPT", default))


def _gpu_overrides(emb: str, out: Path, vendor: str) -> list[str]:
    debug = (
        []
        if vendor in PINNED_EPISODES
        else [f"+data.train_datasets.{emb}.resolver.debug=2"]
    )
    return debug + [
        f"paths.output_dir={out}",
        f"data.train_dataloader_params.{emb}.batch_size=4",
        f"data.train_dataloader_params.{emb}.num_workers=4",
        f"data.valid_dataloader_params.{emb}.batch_size=4",
        f"data.valid_dataloader_params.{emb}.num_workers=4",
        "trainer=default",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "trainer.precision=bf16",
        "trainer.max_epochs=1",
        "trainer.min_epochs=1",
        f"trainer.limit_train_batches={STEPS}",
        "trainer.limit_val_batches=0",
        "+trainer.num_sanity_val_steps=0",
        "+trainer.enable_progress_bar=false",
        "callbacks=checkpoints",
        "callbacks.model_checkpoint.every_n_epochs=1",
        "~logger",
        "~evaluator",
        "+mmap_checkpoint=false",
        "norm_stats.save_cache_dir=null",
        "norm_stats.num_workers=4",
    ]


def _vendor_filter(cfg, emb: str, vendor: str) -> None:
    """Real configs pick episodes by SQL; pin this vendor's known-good episodes
    (or its lab, limited by debug=2) so the run is deterministic."""
    if vendor in PINNED_EPISODES:
        hashes = ", ".join(f"'{h}'" for h in PINNED_EPISODES[vendor])
        lam = f"lambda row: row['episode_hash'] in {{{hashes}}}"
    else:
        lam = (
            f"lambda row: (row['lab'] == '{VENDOR_SQL_LAB[vendor]}') "
            f"& (row['embodiment'] == '{emb}') & (row['is_deleted'] == False)"
        )
    filters = {
        "_target_": "egomimic.rldb.filters.DatasetFilter",
        "filter_lambdas": [lam],
    }
    with open_dict(cfg):
        cfg.data.train_datasets[emb].filters = filters
        # The vendor configs' valid split is an interpolation to the train node
        # (so this write is redundant there); cotrain_pi_base copies fields, so
        # set it explicitly for the Pi human rows.
        if cfg.data.valid_datasets.get(emb) is not None:
            cfg.data.valid_datasets[emb].filters = filters


def _run(cfg, out: Path) -> float:
    metrics, objects = train_hydra.train(cfg)
    loss = float(metrics["Train/action_loss"])
    assert math.isfinite(loss), loss
    assert objects["trainer"].global_step == STEPS
    ckpt = out / "checkpoints" / "last.ckpt"
    ckpt_dir = out / "checkpoints"
    assert ckpt.exists(), (
        sorted(ckpt_dir.glob("*")) if ckpt_dir.exists() else "no checkpoints dir"
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    assert state, "empty state_dict"
    return loss


@pytest.mark.parametrize("vendor", VENDOR_NAMES)
def test_hpt_real_recipe(tmp_path, monkeypatch, vendor):
    from egomimic.algo.hpt import HPT

    hermetic_env(monkeypatch)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)  # ResNet weights may download
    recipe = RECIPES[(vendor, "hpt")]
    out = tmp_path / "out"
    out.mkdir()
    cfg = compose_recipe(recipe, _gpu_overrides(recipe.embodiment, out, vendor), out)
    _vendor_filter(cfg, recipe.embodiment, vendor)
    spy = BatchKeySpy(monkeypatch, HPT)
    _run(cfg, out)
    assert "actions_cartesian" in spy.keys[recipe.embodiment]


@pytest.mark.parametrize("vendor", VENDOR_NAMES)
def test_pi_real_recipe(tmp_path, monkeypatch, vendor):
    reason = pi_unavailable()
    if reason:
        pytest.skip(reason)
    from egomimic.algo.pi import PI

    ckpt = _pi_ckpt()
    if not (ckpt / "model.safetensors").is_file():
        pytest.skip(f"no Pi checkpoint at {ckpt} (set EGOVERSE_PI05_CKPT)")
    hermetic_env(monkeypatch)
    recipe = RECIPES[(vendor, "pi")]
    out = tmp_path / "out"
    out.mkdir()
    cfg = compose_recipe(
        recipe,
        _gpu_overrides(recipe.embodiment, out, vendor)
        + [f"model.robomimic_model.config.pytorch_weight_path={ckpt}"],
        out,
    )
    _vendor_filter(cfg, recipe.embodiment, vendor)
    spy = BatchKeySpy(monkeypatch, PI)
    _run(cfg, out)
    assert "base_0_rgb" in spy.keys[recipe.embodiment]
