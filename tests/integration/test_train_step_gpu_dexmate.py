"""GPU tier for dexmate_bimanual: the real SharpaDex HPT keypoints recipe takes
a few optimizer steps on converted SharpaDex episodes, writes a checkpoint, and
the checkpoint alone rebuilds the trained model.

The recipe is RECIPES[("dexmate", "hpt")] (train_zarr_dexmate_keypoints,
data/dexmate_keypoints, model/hpt_bc_keypoints_dexmate) at full size. Episodes
come from a local directory of sharpa_to_zarr outputs, read through
LocalEpisodeResolver; the first EGOVERSE_TEST_DEXMATE_EPISODES (default 3)
episode hashes, sorted, are pinned. The ResNet encoders start from random
weights (weights=null), so the test needs no network.

Run on a GPU node:
  EGOVERSE_TEST_DEXMATE_DIR=/path/to/converted \\
  srun -p rl2-lab -A rl2-lab --gres=gpu:a40:1 -c 12 --mem=30G -t 60 \\
    python -m pytest --integration -m gpu \\
      tests/integration/test_train_step_gpu_dexmate.py -q
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch
from fixtures.recipes import HPT_CAMERAS, LOCAL_RESOLVER, RECIPES
from fixtures.train_harness import BatchKeySpy, compose_recipe, hermetic_env

import egomimic.trainHydra as train_hydra
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.embodiment.dexmate import Dexmate
from egomimic.rldb.embodiment.embodiment import get_embodiment_id

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

STEPS = 5
RECIPE = RECIPES[("dexmate", "hpt")]
EMB = RECIPE.embodiment


def _episodes() -> tuple[Path, list[str]]:
    root = os.environ.get("EGOVERSE_TEST_DEXMATE_DIR")
    if not root:
        pytest.skip("set EGOVERSE_TEST_DEXMATE_DIR to a dir of converted episodes")
    data = Path(root)
    hashes = sorted(p.name[: -len(".zarr")] for p in data.glob("*.zarr"))
    n = int(os.environ.get("EGOVERSE_TEST_DEXMATE_EPISODES", "3"))
    if len(hashes) < n:
        pytest.skip(f"{data} holds {len(hashes)} episodes, need {n}")
    return data, hashes[:n]


def _overrides(data: Path, out: Path, hashes: list[str]) -> list[str]:
    pins = ", ".join(hashes)
    return [
        f"paths.dataset_dir={data}",
        f"paths.output_dir={out}",
        f"paths.cache_dir={out}/cache",  # never reuse stats from an earlier run
        f"data.train_datasets.{EMB}.resolver._target_={LOCAL_RESOLVER}",
        # Reset first: the config's filter_lambdas would merge into the pins.
        f"data.train_datasets.{EMB}.filters=null",
        f"++data.train_datasets.{EMB}.filters={{_target_: "
        f"egomimic.rldb.filters.DatasetFilter, episode_hashes: [{pins}]}}",
        f"data.train_dataloader_params.{EMB}.batch_size=4",
        f"data.train_dataloader_params.{EMB}.num_workers=4",
        f"data.valid_dataloader_params.{EMB}.batch_size=4",
        f"data.valid_dataloader_params.{EMB}.num_workers=4",
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
        "+mmap_checkpoint=false",
        "norm_stats.save_cache_dir=null",
        "norm_stats.num_workers=4",
        *[
            f"+model.robomimic_model.encoder_specs.{c}.weights=null"
            for c in HPT_CAMERAS[EMB]
        ],
    ]


def test_hpt_dexmate_real_recipe(tmp_path, monkeypatch):
    from egomimic.algo.hpt import HPT

    hermetic_env(monkeypatch)
    data, hashes = _episodes()
    out = tmp_path / "out"
    out.mkdir()
    cfg = compose_recipe(RECIPE, _overrides(data, out, hashes), out)
    spy = BatchKeySpy(monkeypatch, HPT)

    metrics, objects = train_hydra.train(cfg)
    loss = float(metrics["Train/action_loss"])
    print(f"dexmate HPT: Train/action_loss={loss:.6f} after {STEPS} steps on {hashes}")
    assert math.isfinite(loss), loss
    assert objects["trainer"].global_step == STEPS
    seen = spy.keys[EMB]
    expected = {
        "actions_keypoints",
        "observations.state.keypoints",
        Dexmate.VIZ_IMAGE_KEY,
        *Dexmate.WRIST_IMAGE_KEYS.values(),
    }
    assert expected <= seen, f"missing {expected - seen}; saw {sorted(seen)}"

    ckpt = out / "checkpoints" / "last.ckpt"
    assert ckpt.exists(), sorted((out / "checkpoints").glob("*"))
    live = {k: v.detach().cpu() for k, v in objects["model"].state_dict().items()}
    for v in live.values():
        if v.is_floating_point():
            assert torch.isfinite(v).all()

    # The checkpoint alone rebuilds the model: its hyperparameters hold the
    # config tree and the norm stats, keyed by embodiment id 8.
    emb_id = get_embodiment_id(EMB)
    reloaded = ModelWrapper.load_from_checkpoint(
        ckpt, map_location="cpu", weights_only=False
    )
    assert reloaded.hparams.norm_stats_state["norm_stats"].keys() == {emb_id}
    assert reloaded.model.ac_keys[emb_id] == "actions_keypoints"
    state = reloaded.state_dict()
    assert state.keys() == live.keys(), sorted(state.keys() ^ live.keys())[:10]
    differ = [k for k, v in live.items() if not torch.equal(state[k], v)]
    assert not differ, differ[:10]
