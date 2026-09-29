"""The ABC sim fine-tune recipe end to end on CPU: two optimizer steps of a
tiny RDT on synthetic eva JOINT episodes through the real trainHydra.train()
path, then the saved checkpoint served by policy_server (load, rebuild the
training inputs from the checkpoint's own config, infer)."""

from __future__ import annotations

import math

from fixtures.recipes import LOCAL_RESOLVER, common_overrides, cpu_trainer_overrides
from fixtures.train_harness import BatchKeySpy, compose_recipe, hermetic_env, write_fixtures
from fixtures.recipes import Recipe

import numpy as np

import egomimic.trainHydra as train_hydra
from egomimic.algo.hpt import HPT
from egomimic.rldb.embodiment.eva import JOINT_ACTION_KEY, JOINT_STATE_KEY, Eva
from egomimic.scripts.abc_sim.policy_server import JointPolicy

EMB = "eva_bimanual"
RM = "model.robomimic_model"


def _tiny_rdt(emb: str) -> list[str]:
    enc = f"{RM}.encoder_specs.front_img_1"
    return [
        f"{RM}.width=32",
        f"{RM}.trunk.depth=2",
        f"{RM}.trunk.num_heads=4",
        f"{RM}.head_specs.{emb}.num_inference_steps=2",
        f"~{RM}.shared_stem_specs.annotation",
        f"{RM}.shared_obs_keys=[front_img_1]",
        f"{RM}.annotation_key=null",
        f"{enc}.model_name=vit_small_patch16_dinov3",
        f"{enc}.image_size=[32,48]",
        f"+{enc}.pretrained=false",
        f"+{enc}.tower_kwargs={{embed_dim: 32, depth: 1, num_heads: 2}}",
    ]


def test_abc_sim_finetune_train_step(tmp_path, monkeypatch):
    hermetic_env(monkeypatch)
    data, out, hashes = write_fixtures(tmp_path, "eva")
    recipe = Recipe("train_zarr_abc_sim_rdt_ft", "abc_sim_joints", "rdt_1b_abc_joints_dinov3_vitb", EMB)
    cfg = compose_recipe(
        recipe,
        common_overrides(EMB, data, out, batch_size=2, num_workers=0, episode_hashes=hashes)
        + cpu_trainer_overrides(2)
        + _tiny_rdt(EMB)
        + [
            # the abc_sim_joints resolver is already local; point it at the fixtures
            f"data.train_datasets.{EMB}.resolver.folder_path={data}",
            f"data.valid_datasets.{EMB}.filters=null",
        ],
        out,
    )
    assert cfg.data.train_datasets[EMB].resolver._target_ == LOCAL_RESOLVER
    spy = BatchKeySpy(monkeypatch, HPT)
    metrics, objects = train_hydra.train(cfg)
    loss = float(metrics["Train/action_loss"])
    assert math.isfinite(loss)
    assert objects["trainer"].global_step == 2
    seen = spy.keys[EMB]
    expected = {JOINT_ACTION_KEY, JOINT_STATE_KEY, Eva.VIZ_IMAGE_KEY, f"{Eva.VIZ_IMAGE_KEY}_hist", "fps"}
    assert expected <= seen, f"missing {expected - seen}; saw {sorted(seen)}"
    assert "observations.images.left_wrist_img" not in seen  # joints_front keymap

    # The checkpoint alone rebuilds the training pipeline and serves it.
    ckpt = out / "served.ckpt"
    objects["trainer"].save_checkpoint(ckpt)
    policy = JointPolicy(str(ckpt), device="cpu")
    assert policy.cameras == ["top"] and policy.lag_frames == 3
    img = np.random.default_rng(0).random((3, 64, 64), dtype=np.float32)
    actions = policy.infer(np.zeros(14, np.float32), {"top": img, "top_hist": img}, "pick up the red cube")
    assert actions.shape == (45, 14) and np.isfinite(actions).all()
