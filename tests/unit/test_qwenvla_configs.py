"""QwenVLA configs compose, wire the algo, and train two stub steps."""

from __future__ import annotations

import pytest
from fixtures.fake_qwen35 import STUB_HIDDEN, STUB_IMAGE_SIZE, install_fake_qwen35
from hydra import compose, initialize_config_module

MODEL = "qwenvla_bc_mecka_6d_0p8b"


def _compose(config_name: str, overrides: list[str]):
    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        return compose(config_name=config_name, overrides=overrides)


def test_0p8b_config_wires_the_algo():
    rm = _compose("train_zarr_cartesian", [f"model={MODEL}"]).model.robomimic_model
    assert rm._target_ == "egomimic.algo.qwenvla.QwenVLA"
    assert rm.backbone._target_ == "egomimic.models.qwenvla_nets.QwenVLBackbone"
    assert rm.backbone.model_name == "Qwen/Qwen3.5-0.8B"
    assert rm.backbone.freeze is False and rm.backbone.dtype == "float32"
    assert rm.backbone.image_size == [352, 640]
    assert rm.head._target_ == "egomimic.models.layerwise_dit.LayerwiseFMHead"
    assert rm.head._partial_ is True
    assert rm.head.action_horizon == 100
    assert rm.head.dit_hidden is None
    assert rm.head.interleave_self_attention is True
    assert rm.head.repeated_diffusion_steps == 1
    # the HPT baseline head samples with 50 steps; the metric compares sampled actions
    assert rm.head.num_inference_steps == 50
    assert rm.dims.human_bimanual == {"proprio": 20, "action": 18}
    assert rm.action_width == 18
    assert rm.embodiment_label is True
    assert rm.history_len == 1


def test_2b_config_compresses_the_head():
    rm = _compose(
        "train_zarr_cartesian", ["model=qwenvla_bc_mecka_6d_2b"]
    ).model.robomimic_model
    assert rm.backbone.model_name == "Qwen/Qwen3.5-2B"
    assert rm.head.dit_hidden == 1024
    assert rm.backbone.freeze is True and rm.backbone.trainable_layers == 8


def test_cotrain_config_shares_one_padded_action_width():
    rm = _compose(
        "train_zarr_cartesian", ["model=qwenvla_cotrain_eva_mecka_6d"]
    ).model.robomimic_model
    assert list(rm.domains) == ["eva_bimanual", "human_bimanual"]
    assert rm.action_width == 20
    assert rm.dims.eva_bimanual.action == 20 and rm.dims.human_bimanual.action == 18


def test_flagship_recipe_selects_qwenvla_and_the_hpt_evaluator():
    cfg = _compose("train_zarr_mecka_flagship_6d_qwenvla", [])
    assert cfg.model.robomimic_model._target_ == "egomimic.algo.qwenvla.QwenVLA"
    assert cfg.evaluator._target_ == "egomimic.eval.eval_hpt.HPTEvalVideo"
    assert cfg.model.backbone_lr_scale == 0.1
    assert cfg.model.optimizer.lr == 1e-4


KP_MODEL = "qwenvla_bc_mecka_kp_0p8b"


def test_kp_config_is_the_0p8b_on_the_144d_keypoint_action():
    rm = _compose("train_zarr_cartesian", [f"model={KP_MODEL}"]).model.robomimic_model
    assert rm.dims.human_bimanual == {"proprio": 144, "action": 144}
    assert rm.ac_keys.human_bimanual == "actions_keypoints"
    assert rm.action_width == 144
    assert rm["6dof"] is False
    # everything else is inherited from the 6D config
    assert rm.backbone.model_name == "Qwen/Qwen3.5-0.8B"
    assert rm.head.num_inference_steps == 50


@pytest.mark.parametrize(
    "recipe, target",
    [
        ("train_zarr_mecka_flagship_kp_qwenvla", "egomimic.algo.qwenvla.QwenVLA"),
        ("train_zarr_mecka_flagship_kp_hpt", "egomimic.algo.hpt.HPT"),
    ],
)
def test_kp_recipes_pair_the_keypoint_data_evaluator_and_prompt(recipe, target):
    cfg = _compose(recipe, [])
    rm = cfg.model.robomimic_model
    assert rm._target_ == target
    assert rm.ac_keys.human_bimanual == "actions_keypoints"
    assert rm.dims.human_bimanual.action == 144
    # both arms of the comparison see the same prompt (annotations are empty)
    assert rm.default_prompt == "fold clothes"
    assert cfg.evaluator._target_ == "egomimic.eval.eval_hpt.HPTEvalVideo"
    assert (
        "keypoints_revert_6d_wristframe"
        in cfg.evaluator.transform_lists.human_bimanual._target_
    )
    resolver = cfg.data.train_datasets.human_bimanual.resolver
    assert resolver.key_map.keymap_mode == "keypoints"
    assert resolver.transform_list.mode == "keypoints_wristframe_6d"
    assert resolver.transform_list.pad_proprio_gripper is False


def test_kp_top3_data_keeps_the_three_operators_and_the_keypoint_pipeline():
    cfg = _compose(
        "train_zarr_mecka_flagship_kp_hpt",
        ["data=mecka_fold_flagship_top3_hpt_keypoints"],
    )
    lambdas = cfg.data.train_datasets.human_bimanual.filters.filter_lambdas
    assert any("6903686e0e94ce070afd1f24" in fn for fn in lambdas)
    assert (
        cfg.data.train_datasets.human_bimanual.resolver.transform_list.mode
        == "keypoints_wristframe_6d"
    )


def test_hpt_kp_model_carries_the_history_and_backbone_lr_knobs():
    cfg = _compose("train_zarr_cartesian", ["model=hpt_bc_keypoints_wrist_300M"])
    stem = cfg.model.robomimic_model.stem_specs.human_bimanual.state_keypoints
    assert stem.history_len == 1 and stem.history_dropout == 0.2
    assert cfg.model.backbone_lr_scale == 1.0


def test_qwenvla_train_step_on_the_stub(tmp_path, monkeypatch):
    from fixtures.recipes import Recipe, common_overrides, cpu_trainer_overrides
    from fixtures.train_harness import compose_recipe, hermetic_env, write_fixtures

    import egomimic.trainHydra as train_hydra

    snapshot = install_fake_qwen35(tmp_path, monkeypatch)
    hermetic_env(monkeypatch)
    recipe = Recipe("train_zarr_cartesian", "mecka_all_6d", MODEL, "human_bimanual")
    data, out, hashes = write_fixtures(tmp_path, "mecka")
    cfg = compose_recipe(
        recipe,
        common_overrides(
            recipe.embodiment,
            data,
            out,
            batch_size=2,
            num_workers=0,
            episode_hashes=hashes,
        )
        + cpu_trainer_overrides(2)
        + [
            f"model.robomimic_model.backbone.model_name={snapshot}",
            "model.robomimic_model.backbone.dtype=float32",
            f"model.robomimic_model.backbone.image_size=[{STUB_IMAGE_SIZE[0]},{STUB_IMAGE_SIZE[1]}]",
            "model.robomimic_model.head.head_dim=8",
            "model.robomimic_model.head.num_register_tokens=2",
            "model.robomimic_model.head.num_inference_steps=2",
        ],
        out,
    )
    metrics, objects = train_hydra.train(cfg)
    loss = float(metrics["Train/action_loss"])
    assert loss == loss, "non-finite loss"
    assert objects["trainer"].global_step == 2
    # objects["model"] is the ModelWrapper (trainHydra.py object_dict)
    policy = objects["model"].model.nets["policy"]
    assert policy.head.dit_hidden == STUB_HIDDEN
    assert len(policy.head.projectors) == policy.backbone.num_layers


def test_qwenvla_kp_train_step_on_the_stub(tmp_path, monkeypatch):
    from fixtures.recipes import Recipe, common_overrides, cpu_trainer_overrides
    from fixtures.train_harness import compose_recipe, hermetic_env, write_fixtures

    import egomimic.trainHydra as train_hydra

    snapshot = install_fake_qwen35(tmp_path, monkeypatch)
    hermetic_env(monkeypatch)
    recipe = Recipe("train_zarr_cartesian", "mecka", KP_MODEL, "human_bimanual")
    data, out, hashes = write_fixtures(tmp_path, "mecka")
    emb = "data.train_datasets.human_bimanual.resolver"
    cfg = compose_recipe(
        recipe,
        common_overrides(
            recipe.embodiment,
            data,
            out,
            batch_size=2,
            num_workers=0,
            episode_hashes=hashes,
        )
        + cpu_trainer_overrides(2)
        + [
            f"{emb}.key_map.include_ee_pose=false",
            f"{emb}.transform_list.include_ee_pose=false",
            f"{emb}.transform_list.pad_proprio_gripper=false",
            f"model.robomimic_model.backbone.model_name={snapshot}",
            "model.robomimic_model.backbone.dtype=float32",
            f"model.robomimic_model.backbone.image_size=[{STUB_IMAGE_SIZE[0]},{STUB_IMAGE_SIZE[1]}]",
            "model.robomimic_model.head.head_dim=8",
            "model.robomimic_model.head.num_register_tokens=2",
            "model.robomimic_model.head.num_inference_steps=2",
        ],
        out,
    )
    metrics, objects = train_hydra.train(cfg)
    loss = float(metrics["Train/action_loss"])
    assert loss == loss, "non-finite loss"
    assert objects["trainer"].global_step == 2
    policy = objects["model"].model.nets["policy"]
    assert policy.head.action_width == 144
    assert policy.head.state_encoder.state_dims == {"human_bimanual": 144}
