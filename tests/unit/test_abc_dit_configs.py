"""The eight ABC-DiT model configs and their two recipes compose and wire."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_module

from egomimic.models.abc_dit_nets import ABC_DIT_SIZES

SIZES = ["s", "b", "l", "xl"]


def _compose(config_name: str, overrides: list[str]):
    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        return compose(config_name=config_name, overrides=overrides)


def _model(name: str):
    return _compose("train_zarr_cartesian", [f"model={name}"]).model


def test_base_wires_the_algo_the_dit_and_both_encoders():
    model = _model("abc_dit_bc_mecka_kp_b")
    rm = model.robomimic_model
    assert rm._target_ == "egomimic.algo.abc_dit.ABCDiT"
    assert rm.policy._target_ == "egomimic.models.abc_dit_nets.ABCDiTPolicy"
    assert rm.policy._partial_ is True  # ABCDiT fills action_width / state_dims
    assert rm.policy.camera_roles == ["observations.images.front_img_1"]
    assert rm.policy.action_horizon == 100
    assert rm.policy.vision._target_ == "egomimic.models.abc_dit_nets.DINOv3Tower"
    assert rm.policy.vision.model_name == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert rm.policy.vision.freeze is False and rm.policy.vision.image_size == 224
    assert rm.policy.vision.bf16_autocast is True  # ABC's dino_bf16
    assert (
        rm.policy.task_encoder._target_
        == "egomimic.models.abc_dit_nets.CLIPTaskEncoder"
    )
    assert rm.policy.task_encoder.model_name == "openai/clip-vit-base-patch32"
    assert rm.policy.task_encoder.cache_text_features is True
    # ABC's pooling and flow knobs
    assert rm.policy.vision_pool_num_queries == 12
    assert rm.policy.mask_state_ratio == 0.1
    assert rm.policy.max_action_prefix == 0  # ABC's production default is 8
    assert rm.policy.time_dist == "uniform"
    # 50 to match the HPT / QwenVLA / FlowVLA arms' sampled-action metrics
    assert rm.policy.num_inference_steps == 50
    assert rm.embodiment_label is False and rm.history_len == 1


def test_optimizer_follows_abcs_training_details():
    model = _model("abc_dit_bc_mecka_kp_b")
    assert model.optimizer.lr == 1e-4
    assert list(model.optimizer.betas) == [0.9, 0.95]
    assert model.optimizer.weight_decay == 0.01
    assert model.scheduler is None
    # "a learning rate scale of 0.1 for the vision encoder" (paper, B.3)
    assert model.backbone_lr_scale == 0.1


def test_the_augs_do_not_normalise():
    """The DINOv3 tower applies the checkpoint's own mean / std, so a Normalize
    in the augs would double-normalise (the SigLIP stem has the same rule)."""
    for name in ("abc_dit_bc_mecka_kp_b", "abc_dit_bc_mecka_6d_b"):
        model = _model(name)
        targets = [
            t._target_ for t in model.robomimic_model.train_image_augs.transforms
        ]
        assert targets == ["torchvision.transforms.ColorJitter"]
        assert model.robomimic_model.eval_image_augs is None


@pytest.mark.parametrize("size", SIZES)
def test_every_size_selects_its_rung(size):
    for action in ("kp", "6d"):
        rm = _model(f"abc_dit_bc_mecka_{action}_{size}").robomimic_model
        assert rm.policy.size == size
        assert size in ABC_DIT_SIZES
        # the size is the ONLY difference: the shape overrides stay unset
        assert rm.policy.hidden_size is None
        assert rm.policy.depth is None and rm.policy.num_heads is None


@pytest.mark.parametrize("size", SIZES)
def test_keypoint_configs_are_the_144d_wrist_frame_action(size):
    rm = _model(f"abc_dit_bc_mecka_kp_{size}").robomimic_model
    assert rm.dims.human_bimanual == {"proprio": 144, "action": 144}
    assert rm.ac_keys.human_bimanual == "actions_keypoints"
    assert rm.action_width == 144
    assert rm["6dof"] is False
    # ABC-DiT has one state vector: name the keypoints, not data/mecka's ee_pose
    assert list(rm.proprio_keys) == ["observations.state.keypoints"]


@pytest.mark.parametrize("size", SIZES)
def test_cartesian_configs_are_the_18d_6d_action(size):
    rm = _model(f"abc_dit_bc_mecka_6d_{size}").robomimic_model
    assert rm.dims.human_bimanual == {"proprio": 20, "action": 18}
    assert rm.ac_keys.human_bimanual == "actions_cartesian"
    assert rm.action_width == 18
    assert rm["6dof"] is True
    assert list(rm.proprio_keys) == ["observations.state.ee_pose"]


def test_recipes_select_abc_dit_the_flagship_split_and_the_hpt_evaluators():
    kp = _compose("train_zarr_mecka_flagship_kp_abc_dit", [])
    assert kp.model.robomimic_model._target_ == "egomimic.algo.abc_dit.ABCDiT"
    assert kp.model.robomimic_model.policy.size == "b"
    assert kp.model.robomimic_model.default_prompt == "fold clothes"
    assert kp.evaluator._target_ == "egomimic.eval.eval_hpt.HPTEvalVideo"

    sixd = _compose("train_zarr_mecka_flagship_6d_abc_dit", [])
    assert sixd.model.robomimic_model.ac_keys.human_bimanual == "actions_cartesian"
    assert sixd.evaluator._target_ == "egomimic.eval.eval_hpt.HPTEvalVideo"
    assert sixd.model.robomimic_model.default_prompt == "fold clothes"


@pytest.mark.parametrize("size", SIZES)
def test_a_size_is_one_word_on_the_recipe(size):
    """Every rung is reachable from the recipe with a single model= override."""
    cfg = _compose(
        "train_zarr_mecka_flagship_kp_abc_dit", [f"model=abc_dit_bc_mecka_kp_{size}"]
    )
    assert cfg.model.robomimic_model.policy.size == size
    assert cfg.model.robomimic_model.default_prompt == "fold clothes"
