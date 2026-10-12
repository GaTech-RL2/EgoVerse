"""The flagship keypoint HPT arm: recipe, data split and stem knobs compose."""

from __future__ import annotations

import pytest

RECIPE = "train_zarr_mecka_flagship_kp_hpt"


def test_kp_recipe_pairs_the_keypoint_data_evaluator_and_prompt(compose_resolve):
    cfg = compose_resolve(RECIPE, [])
    rm = cfg.model.robomimic_model
    assert rm._target_ == "egomimic.algo.hpt.HPT"
    assert rm.ac_keys.human_bimanual == "actions_keypoints"
    assert rm.dims.human_bimanual.action == 144
    assert rm.dims.human_bimanual.proprio == 144
    # The flagship split carries no per-frame annotations, so every sample is
    # meant to see one constant prompt; a comparison arm must set the same one.
    assert rm.default_prompt == "fold clothes"
    assert cfg.evaluator._target_ == "egomimic.eval.eval_hpt.HPTEvalVideo"
    assert (
        "keypoints_revert_6d_wristframe"
        in cfg.evaluator.transform_lists.human_bimanual._target_
    )
    # The train-time overlay wraps its own evaluator under `base`, and renders
    # the keypoint action through Human.viz(mode="keypoints").
    train_viz = cfg.train_viz_evaluator.base
    assert train_viz._target_ == "egomimic.eval.eval_hpt.HPTEvalVideo"
    assert train_viz.viz_func.human_bimanual.mode == "keypoints"
    assert train_viz.viz_func.human_bimanual.action_key == "actions_keypoints"
    assert (
        train_viz.transform_lists.human_bimanual._target_
        == cfg.evaluator.transform_lists.human_bimanual._target_
    )
    resolver = cfg.data.train_datasets.human_bimanual.resolver
    assert resolver.key_map.keymap_mode == "keypoints"
    assert resolver.transform_list.mode == "keypoints_wristframe_6d"
    assert resolver.transform_list.pad_proprio_gripper is False


def test_kp_top3_data_keeps_the_three_operators_and_the_keypoint_pipeline(
    compose_resolve,
):
    cfg = compose_resolve(RECIPE, ["data=mecka_fold_flagship_top3_hpt_keypoints"])
    lambdas = cfg.data.train_datasets.human_bimanual.filters.filter_lambdas
    assert any("6903686e0e94ce070afd1f24" in fn for fn in lambdas)
    assert (
        cfg.data.train_datasets.human_bimanual.resolver.transform_list.mode
        == "keypoints_wristframe_6d"
    )


def test_hpt_kp_model_carries_the_history_and_backbone_lr_knobs(compose_resolve):
    cfg = compose_resolve("train_zarr_cartesian", ["model=hpt_bc_keypoints_wrist_300M"])
    stem = cfg.model.robomimic_model.stem_specs.human_bimanual.state_keypoints
    assert stem.history_len == 1
    assert stem.history_dropout == 0.2
    assert cfg.model.backbone_lr_scale == 1.0


# The two knobs are one setting in two files: the keymap decides how many
# proprio steps a sample carries, the stem how many tokens consume them, and
# HPTModel.stem_process raises when they disagree. Composition alone cannot see
# it, so every keypoint data config this model is documented to pair with is
# checked here.
@pytest.mark.parametrize(
    "data_cfg",
    [
        "mecka_fold_flagship_opsplit_hpt_keypoints",
        "mecka_fold_flagship_top3_hpt_keypoints",
    ],
)
def test_keymap_proprio_history_matches_the_stem_that_consumes_it(
    data_cfg, compose_resolve
):
    cfg = compose_resolve(RECIPE, [f"data={data_cfg}"])
    key_map = cfg.data.train_datasets.human_bimanual.resolver.key_map
    stem = cfg.model.robomimic_model.stem_specs.human_bimanual.state_keypoints
    assert key_map.get("proprio_history", 1) == stem.history_len
