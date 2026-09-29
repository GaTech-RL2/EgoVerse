"""The flagship keypoint HPT arm: the keymap's proprio history and the stem
that consumes it agree, and the recipe pairs the keypoint evaluator and prompt."""

from __future__ import annotations

import hydra

RECIPE = "train_zarr_mecka_flagship_kp_hpt"


# The two knobs are one setting in two files: the keymap decides how many
# proprio steps a sample carries, the stem how many tokens consume them, and
# HPTModel.stem_process raises when they disagree. Composition alone cannot see
# it; the flagship 6D parent ships K = 3, which the recipe must reset.
def test_keymap_proprio_history_matches_the_stem_that_consumes_it(compose_resolve):
    cfg = compose_resolve(RECIPE, ["data=mecka_fold_flagship_top3_hpt_6d"])
    key_map = cfg.data.train_datasets.human_bimanual.resolver.key_map
    stem = cfg.model.robomimic_model.stem_specs.human_bimanual.state_keypoints
    assert key_map.get("proprio_history", 1) == stem.history_len
    # built, not just read: a misspelled kwarg in the recipe's `data:` block
    assert hydra.utils.instantiate(key_map)
    res = cfg.data.train_datasets.human_bimanual.resolver
    assert hydra.utils.instantiate(res.transform_list)


def test_kp_recipe_pairs_the_keypoint_evaluator_and_prompt(compose_resolve):
    cfg = compose_resolve(RECIPE, [])
    # every flagship prompt is the default one; a comparison arm must match it
    assert cfg.model.robomimic_model.default_prompt == "fold clothes"
    ev = cfg.evaluator
    assert (
        "keypoints_revert_6d_wristframe" in ev.transform_lists.human_bimanual._target_
    )
    assert ev.viz_func.human_bimanual.mode == "keypoints"
