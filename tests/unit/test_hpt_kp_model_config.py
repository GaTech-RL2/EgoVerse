"""The keypoint HPT model's stem knobs compose and agree with their data config."""

from __future__ import annotations

RECIPE = "train_zarr_mecka_kp_wrist_hpt"


def test_hpt_kp_model_carries_the_history_and_backbone_lr_knobs(compose_resolve):
    cfg = compose_resolve("train_zarr_cartesian", ["model=hpt_bc_keypoints_wrist_300M"])
    stem = cfg.model.robomimic_model.stem_specs.human_bimanual.state_keypoints
    assert stem.history_len == 1
    assert stem.history_dropout == 0.2
    assert cfg.model.backbone_lr_scale == 1.0


# The two knobs are one setting in two files: the keymap decides how many
# proprio steps a sample carries, the stem how many tokens consume them, and
# HPTModel.stem_process raises when they disagree. Composition alone cannot see
# it, so the keypoint recipe this model pairs with is checked here.
def test_keymap_proprio_history_matches_the_stem_that_consumes_it(compose_resolve):
    cfg = compose_resolve(RECIPE, [])
    key_map = cfg.data.train_datasets.human_bimanual.resolver.key_map
    stem = cfg.model.robomimic_model.stem_specs.human_bimanual.state_keypoints
    assert key_map.get("proprio_history", 1) == stem.history_len
