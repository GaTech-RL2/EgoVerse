"""The RDT memory-ablation arms: each model tree resolves on its own (``ModelWrapper``
builds the model from the model config alone, so ``${data.*}`` there fails only
at train time), and the model's history / memory sizes match the keymap's."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from egomimic.trainHydra import _build_model_config_tree

ARM_NAMES = {
    "20pct_6d": ("base", "proprio", "vision", "both"),
    "all_kp": ("base", "proprio", "proprio_frames", "short_long_encoder"),
}
ARMS = [
    f"train_zarr_mecka_{variant}_rdt_mem_{arm}"
    for variant in ("20pct_6d", "all_kp")
    for arm in ARM_NAMES[variant]
]


def _compose(arm, compose_resolve):
    cfg = compose_resolve(arm, [], keep_hydra=True)
    return OmegaConf.masked_copy(cfg, [k for k in cfg if k != "hydra"])


@pytest.mark.parametrize("arm", ARMS)
def test_model_tree_resolves_without_the_rest_of_the_config(arm, compose_resolve):
    tree = _build_model_config_tree(_compose(arm, compose_resolve))
    tree._set_flag("allow_objects", True)
    OmegaConf.resolve(tree)


@pytest.mark.parametrize("arm", ARMS)
def test_model_sizes_match_the_keymap(arm, compose_resolve):
    cfg = _compose(arm, compose_resolve)
    OmegaConf.resolve(cfg)
    km = cfg.data.train_datasets.human_bimanual.resolver.key_map
    model = cfg.model.robomimic_model
    denoiser = model.head_specs.human_bimanual.model
    history = km.get("proprio_history", 1)
    (state_stem,) = model.stem_specs.human_bimanual.values()
    assert state_stem.history_len == history
    assert denoiser.n_state_tokens == history
    frames = model.trunk.get("image_history", 1)
    if km.get("image_history_gap_s") is None:
        assert frames == 1
    else:
        assert frames == km.get("image_history", 2)
    memory = model.trunk.get("memory")
    if memory is None:
        assert not km.get("image_memory") and not denoiser.get("n_memory_tokens")
        return
    assert memory.frames == km.image_memory
    assert memory.stride_s == km.image_memory_stride_s
    assert denoiser.n_memory_tokens == memory.frames
    assert memory.encoder.pool == "cls"
    if memory.fuse_proprio:
        assert history >= memory.frames
        assert km.history_stride_s == memory.stride_s
