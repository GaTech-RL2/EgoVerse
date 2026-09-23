"""The RDT memory-ablation arms: each model tree resolves on its own (``ModelWrapper``
builds the model from the model config alone, so ``${data.*}`` there fails only
at train time), and the model's history / memory sizes match the keymap's."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from egomimic.trainHydra import _build_model_config_tree

ARMS = ("base", "proprio", "vision", "both")


def _compose(arm, compose_resolve):
    cfg = compose_resolve(
        f"train_zarr_mecka_20pct_6d_rdt_mem_{arm}", [], keep_hydra=True
    )
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
    assert model.stem_specs.human_bimanual.state_ee_pose.history_len == history
    assert denoiser.n_state_tokens == history
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
