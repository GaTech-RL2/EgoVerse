from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from egomimic.pipeline import checkpoint_compat as compat


def _legacy_model_config():
    return {
        "_target_": "egomimic.pipeline.algo.PipelineAlgo",
        "action_horizon": 2560,
        "domains": ["eva_bimanual", "human_bimanual"],
        "ac_keys": {
            "eva_bimanual": "actions_cartesian",
            "human_bimanual": "actions_cartesian",
        },
        "stages": [
            {
                "_target_": "egomimic.pipeline.stages_io.NormalObsExpand",
                "n_obs_steps": 2,
            },
            {"_target_": "egomimic.pipeline.stages_io.ObsEncoders"},
            {
                "_target_": "egomimic.pipeline.stages_seq.Rename",
                "mapping": {"A": "a_top", "S": "s"},
            },
            {
                "_target_": "egomimic.pipeline.stages_seq.ObsStack",
                "in_keys": ["a_top", "s"],
                "out_keys": ["a_top", "s"],
                "n_obs_steps": 2,
            },
            {
                "_target_": "egomimic.pipeline.stages_io.NormalObsCollapse",
                "keys": ["a_top", "s"],
            },
            {
                "_target_": "egomimic.pipeline.stages_flow.DiffusionHead",
                "chunk_len": 100,
                "denoiser": "single",
                "moe_experts": 0,
            },
            {
                "_target_": "egomimic.pipeline.stages_flow.MaskedActionLoss",
                "name": "ddpm",
            },
        ],
    }


def _norm_state():
    return {
        "norm_mode": "min_max",
        "embodiments": [8, 18],
        "key_types": {8: {"actions_cartesian": "action_keys"}, 18: {}},
        "zarr_keys": {8: {"actions_cartesian": "actions"}, 18: {}},
        "shapes": {8: {"actions_cartesian": [20]}, 18: {}},
        "norm_stats": {
            8: {"actions_cartesian": {"min": torch.zeros(20)}},
            18: {},
        },
    }


def _checkpoint(model):
    return {
        "epoch": 3,
        "global_step": 130756,
        "pytorch-lightning_version": "2.6.1",
        "state_dict": {"nets.policy.example": torch.ones(2, 3)},
        "hyper_parameters": {
            "config_tree": OmegaConf.create({"model": {"robomimic_model": model}}),
            "norm_stats_state": _norm_state(),
        },
        "optimizer_states": [{"large": "training-only"}],
        "loops": {"training-only": True},
        "callbacks": {"training-only": True},
    }


def test_fold_migration_is_stripped_idempotent_and_remaps_ids(
    tmp_path: Path, monkeypatch
):
    model = _legacy_model_config()
    checkpoint = _checkpoint(model)
    monkeypatch.setattr(
        compat,
        "LEGACY_FOLD_MODEL_SHA256",
        compat._canonical_sha256(model),
    )
    source = tmp_path / "epoch3.ckpt"
    torch.save(checkpoint, source)

    artifact = Path(
        compat.prepare_legacy_fold_rollout_checkpoint(
            str(source), checkpoint=checkpoint
        )
    )
    first_mtime = artifact.stat().st_mtime_ns
    assert artifact != source
    assert artifact.is_file()
    assert Path(f"{artifact}.json").is_file()

    migrated = torch.load(artifact, map_location="cpu", weights_only=False)
    assert "optimizer_states" not in migrated
    assert "loops" not in migrated
    assert "callbacks" not in migrated
    assert migrated["state_dict"].keys() == checkpoint["state_dict"].keys()
    torch.testing.assert_close(
        migrated["state_dict"]["nets.policy.example"],
        checkpoint["state_dict"]["nets.policy.example"],
    )
    hyper_parameters = migrated["hyper_parameters"]
    migrated_model = OmegaConf.to_container(
        hyper_parameters["config_tree"], resolve=True
    )["model"]["robomimic_model"]
    assert migrated_model["action_horizon"] == 100
    assert (
        tuple(stage["_target_"] for stage in migrated_model["stages"])
        == compat._COMPAT_TARGETS
    )
    norm_state = hyper_parameters["norm_stats_state"]
    assert norm_state["embodiments"] == [3, 6]
    for field in ("key_types", "zarr_keys", "shapes", "norm_stats"):
        assert set(norm_state[field]) == {3, 6}

    again = compat.prepare_legacy_fold_rollout_checkpoint(
        str(source), checkpoint=checkpoint
    )
    assert again == str(artifact)
    assert artifact.stat().st_mtime_ns == first_mtime


def test_fold_migration_refuses_a_different_legacy_config(tmp_path: Path):
    model = _legacy_model_config()
    model["stages"][5]["moe_experts"] = 8
    checkpoint = _checkpoint(model)
    source = tmp_path / "different.ckpt"
    torch.save(checkpoint, source)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        compat.prepare_legacy_fold_rollout_checkpoint(
            str(source), checkpoint=checkpoint
        )


def test_fold_compat_head_refuses_architecture_guessing():
    common = dict(
        d_a=4,
        d_s=2,
        action_dim=3,
        chunk_len=5,
        d_model_a=8,
        n_layers=1,
        n_heads=2,
    )
    with pytest.raises(ValueError, match="denoiser='single'"):
        compat.DiffusionHead(**common, denoiser="dual")
    with pytest.raises(ValueError, match="does not support MoE"):
        compat.DiffusionHead(**common, denoiser="single", moe_experts=8)
    with pytest.raises(ValueError, match="heterogeneous actions"):
        compat.DiffusionHead(
            **common,
            denoiser="single",
            action_dims={"eva_bimanual": 20},
        )


def test_compat_loss_is_explicitly_train_only_for_rollout_planning():
    assert compat.MaskedActionLoss.train_only is True
