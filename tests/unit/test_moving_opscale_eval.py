"""Protect validation isolation and fail-closed completion for moving opscale."""

import copy
from types import SimpleNamespace

import pytest

from egomimic.eval.moving_opscale_eval import (
    MovingOpscaleEvaluation,
    validate_selections,
)
from egomimic.eval.opscale_reference_replay import chunk_step_for_video_frame


def _selection():
    operators = [f"operator_{i}" for i in range(4)]
    episodes = {}
    split = {
        "operators": operators,
        "train": [f"train_{i}" for i in range(4)],
        "seen_matched": [],
        "seen_perlevel": [],
        "unseen": [],
        "video_seen_matched": [],
        "video_seen_perlevel": [],
        "video_unseen": [],
    }
    for cohort, ops, n in [
        ("seen_matched", operators, 10),
        ("seen_perlevel", operators, 3),
        ("unseen", [f"unseen_{i}" for i in range(5)], 20),
    ]:
        for operator in ops:
            ids = [f"{cohort}_{operator}_{j}" for j in range(n)]
            split[cohort].extend(ids)
            split[f"video_{cohort}"].append(ids[0])
            episodes.update({eid: {"user_id": operator} for eid in ids})
    return split, episodes


def test_validation_pins_require_distinct_operators_and_no_leakage():
    split, episodes = _selection()
    assert {k: len(v) for k, v in validate_selections(split, episodes).items()} == {
        "seen_matched": 4,
        "seen_perlevel": 4,
        "unseen": 5,
    }
    leaking = copy.deepcopy(split)
    leaking["train"].append(leaking["unseen"][0])
    with pytest.raises(ValueError, match="overlap"):
        validate_selections(leaking, episodes)
    repeated = copy.deepcopy(split)
    repeated["video_seen_matched"][1] = repeated["seen_matched"][1]
    with pytest.raises(ValueError, match="repeated visualization operator"):
        validate_selections(repeated, episodes)


def test_full_episode_coverage_is_enforced_before_rendering(tmp_path):
    split, episodes = _selection()
    callback = MovingOpscaleEvaluation(str(tmp_path), "example")
    callback.split, callback.episodes = split, episodes
    callback.observed = {
        cohort: set(split[cohort])
        for cohort in ("seen_matched", "seen_perlevel", "unseen")
    }
    callback.observed["unseen"].remove(split["unseen"][-1])
    trainer = SimpleNamespace(sanity_checking=False)
    with pytest.raises(RuntimeError, match="Incomplete unseen evaluation"):
        callback.on_validation_epoch_end(trainer, None)
    assert not (tmp_path / "evaluation_complete.json").exists()


def test_smoke_training_never_claims_full_experiment_completion(tmp_path):
    callback = MovingOpscaleEvaluation(str(tmp_path), "example")
    callback.final_evidence = {"global_step": 100}
    callback.on_fit_end(
        SimpleNamespace(global_step=100, default_root_dir=str(tmp_path)), None
    )
    assert not (tmp_path / "evaluation_complete.json").exists()


def test_reference_overlay_matches_source_frame_clock():
    assert chunk_step_for_video_frame(0, 30, 100) == 0
    assert chunk_step_for_video_frame(15, 30, 100) == 51
    assert chunk_step_for_video_frame(29, 30, 100) == 99


def test_lightning_epoch_hook_sees_current_cohort_metrics(tmp_path):
    """Exercise actual Lightning callback ordering; no policy or training."""
    import json

    import torch
    from lightning.pytorch import LightningModule, Trainer
    from torch.utils.data import DataLoader

    split, episodes = _selection()
    callback = MovingOpscaleEvaluation(str(tmp_path), "example")
    callback.split, callback.episodes = split, episodes
    callback.pins = validate_selections(split, episodes)
    cohorts = ["seen_matched", "seen_perlevel", "unseen"]

    class ValidationOnly(LightningModule):
        val_loader_names = ["valid", "extra_valid", "unseen_op_valid"]

        def validation_step(self, batch, batch_idx, dataloader_idx=0):
            self.log(
                f"val/{cohorts[dataloader_idx]}/Valid/action_loss",
                torch.tensor(float(dataloader_idx + 1)),
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
                batch_size=len(batch["human_bimanual"]["episode_hash"]),
            )

    loaders = [
        DataLoader(
            [
                {
                    "human_bimanual": {
                        "episode_hash": eid,
                        "evaluation_tracking_valid": True,
                    }
                }
                for eid in split[cohort]
            ],
            batch_size=8,
        )
        for cohort in cohorts
    ]
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[callback],
        default_root_dir=str(tmp_path),
    )
    trainer.validate(ValidationOnly(), dataloaders=loaders, verbose=False)
    status = json.loads((tmp_path / "evaluation_status.json").read_text())
    assert status["coverage"]["seen_matched"] == {"expected": 40, "observed": 40}
    assert status["metrics"] == {
        f"val/{cohort}/Valid/action_loss": float(i + 1)
        for i, cohort in enumerate(cohorts)
    }
    assert not (tmp_path / "evaluation_complete.json").exists()
