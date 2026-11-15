import json

import pytest

from reproductions.leact.reproduction import persist_selected
from reproductions.leact.training import prepare_data
from reproductions.leact.pipeline import generate_batch
from reproductions.leact.tests.test_pipeline import RecordingBackend, state


def test_sft_excludes_positive_deltas_from_invalid_baselines(tmp_path):
    raw = generate_batch([state()], "rules", RecordingBackend(), method="leact", n=2, m=2)
    metadata = {"rules": "rules", "sha256": "sha"}
    valid = prepare_data([state()], metadata, tmp_path / "valid", method="leact", raw=raw,
                         require_valid_policies=True)
    assert valid["sft_examples"] == 4 and valid["invalid_baseline_states"] == 0
    raw[0]["baselines"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError, match="No examples survived"):
        prepare_data([state()], metadata, tmp_path / "invalid", method="leact", raw=raw,
                     require_valid_policies=True)
    assert not (tmp_path / "invalid").exists()


def test_persistence_keeps_selected_weights_and_all_epoch_records(tmp_path):
    scratch_root = tmp_path / "scratch"
    scratch = scratch_root / "this-round"
    models = tmp_path / "persistent/checkpoints"
    models.parent.mkdir()
    for name in ["", "checkpoint-1", "checkpoint-2"]:
        directory = scratch / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.safetensors").write_bytes(name.encode() or b"final weights")
        (directory / "trainer_state.json").write_text(json.dumps({"epoch": name}))
    models.symlink_to(scratch, target_is_directory=True)
    selected = models / "checkpoint-1"
    assert persist_selected(models, selected, scratch_root) == selected
    assert not models.is_symlink() and not scratch.exists()
    assert (selected / "model.safetensors").read_bytes() == b"checkpoint-1"
    assert not (models / "checkpoint-2/model.safetensors").exists()
    assert (models / "checkpoint-2/trainer_state.json").exists()
    assert (models / "trainer_state.json").exists()
    assert json.loads((models / "weight_retention.json").read_text())["weight_sha256"]
    assert persist_selected(models, selected, scratch_root) == selected


def test_persistence_rejects_unowned_scratch_before_deleting_anything(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.safetensors").write_bytes(b"weights")
    models = tmp_path / "checkpoints"
    models.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside this task"):
        persist_selected(models, models, tmp_path / "owned")
    assert (outside / "model.safetensors").read_bytes() == b"weights"
    assert models.is_symlink()
