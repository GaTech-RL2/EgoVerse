from pathlib import Path

from egomimic.trainHydra import _resolve_slurm_requeue_checkpoint


def test_non_requeue_preserves_configured_checkpoint(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    configured = "/checkpoints/explicit.ckpt"

    assert _resolve_slurm_requeue_checkpoint(str(tmp_path), configured) == configured


def test_requeue_uses_existing_last_checkpoint(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    last_checkpoint = checkpoint_dir / "last.ckpt"
    last_checkpoint.touch()

    assert _resolve_slurm_requeue_checkpoint(str(tmp_path), None) == str(
        last_checkpoint
    )


def test_requeue_without_last_checkpoint_stays_fresh(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")

    assert _resolve_slurm_requeue_checkpoint(str(tmp_path), None) is None


def test_requeue_without_last_checkpoint_preserves_explicit_checkpoint(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    configured = "/checkpoints/explicit.ckpt"

    assert _resolve_slurm_requeue_checkpoint(str(tmp_path), configured) == configured
