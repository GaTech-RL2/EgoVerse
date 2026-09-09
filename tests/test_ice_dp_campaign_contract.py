from pathlib import Path

ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts/ice/launch_planar_v2_dp_cotrain_clean.sbatch"
README = ROOT / "scripts/ice/README.md"


def test_dp_ice_launcher_has_one_entrypoint_and_fresh_run_guards():
    text = LAUNCHER.read_text()
    assert "MODE=${MODE:-preflight}" in text
    assert "case \"$MODE\" in preflight|smoke|full)" in text
    assert "ckpt_path=null" in text
    assert "test -z \"${ICE_INITIAL_CHECKPOINT:-}\"" in text
    assert "test -z \"${ICE_INITIAL_CHECKPOINT_SHA256:-}\"" in text
    assert "test -z \"${CKPT_PATH:-}\"" in text
    assert "PASSED_SMOKE_DIR" in text
    assert "trainer.max_steps=$max_steps" in text
    assert "FULL_STEPS=240000" in text
    assert "VALIDATE_EVERY=10000" in text
    assert "CHECKPOINT_EVERY=20000" in text
    assert "NVIDIA H100 80GB HBM3" in text and "NVIDIA H200" in text


def test_dp_ice_docs_distinguish_smoke_full_and_requeue_from_restore():
    text = README.read_text()
    assert "Do not infer full-run completion from a completed smoke" in text
    assert "scheduler-generated\nHPC checkpoint" in text
    assert "H100 and H200 are\ninterchangeable" in text
