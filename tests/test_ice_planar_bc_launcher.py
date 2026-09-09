import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts" / "ice" / "launch_planar_bc.sbatch"


def test_launcher_is_valid_bash_and_has_single_gpu_requeue_contract():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = LAUNCHER.read_text()

    assert "#SBATCH --account=coc" in text
    assert "#SBATCH --partition=ice-gpu" in text
    assert "#SBATCH --qos=coc-ice" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks-per-node=1" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --constraint=H100|H200" in text
    assert "#SBATCH --requeue" in text
    assert "#SBATCH --signal=B:USR1@600" in text
    assert "trainer.strategy=auto" in text
    assert "trainer.devices=1" in text
    assert "trainer.num_nodes=1" in text
    assert "trainer.precision=$ICE_PRECISION" in text
    assert "ICE_EXPECTED_GPU_NAME" not in text
    assert text.count("--allowed-gpu-name") == 2


def test_launcher_uses_strict_external_runner_and_completion_sentinel():
    text = LAUNCHER.read_text()

    assert 'runtime.slurm_requeue_owner=runner' in text
    assert 'runtime.slurm_save_signal=SIGUSR2' in text
    assert "--checkpoint-validator" in text
    assert "--checkpoint-signal USR2" in text
    assert "--checkpoint-forwarding slurm-steps" in text
    assert "--requeue-owner runner" in text
    assert "--confirm-child-requeue-disabled" in text
    assert 'ICE_OUTPUT_DIR/COMPLETE.json' in text
    assert "allow-unvalidated" not in text
    assert "ICE_RESUME_CHECKPOINT" in text
    assert "ICE_RESUME_CHECKPOINT_SHA256" in text
    assert "ICE_RESUME_GLOBAL_STEP" in text


def test_parent_runner_inherits_exact_repo_pythonpath_for_checkpoint_validation():
    text = LAUNCHER.read_text()

    parent_export = text.index("export PYTHONPATH=$ICE_REPO")
    preflight_runner = text.index('"$ICE_PYTHON" "$ICE_REQUEUE_RUNNER"')
    final_runner = text.rindex('exec "$ICE_PYTHON" "$ICE_REQUEUE_RUNNER"')
    child_definition = text.index("read -r -d '' CHILD_CODE")

    assert parent_export < preflight_runner < child_definition < final_runner


def test_launcher_exposes_supported_bc_configs_and_explicit_training_limits():
    text = LAUNCHER.read_text()

    assert "pusht/planar_v2_usocket_direct_bc" in text
    assert "pusht/planar_v2_usocket_arc_bc" in text
    assert "pusht/planar_v2_usocket_arc_hybrid_D40_M16_R24deg_bc" in text
    assert text.count(
        "data.train_datasets.pushshapes_sim_u_socket.resolver.folder_path="
    ) == 1
    assert text.count(
        "data.valid_datasets.pushshapes_sim_u_socket.resolver.folder_path="
    ) == 1
    for value in (
        "trainer.max_steps=$ICE_MAX_STEPS",
        "trainer.val_check_interval=$ICE_VAL_CHECK_INTERVAL",
        "trainer.limit_train_batches=$ICE_LIMIT_TRAIN_BATCHES",
        "trainer.limit_val_batches=$ICE_LIMIT_VAL_BATCHES",
        "+callbacks.model_checkpoint.every_n_train_steps=$ICE_CHECKPOINT_EVERY_N_STEPS",
    ):
        assert value in text
    assert "unique ICE_OUTPUT_DIR already exists" in text
    assert "ICE_LAUNCH_MODE" in text
    assert "--dry-run" in text
    assert "validate_planar_dataset.py" in text
    assert "ICE_EXPECTED_SPLIT_MANIFEST_SHA256" in text
    assert "--expected-manifest-sha256" in text
    assert '"++paths.root_dir=$ICE_REPO"' in text
    assert '"norm_stats.save_cache_dir=$ICE_OUTPUT_DIR"' in text
    assert '"norm_stats.save_cache_dir=$ICE_OUTPUT_DIR/norm_stats"' not in text
    assert "$ICE_OUTPUT_DIR/norm_stats/norm_stats.json" in text
    assert "requeue requires the original run's norm_stats/norm_stats.json" in text
    assert "norm_stats.save_cache_dir=null" in text


def test_launcher_contains_no_deployment_identity_or_credentials():
    text = LAUNCHER.read_text()

    forbidden = (
        "/coc/",
        "/storage/",
        "paphiwetsa3",
        "WANDB_API_KEY=",
        "wandb.ai/",
    )
    assert not any(value in text for value in forbidden)
    assert re.search(r"\b[0-9]{7}\b", text) is None
    assert "BASH_SOURCE" not in text
