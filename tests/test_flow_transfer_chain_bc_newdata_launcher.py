from pathlib import Path
from re import DOTALL, findall
from subprocess import run

LAUNCHER = (
    Path(__file__).parents[1]
    / "scripts"
    / "train"
    / "flow_transfer_chain_bc_newdata_h16.sbatch"
)


def test_chain_bc_newdata_launcher_has_valid_bash_syntax() -> None:
    run(["bash", "-n", str(LAUNCHER)], check=True)


def test_chain_bc_newdata_launcher_embedded_python_compiles() -> None:
    blocks = findall(r"<<'PY'\n(.*?)\nPY", LAUNCHER.read_text(), flags=DOTALL)
    assert len(blocks) >= 8
    for index, block in enumerate(blocks):
        compile(block, f"{LAUNCHER.name}:heredoc-{index}", "exec")


def test_chain_bc_newdata_launcher_pins_arms_resources_and_data() -> None:
    launcher = LAUNCHER.read_text()

    for contract in (
        "ARM=${ARM:?set ARM=latent_h16 or dp}",
        "MODE=${MODE:?set MODE=smoke or full}",
        "EXPECTED_HEAD=${EXPECTED_HEAD:?",
        "EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?",
        "CHAIN_BASE_INVENTORY=${CHAIN_BASE_INVENTORY:?",
        "EXPECTED_CHAIN_BASE_INVENTORY_SHA=${EXPECTED_CHAIN_BASE_INVENTORY_SHA:?",
        "CHAIN_BASE_EPISODE_METADATA=${CHAIN_BASE_EPISODE_METADATA:?",
        "EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA=${EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA:?",
        "CHAIN_GEN_INVENTORY=${CHAIN_GEN_INVENTORY:?",
        "EXPECTED_CHAIN_GEN_INVENTORY_SHA=${EXPECTED_CHAIN_GEN_INVENTORY_SHA:?",
        "CHAIN_GEN_EPISODE_METADATA=${CHAIN_GEN_EPISODE_METADATA:?",
        "EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA=${EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA:?",
        "NORM_ARTIFACT=${NORM_ARTIFACT:?",
        "EXPECTED_NORM_SHA=${EXPECTED_NORM_SHA:?",
        "NORM_VALIDATION=${NORM_VALIDATION:?",
        "EXPECTED_NORM_VALIDATION_SHA=${EXPECTED_NORM_VALIDATION_SHA:?",
    ):
        assert contract in launcher

    assert "latent_h16)" in launcher
    assert "dp)" in launcher
    assert "pipeline_sampler_chain_gripper_newdata_points_dense_medium_h16" in launcher
    assert "pipeline_diffusion_chain_gripper_newdata_points_h16" in launcher
    assert 'test "${SLURM_JOB_PARTITION:?}" = hoffman-lab' in launcher
    assert 'test "${SLURM_JOB_ACCOUNT:?}" = hoffman-lab' in launcher
    assert "TimeLimit=2-00:00:00" in launcher
    assert 'if test "$MODE" = full; then' in launcher
    assert "EXPECTED_WORLD_SIZE=1" in launcher
    assert "EXPECTED_LOCAL_BATCH=64" in launcher
    assert "--ntasks=1 --gpus-per-task=1" in launcher
    assert "grep -ciE '(^|[ ,])A40([ ,]|$)'" in launcher

    for value in (
        "EXPECTED_CHAIN_BASE_COUNT=3000",
        "EXPECTED_CHAIN_BASE_FRAMES=931061",
        "EXPECTED_CHAIN_GEN_COUNT=720",
        "EXPECTED_CHAIN_GEN_SOURCE_FRAMES=309709",
        "EXPECTED_CHAIN_GEN_EFFECTIVE_COUNT=719",
        "EXPECTED_CHAIN_GEN_EFFECTIVE_FRAMES=306591",
        "EXPECTED_EXCLUDED_FRAMES=3118",
        "EXPECTED_CHAIN_EFFECTIVE_COUNT=3719",
        "EXPECTED_CHAIN_EFFECTIVE_FRAMES=1237652",
        "EXCLUDED_EPISODE=episode_T_chain_gripper_obs7_000050",
    ):
        assert value in launcher
    assert launcher.count("validate_chain_contract") >= 3
    assert "inventory drift" in launcher
    assert "metadata drift" in launcher


def test_chain_bc_newdata_launcher_pins_training_and_logging_contract() -> None:
    launcher = LAUNCHER.read_text()

    assert "WANDB_PROJECT=pushshapes-flow-transfer" in launcher
    assert "WANDB_GROUP=flow_transfer_chain_newdata3719_bc_h16_20260828" in launcher
    assert "ft_bc_chain_newdata3719_latent_dense_m96_h16_s42_20260828" in launcher
    assert "ft_bc_chain_newdata3719_dp_h16_s42_20260828" in launcher
    assert "WANDB_RESUME=never" in launcher
    assert "WANDB_RESUME=allow" in launcher
    assert "trainer.precision=bf16" in launcher
    assert "trainer.accumulate_grad_batches=1" in launcher
    assert "trainer.log_every_n_steps=1" in launcher
    assert "trainer.max_steps=240000" in launcher
    assert "trainer.limit_val_batches=0" in launcher
    assert "callbacks.model_checkpoint.every_n_train_steps=null" in launcher
    assert "checkpoint.train_time_interval.hours == 1" in launcher
    assert "terminal.every_n_train_steps == cfg.trainer.max_steps" in launcher
    assert "cfg.model.optimizer.lr == 3.0e-5" in launcher
    assert "cfg.model.scheduler.warmup_steps == 3_000" in launcher
    assert "cfg.model.scheduler.eta_min == 3.0e-6" in launcher
    assert "cfg.model.train_metrics_on_step is True" in launcher
    assert "cfg.model.train_metrics_on_epoch is True" in launcher
    assert "gradient_clip_val" in launcher
    assert "action_encoder" in launcher
    assert "arc_length" in launcher


def test_chain_bc_newdata_launcher_smoke_and_full_are_fail_closed() -> None:
    launcher = LAUNCHER.read_text()

    assert "trainer.max_steps=2" in launcher
    assert "trainer.limit_train_batches=2" in launcher
    assert "trainer.val_check_interval=1" in launcher
    assert "trainer.limit_val_batches=1" in launcher
    assert "trainer.num_sanity_val_steps=0" in launcher
    assert "--required-embodiments 20" in launcher
    assert '--expected-world-size "$EXPECTED_WORLD_SIZE"' in launcher
    assert "--dry-run" in launcher
    assert "verify_training_smoke.py" in launcher
    assert "WANDB_MODE=offline" in launcher
    assert "logger.wandb.offline=true" in launcher
    assert "Fatal signature found in combined training stdout/stderr" in launcher

    assert "SMOKE_RESULT=${SMOKE_RESULT:?" in launcher
    assert "EXPECTED_SMOKE_RESULT_SHA=${EXPECTED_SMOKE_RESULT_SHA:?" in launcher
    assert "stat -c '%a' \"$SMOKE_RESULT\"" in launcher
    assert "validate_semantic_smoke" in launcher
    assert 'result["launcher_status"] == "PASS"' in launcher
    assert 'result["dense_training_steps"] == [0, 1]' in launcher
    assert 'result["validation_trainer_global_step"] >= 1' in launcher
    assert 'result["required_embodiments"] == [20]' in launcher
    assert 'result["gpu_model"] == "A40"' in launcher
    assert 'result["global_batch"] == 64' in launcher
    assert 'result["chain_effective_episodes"] == 3719' in launcher
    assert 'result["chain_effective_frames"] == 1237652' in launcher
    assert 'chmod 0444 "$RUN_DIR/SMOKE_RESULT.json"' in launcher
    assert "terminal checkpoint is missing" in launcher

    # This file executes inside an allocation. It must not submit another job.
    assert "\nsbatch " not in launcher
