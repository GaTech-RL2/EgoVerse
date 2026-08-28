import copy
import importlib.util
import sys
from pathlib import Path
from re import DOTALL, findall
from subprocess import run

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).parents[1]
TRAIN_DIR = REPO_ROOT / "scripts" / "train"
SMOKE = TRAIN_DIR / "flow_transfer_newdata_h16_world2_dp_a40_smoke.sbatch"
FULL = TRAIN_DIR / "flow_transfer_newdata_h16_world2_dp_a40_full.sbatch"
VERIFIER = (
    REPO_ROOT
    / "egomimic"
    / "scripts"
    / "verify_newdata_dp_a40_full_smoke_gate.py"
)
COMMON_SMOKE_VERIFIER = (
    REPO_ROOT / "egomimic" / "scripts" / "verify_training_smoke.py"
)
CONFIG_DIR = REPO_ROOT / "egomimic" / "hydra_configs"
DP_EXPERIMENT = "pusht/pipeline_diffusion_usocket_chain_newdata_h16"


def _compose_dp(extra_overrides=()):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={DP_EXPERIMENT}", *extra_overrides],
        )


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_verifier():
    return _load_module(VERIFIER, "newdata_dp_a40_gate")


@pytest.mark.parametrize("launcher", [SMOKE, FULL])
def test_dp_a40_launchers_and_embedded_python_have_valid_syntax(
    launcher: Path,
) -> None:
    run(["bash", "-n", str(launcher)], check=True)
    blocks = findall(r"<<'PY'\n(.*?)\nPY", launcher.read_text(), flags=DOTALL)
    assert len(blocks) >= (5 if launcher == SMOKE else 7)
    for index, block in enumerate(blocks):
        compile(block, f"{launcher.name}:heredoc-{index}", "exec")
    compile(VERIFIER.read_text(), VERIFIER.name, "exec")

@pytest.mark.parametrize("launcher", [SMOKE, FULL])
def test_pre_activation_python_is_pinned(launcher: Path) -> None:
    text = launcher.read_text()
    pre_activation, _ = text.split('source "$PY_ENV/bin/activate"', maxsplit=1)
    assert "\npython " not in pre_activation
    assert '"$PY_ENV/bin/python" - "$PROVENANCE_DIR/slurm_job' in pre_activation




def test_smoke_is_dp_only_world2_a40_real_validation_gate() -> None:
    launcher = SMOKE.read_text()
    for directive in (
        "#SBATCH --partition=hoffman-lab",
        "#SBATCH --account=hoffman-lab",
        "#SBATCH --qos=short",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=2",
        "#SBATCH --time=02:00:00",
        "#SBATCH --gres=gpu:a40:2",
    ):
        assert directive in launcher
    assert "ARM=${ARM:?" not in launcher
    assert "latent_h16" not in launcher
    assert "pipeline_diffusion_usocket_chain_newdata_h16" in launcher
    assert "pipeline_sampler_usocket_chain_newdata" not in launcher
    assert 'test "${SLURM_NTASKS:?}" = "$EXPECTED_WORLD_SIZE"' in launcher
    assert 'test "${SLURM_JOB_PARTITION:?}" = hoffman-lab' in launcher
    assert 'test "${SLURM_JOB_ACCOUNT:?}" = hoffman-lab' in launcher
    assert 'grep -ci A40 "$PROVENANCE_DIR/gpu.txt"' in launcher
    assert 'expected_qos, expected_time, expected_requeue' in launcher
    assert 'assert "mem=128G" in fields["ReqTRES"].split(",")' in launcher
    assert "trainer.max_steps=2" in launcher
    assert "trainer.limit_train_batches=2" in launcher
    assert "trainer.val_check_interval=1" in launcher
    assert "trainer.limit_val_batches=1" in launcher
    assert "trainer.num_sanity_val_steps=0" in launcher
    assert "trainer.log_every_n_steps=1" in launcher
    assert "callbacks.model_checkpoint.every_n_train_steps=2" in launcher
    assert "callbacks.terminal_checkpoint.every_n_train_steps=1" in launcher
    assert "recovery_callback.state_key != terminal_callback.state_key" in launcher
    assert "logger.wandb.offline=true" in launcher
    assert "++logger.wandb.resume=never" in launcher
    assert launcher.count("--required-embodiments 19,20") == 2
    assert launcher.count("--dry-run") == 1
    assert launcher.index("--dry-run") < launcher.index(
        "# A live append or artifact mutation"
    ) < launcher.index("# Persist PASS only after")
    assert '"$SLURM_BIN/sbatch"' not in launcher


def test_both_launchers_pin_source_norm_inventory_and_episode_metadata() -> None:
    for launcher_path in (SMOKE, FULL):
        launcher = launcher_path.read_text()
        for contract in (
            "EXPECTED_HEAD=${EXPECTED_HEAD:?",
            "EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?",
            "NORM_ARTIFACT=${NORM_ARTIFACT:?",
            "EXPECTED_NORM_SHA=${EXPECTED_NORM_SHA:?",
            "U_INVENTORY=${U_INVENTORY:?",
            "EXPECTED_U_INVENTORY_SHA=${EXPECTED_U_INVENTORY_SHA:?",
            "U_EPISODE_METADATA=${U_EPISODE_METADATA:?",
            "EXPECTED_U_EPISODE_METADATA_SHA=${EXPECTED_U_EPISODE_METADATA_SHA:?",
            "CHAIN_BASE_INVENTORY=${CHAIN_BASE_INVENTORY:?",
            "EXPECTED_CHAIN_BASE_INVENTORY_SHA=${EXPECTED_CHAIN_BASE_INVENTORY_SHA:?",
            "CHAIN_BASE_EPISODE_METADATA=${CHAIN_BASE_EPISODE_METADATA:?",
            "EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA=${EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA:?",
            "CHAIN_GEN_INVENTORY=${CHAIN_GEN_INVENTORY:?",
            "EXPECTED_CHAIN_GEN_INVENTORY_SHA=${EXPECTED_CHAIN_GEN_INVENTORY_SHA:?",
            "CHAIN_GEN_EPISODE_METADATA=${CHAIN_GEN_EPISODE_METADATA:?",
            "EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA=${EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA:?",
        ):
            assert contract in launcher
        assert 'test -z "$(git -C "$REPO" status --porcelain=v1' in launcher
        assert "live inventory differs from pinned artifact" in launcher
        assert "live episode metadata differs from pinned artifact" in launcher
        assert "episode_T_chain_gripper_obs7_000050" in launcher
        assert "EXCLUDED_CHAIN_FRAMES=3118" in launcher
        assert "effective_chain_frames" in launcher
        assert "1237652" in launcher


def test_full_is_dp_only_two_day_a40_and_semantically_smoke_gated() -> None:
    launcher = FULL.read_text()
    for directive in (
        "#SBATCH --partition=hoffman-lab",
        "#SBATCH --account=hoffman-lab",
        "#SBATCH --qos=long",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=2",
        "#SBATCH --time=2-00:00:00",
        "#SBATCH --gres=gpu:a40:2",
        "#SBATCH --requeue",
        "#SBATCH --signal=USR1@300",
    ):
        assert directive in launcher
    assert "ARM=${ARM:?" not in launcher
    assert "latent" not in launcher.lower()
    assert "PACE" not in launcher
    assert "DP_SMOKE_RESULT=${DP_SMOKE_RESULT:?" in launcher
    assert "EXPECTED_DP_SMOKE_RESULT_SHA=${EXPECTED_DP_SMOKE_RESULT_SHA:?" in launcher
    assert "EXPECTED_SMOKE_LAUNCHER_SHA=${EXPECTED_SMOKE_LAUNCHER_SHA:?" in launcher
    assert "--expected-smoke-launcher-sha" in launcher
    assert "--latent-result" not in launcher
    assert launcher.count("verify_smoke_gate") >= 3
    assert launcher.index("smoke_gate_attempt_${RESTART_COUNT}_before.json") < launcher.index(
        "smoke_gate_attempt_${RESTART_COUNT}_after.json"
    )
    assert "cmp -s" in launcher
    assert 'test "${SLURM_JOB_PARTITION:?}" = hoffman-lab' in launcher
    assert 'test "${SLURM_JOB_ACCOUNT:?}" = hoffman-lab' in launcher
    assert 'grep -ci A40 "$PROVENANCE_DIR/gpu_attempt_${RESTART_COUNT}.txt"' in launcher
    assert 'expected_qos, expected_time, expected_requeue' in launcher
    assert 'assert fields["TimeLimit"] == expected_time' in launcher
    assert 'assert "mem=128G" in fields["ReqTRES"].split(",")' in launcher


def test_full_has_required_training_recovery_and_reserved_wandb_contract() -> None:
    launcher = FULL.read_text()
    assert "trainer.max_steps=240000" in launcher
    assert "trainer.limit_train_batches=1.0" in launcher
    assert "trainer.limit_val_batches=0" in launcher
    assert "trainer.num_sanity_val_steps=0" in launcher
    assert "trainer.log_every_n_steps=1" in launcher
    assert 'cfg.trainer.precision == "bf16"' in launcher
    assert 'cfg.trainer.get("gradient_clip_val") is None' in launcher
    assert "cfg.model.enable_grad_norm is False" in launcher
    assert "cfg.model.optimizer.lr == 3.0e-5" in launcher
    assert "cfg.model.scheduler.eta_min == 3.0e-6" in launcher
    assert "cfg.model.scheduler.warmup_steps == 3_000" in launcher
    assert "cfg.model.scheduler.max_steps == 240_000" in launcher
    assert "batch_size * 2 == 64" in launcher
    assert "recovery.train_time_interval.hours == 1" in launcher
    assert "terminal.every_n_train_steps == 240_000" in launcher
    assert "ft_cotrain_newdata3719_dp_h16_s42_world2_a40x2_20260828" in launcher
    assert "world2_l40s" not in launcher
    assert "WANDB_GROUP=flow_transfer_newdata3719_cotrain_h16_20260828" in launcher
    assert "export WANDB_MODE=online" in launcher
    assert "logger.wandb.offline=false" in launcher
    assert launcher.index("WANDB_RESUME=never") < launcher.index("WANDB_RESUME=allow")
    assert 'test "${#RESUME_CANDIDATES[@]}" -gt 0' in launcher
    assert 'TERMINAL_CANDIDATES=("$RUN_DIR"/checkpoints/final/*.ckpt)' in launcher
    assert "path.stat().st_mtime_ns" in launcher
    assert 'COMMON_OVERRIDES+=("ckpt_path=$RESUME_CKPT")' in launcher


def test_dp_full_experiment_resolves_to_required_h16_curve_and_callbacks() -> None:
    cfg = _compose_dp()
    assert cfg.trainer.max_steps == 240_000
    assert cfg.trainer.limit_train_batches == 1.0
    assert cfg.trainer.limit_val_batches == 0
    assert cfg.trainer.num_sanity_val_steps == 0
    assert cfg.trainer.precision == "bf16"
    assert cfg.trainer.strategy == "ddp"
    assert cfg.launch_params.gpus_per_node == 2
    assert cfg.launch_params.nodes == 1
    assert cfg.model.enable_grad_norm is False
    assert cfg.trainer.get("gradient_clip_val") is None
    assert cfg.model.robomimic_model.action_horizon == 16
    assert cfg.model.optimizer.lr == pytest.approx(3.0e-5)
    assert cfg.model.scheduler.eta_min == pytest.approx(3.0e-6)
    assert cfg.model.scheduler.warmup_steps == 3_000
    assert cfg.model.scheduler.max_steps == 240_000
    for params in cfg.data.train_dataloader_params.values():
        assert params.batch_size == 32
        assert params.batch_size * 2 == 64
    recovery = cfg.callbacks.model_checkpoint
    assert recovery.every_n_epochs is None
    assert recovery.every_n_train_steps is None
    assert recovery.train_time_interval._target_ == "datetime.timedelta"
    assert recovery.train_time_interval.hours == 1
    assert recovery.save_last is True
    terminal = cfg.callbacks.terminal_checkpoint
    assert terminal.every_n_train_steps == 240_000
    assert terminal.save_top_k == 1
    assert terminal.save_last is False


def test_full_embedded_resolved_config_gate_executes(tmp_path: Path) -> None:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    run_dir = tmp_path / "run"
    repo = tmp_path / "repo"
    norm = tmp_path / "norm_stats.json"
    wandb_id = "ft_cotrain_newdata3719_dp_h16_s42_world2_a40x2_20260828"
    cfg = _compose_dp()
    OmegaConf.update(cfg, "paths.root_dir", str(run_dir), force_add=True)
    OmegaConf.update(cfg, "paths.output_dir", str(run_dir), force_add=True)
    OmegaConf.update(cfg, "paths.work_dir", str(repo), force_add=True)
    cfg.ckpt_path = None
    cfg.logger.wandb.offline = False
    OmegaConf.update(cfg, "logger.wandb.id", wandb_id, force_add=True)
    OmegaConf.update(
        cfg,
        "logger.wandb.group",
        "flow_transfer_newdata3719_cotrain_h16_20260828",
    )
    OmegaConf.update(cfg, "logger.wandb.name", wandb_id, force_add=True)
    OmegaConf.update(cfg, "logger.wandb.resume", "never", force_add=True)
    cfg.norm_stats.precomputed_norm_path = str(norm)
    cfg.norm_stats.save_cache_dir = None
    config_path = tmp_path / "resolved_config.yaml"
    OmegaConf.save(cfg, config_path, resolve=True)

    blocks = findall(r"<<'PY'\n(.*?)\nPY", FULL.read_text(), flags=DOTALL)
    preflight = next(
        block
        for block in blocks
        if "assert recovery.get(\"monitor\") is None" in block
    )
    completed = run(
        [
            sys.executable,
            "-",
            str(config_path),
            str(norm),
            str(run_dir),
            str(repo),
            wandb_id,
            "never",
            "",
            "episode_T_chain_gripper_obs7_000050",
        ],
        input=preflight,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_dp_a40_smoke_checkpoint_callbacks_have_distinct_state_keys() -> None:
    cfg = _compose_dp(
        [
            "trainer.max_steps=2",
            "callbacks.model_checkpoint.every_n_train_steps=2",
            "callbacks.model_checkpoint.train_time_interval=null",
            "callbacks.model_checkpoint.save_on_train_epoch_end=false",
            "callbacks.terminal_checkpoint.every_n_train_steps=1",
            "paths.output_dir=/tmp/dp_a40_smoke_callback_test",
        ]
    )
    assert cfg.callbacks.model_checkpoint.every_n_train_steps == 2
    assert cfg.callbacks.terminal_checkpoint.every_n_train_steps == 1
    recovery = instantiate(cfg.callbacks.model_checkpoint)
    terminal = instantiate(cfg.callbacks.terminal_checkpoint)
    assert recovery.state_key != terminal.state_key


def test_semantic_gate_requires_dense_non_epoch_rows_and_scheduled_real_val() -> None:
    verifier = _load_verifier()
    row = {
        "trainer_global_step": 0,
        "train_metrics": {"Train/Loss": 1.0},
        "timing_metrics": {
            "Timing/Process_Batch_Sec": 1.0,
            "Timing/Forward_Pass_Sec": 0.4,
            "Timing/Compute_Losses_Sec": 0.3,
        },
        "optimizer_metrics": {"Optimizer/param_group_0_lr": 3.0e-6},
    }
    payload = {
        "training_history": [row, {**copy.deepcopy(row), "trainer_global_step": 1}],
        "dense_training_steps": [0, 1],
        "validation_trainer_global_step": 1,
        "validation_metrics": {
            "Valid/emb19_actions_action_mse": 0.1,
            "Valid/emb20_actions_action_mse": 0.2,
        },
        "validation_history": [
            {
                "trainer_global_step": 1,
                "validation_metrics": {
                    "Valid/emb19_actions_action_mse": 0.1,
                    "Valid/emb20_actions_action_mse": 0.2,
                },
            }
        ],
    }
    assert verifier._verify_dense_per_step_history(payload) == [0, 1]
    assert verifier._verify_real_validation(payload)[0] == 1
    bad = copy.deepcopy(payload)
    bad["training_history"][0]["train_metrics"]["Train/Loss_epoch"] = 1.0
    with pytest.raises(AssertionError):
        verifier._verify_dense_per_step_history(bad)
    no_val = copy.deepcopy(payload)
    no_val["validation_trainer_global_step"] = 0
    with pytest.raises(AssertionError):
        verifier._verify_real_validation(no_val)


@pytest.mark.parametrize(
    ("path", "name"),
    [
        (COMMON_SMOKE_VERIFIER, "common_training_smoke_verifier"),
        (VERIFIER, "newdata_dp_a40_semantic_gate"),
    ],
)
def test_smoke_verifiers_register_eval_resolver_used_by_saved_hydra_config(
    path: Path,
    name: str,
) -> None:
    module = _load_module(path, name)
    OmegaConf.clear_resolver("eval")
    module._register_training_config_resolvers()
    cfg = OmegaConf.create({"devices": "${eval:'2 * 1'}"})
    assert cfg.devices == 2


def test_wandb_terminal_exit_gate_rejects_failed_stream(monkeypatch) -> None:
    module = _load_module(COMMON_SMOKE_VERIFIER, "wandb_exit_gate")
    record = module.wandb_internal_pb2.Record()
    record.exit.exit_code = 1
    payload = record.SerializeToString()

    class FakeDataStore:
        def __init__(self) -> None:
            self.payloads = [payload, None]
            self.closed = False

        def open_for_scan(self, path: str) -> None:
            assert path.endswith("failed.wandb")

        def scan_data(self):
            return self.payloads.pop(0)

        def close(self) -> None:
            self.closed = True

    fake_store = FakeDataStore()
    monkeypatch.setattr(module, "DataStore", lambda: fake_store)
    with pytest.raises(AssertionError):
        module.read_successful_wandb_exit_code(Path("/tmp/failed.wandb"))
    assert fake_store.closed
