import ast
import json
import sys
from pathlib import Path
from re import DOTALL, findall
from subprocess import CalledProcessError, run

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import egomimic.scripts.verify_training_smoke as smoke_verifier
from egomimic.scripts.verify_training_smoke import _load_training_config

ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "egomimic" / "hydra_configs"
LAUNCHER = (
    ROOT / "scripts" / "train" / "flow_transfer_newdata_h16_pace_h200_latent.sbatch"
)
EXPERIMENT = "pusht/pipeline_sampler_usocket_chain_newdata_dense_medium_h16"
U_DATA = (
    "/storage/project/r-dxu345-0/paphiwetsa3/datasets/flow_transfer_20260826/"
    "u_socket_3000_v2_clean"
)
CHAIN_BASE = (
    "/storage/project/r-dxu345-0/paphiwetsa3/datasets/"
    "flow_transfer_newdata_h16_20260828/chain_gripper_3000_v2"
)
CHAIN_GEN = (
    "/storage/project/r-dxu345-0/paphiwetsa3/datasets/"
    "flow_transfer_newdata_h16_20260828/chain_gripper_gen"
)
RUN_DIR = "/tmp/pace_latent_launcher_contract_run"
WANDB_ID = "ft_cotrain_newdata3719_latent_dense_m96_h16_s42_world1_pace_h200_20260828"
WANDB_GROUP = "flow_transfer_newdata3719_cotrain_h16_20260828"
NORM_ARTIFACT = "/tmp/pinned_norm_stats.json"


def _embedded_block(marker: str) -> str:
    blocks = findall(r"<<'PY'\n(.*?)\nPY", LAUNCHER.read_text(), flags=DOTALL)
    matches = [block for block in blocks if marker in block]
    assert len(matches) == 1, (marker, len(matches))
    return matches[0]


def _run_embedded(block: str, *args: object) -> None:
    run(
        [sys.executable, "-", *(str(arg) for arg in args)],
        input=block,
        text=True,
        check=True,
    )


def _compose(mode: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    overrides = [
        f"+experiment={EXPERIMENT}",
        "launch_params.gpus_per_node=1",
        "launch_params.nodes=1",
        "data.train_dataloader_params.pushshapes_sim_u_socket.batch_size=64",
        "data.train_dataloader_params.pushshapes_sim_chain_gripper.batch_size=64",
        (f"data.train_datasets.pushshapes_sim_u_socket.resolver.folder_path={U_DATA}"),
        (f"data.valid_datasets.pushshapes_sim_u_socket.resolver.folder_path={U_DATA}"),
        (
            "data.train_datasets.pushshapes_sim_chain_gripper.resolver."
            f"folder_paths=[{CHAIN_BASE},{CHAIN_GEN}]"
        ),
        (
            "data.valid_datasets.pushshapes_sim_chain_gripper.resolver."
            f"folder_paths=[{CHAIN_BASE},{CHAIN_GEN}]"
        ),
        "trainer.accumulate_grad_batches=1",
        "trainer.log_every_n_steps=1",
    ]
    if mode == "smoke":
        overrides.extend(
            [
                "trainer.max_steps=2",
                "trainer.limit_train_batches=2",
                "trainer.val_check_interval=1",
                "trainer.limit_val_batches=1",
                "trainer.num_sanity_val_steps=0",
                "callbacks.model_checkpoint.every_n_epochs=null",
                "callbacks.model_checkpoint.every_n_train_steps=2",
                "callbacks.model_checkpoint.train_time_interval=null",
                "callbacks.model_checkpoint.save_on_train_epoch_end=false",
                "callbacks.terminal_checkpoint.every_n_train_steps=1",
            ]
        )
    else:
        assert mode == "full"
        overrides.extend(
            [
                "trainer.max_steps=240000",
                "trainer.limit_train_batches=1.0",
                "trainer.val_check_interval=10000",
                "trainer.limit_val_batches=0",
                "trainer.num_sanity_val_steps=0",
                "callbacks.model_checkpoint.every_n_epochs=null",
                "callbacks.model_checkpoint.every_n_train_steps=null",
                "callbacks.model_checkpoint.save_on_train_epoch_end=true",
                "callbacks.terminal_checkpoint.every_n_train_steps=240000",
            ]
        )
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="train_zarr_cartesian", overrides=overrides)


def test_pace_h200_latent_launcher_has_valid_bash_syntax() -> None:
    run(["bash", "-n", str(LAUNCHER)], check=True)


def test_pace_h200_latent_launcher_embedded_python_compiles() -> None:
    blocks = findall(r"<<'PY'\n(.*?)\nPY", LAUNCHER.read_text(), flags=DOTALL)
    assert len(blocks) >= 9
    for index, block in enumerate(blocks):
        compile(block, f"{LAUNCHER.name}:heredoc-{index}", "exec")


def test_pace_h200_latent_embedded_argv_unpack_targets_are_unique() -> None:
    blocks = findall(r"<<'PY'\n(.*?)\nPY", LAUNCHER.read_text(), flags=DOTALL)
    for block_index, block in enumerate(blocks):
        tree = ast.parse(block)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Tuple):
                continue
            target_names = [
                item.id for item in node.targets[0].elts if isinstance(item, ast.Name)
            ]
            if target_names:
                assert len(target_names) == len(set(target_names)), (
                    block_index,
                    target_names,
                )


def test_pace_h200_latent_launcher_is_fail_closed() -> None:
    launcher = LAUNCHER.read_text()

    for contract in (
        "MODE=${MODE:?",
        "EXPECTED_HEAD=${EXPECTED_HEAD:?",
        "EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?",
        "SMOKE_GATE=${SMOKE_GATE:-}",
        "EXPECTED_SMOKE_GATE_SHA=${EXPECTED_SMOKE_GATE_SHA:-}",
        "EXPECTED_ENV_SHA=512f34f6",
        "EXPECTED_NORM_SHA=9af0eb68",
        "EXPECTED_NORM_VALIDATION_SHA=5a250b04",
        "EXPECTED_TRANSPORT_VALIDATION_SHA=c0a73f5c",
        "EXPECTED_WORLD_SIZE=1",
        "EXPECTED_GPU_MODEL=H200",
        "EXPECTED_PARTITION=gpu-h200",
        "EXPECTED_ACCOUNT=gts-dxu345-rl2",
        "EXPECTED_QOS=inferno",
    ):
        assert contract in launcher

    assert 'case "$MODE" in' in launcher
    assert "smoke)" in launcher
    assert "full)" in launcher
    assert "pusht/pipeline_sampler_usocket_chain_newdata_dense_medium_h16" in launcher
    assert "pipeline_diffusion" not in launcher
    assert (
        "ft_cotrain_newdata3719_latent_dense_m96_h16_s42_"
        "world1_pace_h200_20260828" in launcher
    )
    assert "world2_l40s" not in launcher
    assert "flow_transfer_newdata3719_cotrain_h16_20260828" in launcher
    assert 'test "${SLURM_JOB_PARTITION:?}" = "$EXPECTED_PARTITION"' in launcher
    assert 'test "${SLURM_JOB_ACCOUNT:?}" = "$EXPECTED_ACCOUNT"' in launcher
    assert 'grep -q " QOS=$EXPECTED_QOS " <<< "$JOB_RECORD"' in launcher
    assert 'fields["TimeLimit"] == "2-00:00:00"' in launcher
    assert 'fields["Requeue"] == "1"' in launcher
    assert "SubmitLine=(.*?)(?=\\sWorkDir=)" in launcher
    assert '"--signal=USR1@300" in submit_tokens' in launcher
    assert 'submit_tokens[index : index + 2] == ["--signal", "USR1@300"]' in launcher
    assert "launch_params.gpus_per_node=1" in launcher
    assert launcher.count("batch_size=64") == 2
    assert "trainer.accumulate_grad_batches=1" in launcher
    assert "trainer.log_every_n_steps=1" in launcher
    assert "trainer.max_steps=2" in launcher
    assert "trainer.limit_train_batches=2" in launcher
    assert "trainer.val_check_interval=1" in launcher
    assert "trainer.limit_val_batches=1" in launcher
    assert "trainer.max_steps=240000" in launcher
    assert "trainer.limit_val_batches=0" in launcher
    assert "callbacks.model_checkpoint.every_n_train_steps=2" in launcher
    assert "callbacks.terminal_checkpoint.every_n_train_steps=1" in launcher
    assert "callbacks.terminal_checkpoint.every_n_train_steps=240000" in launcher
    assert "callback_state_keys" in launcher
    assert "len(set(callback_state_keys))" in launcher
    assert "WANDB_MODE=offline" in launcher
    assert "WANDB_MODE=online" in launcher
    assert "WANDB_RESUME=never" in launcher
    assert "WANDB_RESUME=allow" in launcher
    assert "read_successful_wandb_exit_code" in launcher
    assert 'run_dir.glob(f"wandb/run-*/run-{wandb_id}.wandb")' in launcher
    assert 'run_dir.glob("wandb/offline-run-*")' in launcher
    assert '"wandb_exit_code": wandb_exit_code' in launcher
    assert "gradient_clip_val" in launcher
    assert "cfg.model.optimizer.lr == 3.0e-5" in launcher
    assert "cfg.model.scheduler.eta_min == 3.0e-6" in launcher
    assert "cfg.model.scheduler.warmup_steps == 3_000" in launcher
    assert "EXCLUDED_CHAIN_EPISODE=episode_T_chain_gripper_obs7_000050" in launcher
    assert "EXCLUDED_CHAIN_FRAMES=3118" in launcher
    assert "EXPECTED_CHAIN_GEN_COUNT=720" in launcher
    assert "EXPECTED_CHAIN_GEN_EFFECTIVE_COUNT=719" in launcher
    assert "EXPECTED_CHAIN_TRAIN_FRAMES=1237652" in launcher
    assert "set(level_counts.values()) == {24}" in launcher
    assert "effective_gen_names == expected_effective_names" in launcher
    assert 'payload["phase"] == "published"' in launcher
    assert '"regular_files": 52090' in launcher
    assert '"slurm_job_id": 3724901' in launcher
    assert '"terminal_marker": "TRANSFER_PASS"' in launcher
    assert "transport_validation_sha256" in launcher
    assert launcher.count("validate_all_inventories") >= 3
    assert 'episode / "zarr.json"' in launcher
    assert "metadata_path.read_bytes() == metadata_bytes" in launcher
    assert 'test "$(stat -c \'%a\' "$SMOKE_GATE")" = 444' in launcher
    assert 'payload["status"] == "PASS"' in launcher
    assert 'payload["smoke_result"]["dense_training_steps"] == [0, 1]' in launcher
    assert 'chmod 444 "$PROVENANCE_DIR/run_validation.json"' in launcher
    assert '--expected-world-size "$EXPECTED_WORLD_SIZE"' in launcher
    assert launcher.count("--required-embodiments 19,20") == 2
    assert 'test -z "$(git -C "$REPO" status --porcelain=v1' in launcher
    assert '"$SLURM_BIN/sbatch"' not in launcher


@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_pace_h200_latent_resolved_contract(mode: str) -> None:
    cfg = _compose(mode)
    model = cfg.model.robomimic_model
    domains = {"pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"}

    assert cfg.launch_params.gpus_per_node == 1
    assert cfg.launch_params.nodes == 1
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.trainer.log_every_n_steps == 1
    assert cfg.trainer.get("gradient_clip_val") is None
    assert cfg.model.optimizer.lr == pytest.approx(3.0e-5)
    assert cfg.model.scheduler.eta_min == pytest.approx(3.0e-6)
    assert cfg.model.scheduler.warmup_steps == 3_000
    assert cfg.model.scheduler.max_steps == 240_000
    assert set(model.domains) == domains
    assert model.action_horizon == 16
    assert model.stages[1].action_horizon == 16
    assert model.stages[1].latent_dim == 96
    assert model.stages[2].action_horizon == 16
    assert model.stages[2].latent_dim == 96
    assert model.stages[2].denoising_module.act_seq == 16
    assert model.stages[2].action_dims.pushshapes_sim_u_socket == 4
    assert model.stages[2].action_dims.pushshapes_sim_chain_gripper == 6
    assert "action_encoder" not in OmegaConf.to_yaml(cfg.model).lower()
    for domain in domains:
        assert cfg.data.train_dataloader_params[domain].batch_size == 64
        assert cfg.data.valid_dataloader_params[domain].batch_size == 16
    assert (
        str(cfg.data.train_datasets.pushshapes_sim_u_socket.resolver.folder_path)
        == U_DATA
    )
    for split in ("train_datasets", "valid_datasets"):
        chain = cfg.data[split].pushshapes_sim_chain_gripper
        assert list(chain.resolver.folder_paths) == [CHAIN_BASE, CHAIN_GEN]
        assert chain.resolver.key_map.action_horizon == 16
        assert list(chain.filters.filter_lambdas) == [
            "lambda row: row.get('episode_hash') != "
            "'episode_T_chain_gripper_obs7_000050'"
        ]

    checkpoint_callbacks = {
        name: callback
        for name, callback in cfg.callbacks.items()
        if str(callback.get("_target_", "")).endswith("ModelCheckpoint")
    }
    assert set(checkpoint_callbacks) == {
        "model_checkpoint",
        "terminal_checkpoint",
    }
    state_key_inputs = [
        (
            callback.get("monitor"),
            callback.get("mode"),
            callback.get("every_n_train_steps"),
            callback.get("every_n_epochs"),
            repr(callback.get("train_time_interval")),
        )
        for callback in checkpoint_callbacks.values()
    ]
    assert len({repr(item) for item in state_key_inputs}) == len(state_key_inputs)

    if mode == "smoke":
        assert cfg.trainer.max_steps == 2
        assert cfg.trainer.limit_train_batches == 2
        assert cfg.trainer.val_check_interval == 1
        assert cfg.trainer.limit_val_batches == 1
        assert cfg.callbacks.model_checkpoint.every_n_train_steps == 2
        assert cfg.callbacks.model_checkpoint.train_time_interval is None
        assert cfg.callbacks.terminal_checkpoint.every_n_train_steps == 1
    else:
        assert cfg.trainer.max_steps == 240_000
        assert cfg.trainer.limit_train_batches == 1.0
        assert cfg.trainer.val_check_interval == 10_000
        assert cfg.trainer.limit_val_batches == 0
        assert cfg.callbacks.model_checkpoint.every_n_train_steps is None
        assert cfg.callbacks.model_checkpoint.train_time_interval.hours == 1
        assert cfg.callbacks.terminal_checkpoint.every_n_train_steps == 240_000


def test_pace_h200_inventory_summary_embedded_block_executes(tmp_path: Path) -> None:
    metadata_sha = "0" * 64
    u_metadata = tmp_path / "u.tsv"
    base_metadata = tmp_path / "base.tsv"
    gen_metadata = tmp_path / "gen.tsv"
    effective_inventory = tmp_path / "effective.txt"
    u_metadata.write_text(f"u.zarr\t5\t{metadata_sha}\n")
    base_metadata.write_text(f"base.zarr\t7\t{metadata_sha}\n")

    excluded = "episode_T_chain_gripper_obs7_000050.zarr"
    names = []
    for level in range(1, 31):
        for index in range(24):
            if level == 7 and index == 0:
                name = excluded
            else:
                name = f"episode_T_chain_gripper_obs{level}_{level * 1000 + index:06d}.zarr"
            names.append(name)
    assert len(names) == 720
    frames = {name: 1 for name in names}
    frames[excluded] = 3
    gen_metadata.write_text(
        "".join(f"{name}\t{frames[name]}\t{metadata_sha}\n" for name in sorted(names))
    )
    effective_names = sorted(set(names) - {excluded})
    effective_inventory.write_text("".join(f"{name}\n" for name in effective_names))

    _run_embedded(
        _embedded_block("def frames_by_episode"),
        u_metadata,
        base_metadata,
        gen_metadata,
        effective_inventory,
        excluded,
        3,
        5,
        7,
        722,
        719,
        719,
        726,
    )


def test_pace_h200_slurm_signal_contract_executes_and_rejects_bad_full() -> None:
    block = _embedded_block("job_record, mode = sys.argv[1:]")
    full = (
        "JobId=123 Requeue=1 TimeLimit=2-00:00:00 "
        "SubmitLine=sbatch -A gts-dxu345-rl2 -q inferno -p gpu-h200 "
        "--requeue --signal=USR1@300 launcher.sbatch "
        "WorkDir=/storage/home/paphiwetsa3"
    )
    smoke = (
        "JobId=124 Requeue=0 TimeLimit=01:00:00 "
        "SubmitLine=sbatch -A gts-dxu345-rl2 -q inferno -p gpu-h200 "
        "launcher.sbatch WorkDir=/storage/home/paphiwetsa3"
    )
    _run_embedded(block, full, "full")
    _run_embedded(block, smoke, "smoke")

    for bad in (
        full.replace("Requeue=1", "Requeue=0"),
        full.replace(" --signal=USR1@300", ""),
        full.replace("--signal=USR1@300", "--signal B:USR1@300"),
        full.replace("TimeLimit=2-00:00:00", "TimeLimit=1-23:59:59"),
    ):
        with pytest.raises(CalledProcessError):
            _run_embedded(block, bad, "full")


@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_pace_h200_resolved_config_embedded_block_executes(
    mode: str, tmp_path: Path
) -> None:
    cfg = _compose(mode)
    OmegaConf.set_struct(cfg, False)
    cfg.mode = "train"
    cfg.paths.root_dir = RUN_DIR
    cfg.paths.output_dir = RUN_DIR
    cfg.paths.work_dir = str(ROOT)
    cfg.trainer.devices = 1
    cfg.trainer.num_nodes = 1
    cfg.model.train_metrics_on_step = True
    cfg.model.train_metrics_on_epoch = True
    cfg.logger.wandb.offline = mode == "smoke"
    cfg.logger.wandb.project = "pushshapes-flow-transfer"
    cfg.logger.wandb.entity = "rl2-group"
    cfg.logger.wandb.group = WANDB_GROUP
    cfg.logger.wandb.id = WANDB_ID
    cfg.logger.wandb.resume = "never"
    cfg.norm_stats.norm_mode = "minmax"
    cfg.norm_stats.reduce_all_but_last = True
    cfg.norm_stats.sample_frac = 1.0
    cfg.norm_stats.save_cache_dir = None
    cfg.norm_stats.precomputed_norm_path = NORM_ARTIFACT
    cfg.callbacks.model_checkpoint.dirpath = str(tmp_path / "checkpoints")
    cfg.callbacks.terminal_checkpoint.dirpath = str(tmp_path / "checkpoints/final")
    config_path = tmp_path / f"resolved_{mode}.yaml"
    OmegaConf.save(cfg, config_path)

    _run_embedded(
        _embedded_block("expected_chain_filter ="),
        config_path,
        mode,
        NORM_ARTIFACT,
        RUN_DIR,
        ROOT,
        U_DATA,
        CHAIN_BASE,
        CHAIN_GEN,
        "episode_T_chain_gripper_obs7_000050",
        WANDB_ID,
        WANDB_GROUP,
        "true" if mode == "smoke" else "false",
        "never",
    )


def test_pace_h200_smoke_gate_embedded_block_executes(tmp_path: Path) -> None:
    smoke_result_path = tmp_path / "SMOKE_RESULT.json"
    output_path = tmp_path / "run_validation.json"
    resolved_config = tmp_path / "resolved_config.yaml"
    smoke_result = {
        "status": "passed",
        "global_step": 2,
        "scheduler_last_epoch": 2,
        "world_size": 1,
        "required_embodiments": [19, 20],
        "dense_training_steps": [0, 1],
        "validation_trainer_global_step": 1,
    }
    smoke_result_path.write_text(json.dumps(smoke_result) + "\n")
    resolved_config.write_text("trainer:\n  max_steps: 2\n")
    hashes = [str(index) * 64 for index in range(1, 9)]

    _run_embedded(
        _embedded_block("smoke_result_arg,"),
        smoke_result_path,
        output_path,
        "a" * 40,
        hashes[0],
        hashes[1],
        hashes[2],
        hashes[3],
        hashes[4],
        hashes[5],
        hashes[6],
        hashes[7],
        "9" * 64,
        "a" * 64,
        "b" * 64,
        WANDB_ID,
        resolved_config,
    )
    payload = json.loads(output_path.read_text())
    assert payload["status"] == "PASS"
    assert payload["smoke_result"] == smoke_result
    assert payload["transport_validation_sha256"] == hashes[3]
    assert payload["episode_metadata_sha256"]["usocket"] == hashes[5]
    assert payload["resolved_config"] == str(resolved_config)


def test_pace_h200_full_completion_rejects_wrong_stable_wandb_id(
    tmp_path: Path,
) -> None:
    block = _embedded_block('terminal = run_dir / "checkpoints" / "final"')
    with pytest.raises(CalledProcessError):
        _run_embedded(
            block,
            tmp_path,
            tmp_path / "result.json",
            "a" * 40,
            "b" * 64,
            "wrong-run-id",
        )


def test_pace_h200_full_completion_rejects_missing_online_wandb_exit(
    tmp_path: Path,
) -> None:
    terminal = tmp_path / "checkpoints" / "final" / "step-240000.ckpt"
    terminal.parent.mkdir(parents=True)
    torch.save(
        {
            "global_step": 240_000,
            "state_dict": {"weight": torch.ones(1)},
            "optimizer_states": [{"param_groups": [{"lr": 3.0e-6}]}],
            "lr_schedulers": [{"last_epoch": 240_000}],
        },
        terminal,
    )
    block = _embedded_block('terminal = run_dir / "checkpoints" / "final"')
    with pytest.raises(CalledProcessError):
        _run_embedded(
            block,
            tmp_path,
            tmp_path / "result.json",
            "a" * 40,
            "b" * 64,
            WANDB_ID,
        )


@pytest.mark.parametrize("exit_code", [None, 0, 1])
def test_wandb_terminal_exit_gate_requires_terminal_success(
    monkeypatch: pytest.MonkeyPatch, exit_code: int | None
) -> None:
    payloads = []
    if exit_code is not None:
        record = smoke_verifier.wandb_internal_pb2.Record()
        record.exit.exit_code = exit_code
        payloads.append(record.SerializeToString())
    payloads.append(None)

    class FakeDataStore:
        def __init__(self) -> None:
            self.payloads = list(payloads)
            self.closed = False

        def open_for_scan(self, path: str) -> None:
            assert path.endswith("terminal.wandb")

        def scan_data(self):
            return self.payloads.pop(0)

        def close(self) -> None:
            self.closed = True

    fake_store = FakeDataStore()
    monkeypatch.setattr(smoke_verifier, "DataStore", lambda: fake_store)
    stream = Path("/tmp/terminal.wandb")
    if exit_code == 0:
        assert smoke_verifier.read_successful_wandb_exit_code(stream) == 0
    else:
        with pytest.raises(AssertionError):
            smoke_verifier.read_successful_wandb_exit_code(stream)
    assert fake_store.closed


def test_training_smoke_verifier_registers_eval_resolver(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "launch_params:\n"
        "  gpus_per_node: 1\n"
        "  nodes: 1\n"
        "trainer:\n"
        "  devices: ${eval:'${launch_params.gpus_per_node} * "
        "${launch_params.nodes}'}\n"
    )
    OmegaConf.clear_resolver("eval")

    cfg = _load_training_config(config_path)

    assert int(cfg.trainer.devices) == 1
    assert OmegaConf.has_resolver("eval")
