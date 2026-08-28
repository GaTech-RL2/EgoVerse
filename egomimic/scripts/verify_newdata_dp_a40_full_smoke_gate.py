"""Fail-closed semantic gate for the new-data DP world-size-2 A40 smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

import egomimic.utils.hydra_resolvers  # noqa: F401 -- project config resolvers

EXPECTED_CHAIN_FILTER = (
    "lambda row: row.get('episode_hash') != "
    "'episode_T_chain_gripper_obs7_000050'"
)
EXPECTED_DOMAINS = {
    "pushshapes_sim_u_socket",
    "pushshapes_sim_chain_gripper",
}
REQUIRED_STEP_METRICS = {
    "train_metrics": {"Train/Loss"},
    "timing_metrics": {
        "Timing/Process_Batch_Sec",
        "Timing/Forward_Pass_Sec",
        "Timing/Compute_Losses_Sec",
    },
    "optimizer_metrics": {"Optimizer/param_group_0_lr"},
}


def _register_training_config_resolvers() -> None:
    """Mirror the resolvers registered by ``egomimic.trainHydra``."""

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(path: Path, expected: str) -> None:
    assert len(expected) == 64
    int(expected, 16)
    assert path.is_file() and path.stat().st_size > 0, path
    assert sha256(path) == expected, path


def _finite_metric_map(metrics: Any) -> dict[str, float]:
    assert isinstance(metrics, dict) and metrics
    output: dict[str, float] = {}
    for key, raw_value in metrics.items():
        assert isinstance(key, str) and not key.endswith("_epoch"), key
        value = float(raw_value)
        assert math.isfinite(value), (key, raw_value)
        output[key] = value
    return output


def _verify_dense_per_step_history(payload: dict[str, Any]) -> list[int]:
    history = payload["training_history"]
    assert isinstance(history, list) and len(history) == 2, history
    steps = [int(row["trainer_global_step"]) for row in history]
    assert steps == [0, 1], steps
    for row in history:
        for category, required in REQUIRED_STEP_METRICS.items():
            metrics = _finite_metric_map(row[category])
            assert required.issubset(metrics), (category, required, metrics)
    assert payload["dense_training_steps"] == [0, 1]
    return steps


def _verify_real_validation(payload: dict[str, Any]) -> tuple[int, list[str]]:
    selected_step = int(payload["validation_trainer_global_step"])
    assert selected_step >= 1
    metrics = _finite_metric_map(payload["validation_metrics"])
    for embodiment in (19, 20):
        prefix = f"Valid/emb{embodiment}_"
        assert any(
            key.startswith(prefix) and key.endswith("_action_mse")
            for key in metrics
        ), (embodiment, metrics)
    history = payload["validation_history"]
    assert isinstance(history, list) and history
    matching = [
        row
        for row in history
        if int(row["trainer_global_step"]) == selected_step
    ]
    assert matching, (selected_step, history)
    selected_history_metrics = _finite_metric_map(
        matching[-1]["validation_metrics"]
    )
    assert selected_history_metrics == metrics
    return selected_step, sorted(metrics)


def _verify_resolved_config(
    config_path: Path,
    arm: str,
    output_dir: Path,
    job_id: str,
    norm_artifact: Path,
) -> None:
    _register_training_config_resolvers()
    cfg = OmegaConf.load(config_path)
    model = cfg.model.robomimic_model

    assert cfg.seed == 42
    assert cfg.mode == "train"
    assert Path(str(cfg.paths.root_dir)).resolve() == output_dir.resolve()
    assert Path(str(cfg.paths.output_dir)).resolve() == output_dir.resolve()
    assert cfg.trainer.precision == "bf16"
    assert cfg.trainer.strategy == "ddp"
    assert cfg.trainer.devices == 2
    assert cfg.trainer.num_nodes == 1
    assert cfg.trainer.sync_batchnorm is True
    assert cfg.trainer.max_steps == 2
    assert cfg.trainer.limit_train_batches == 2
    assert cfg.trainer.val_check_interval == 1
    assert cfg.trainer.limit_val_batches == 1
    assert cfg.trainer.num_sanity_val_steps == 0
    assert cfg.trainer.log_every_n_steps == 1
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.trainer.get("gradient_clip_val") is None
    assert cfg.launch_params.gpus_per_node == 2
    assert cfg.launch_params.nodes == 1
    assert cfg.logger.wandb.offline is True
    assert cfg.logger.wandb.project == "pushshapes-flow-transfer"
    assert cfg.logger.wandb.entity == "rl2-group"
    assert cfg.logger.wandb.group == "flow_transfer_newdata3719_cotrain_h16_20260828"
    expected_wandb_id = (
        f"ft_cotrain_newdata3719_dp_h16_s42_world2_a40_smoke_job_{job_id}"
    )
    assert cfg.logger.wandb.id == expected_wandb_id
    assert cfg.logger.wandb.name == expected_wandb_id
    assert cfg.logger.wandb.resume == "never"
    assert cfg.model.enable_grad_norm is False
    assert cfg.model.train_metrics_on_step is True
    assert cfg.model.optimizer.lr == 3.0e-5
    assert cfg.model.scheduler.max_steps == 240_000
    assert cfg.model.scheduler.warmup_steps == 3_000
    assert cfg.model.scheduler.warmup_start_factor == 0.1
    assert cfg.model.scheduler.eta_min == 3.0e-6
    assert cfg.model.get("scheduler_interval", "step") == "step"
    assert cfg.model.get("scheduler_frequency", 1) == 1
    assert cfg.norm_stats.norm_mode == "minmax"
    assert cfg.norm_stats.reduce_all_but_last is True
    assert cfg.norm_stats.sample_frac == 1.0
    assert cfg.norm_stats.save_cache_dir is None
    assert Path(str(cfg.norm_stats.precomputed_norm_path)).resolve() == norm_artifact.resolve()
    assert cfg.evaluator._target_.endswith("HumanRobotOverlayEval")
    assert model.action_horizon == 16
    assert set(model.domains) == EXPECTED_DOMAINS
    assert set(cfg.data.train_datasets) == EXPECTED_DOMAINS
    assert set(cfg.data.valid_datasets) == EXPECTED_DOMAINS
    for domain in EXPECTED_DOMAINS:
        assert cfg.data.train_dataloader_params[domain].batch_size == 32
        assert cfg.data.valid_dataloader_params[domain].batch_size == 16
    for split in ("train_datasets", "valid_datasets"):
        chain = cfg.data[split].pushshapes_sim_chain_gripper
        assert chain.filters._target_ == "egomimic.rldb.filters.DatasetFilter"
        assert list(chain.filters.filter_lambdas) == [EXPECTED_CHAIN_FILTER]
    chain_resolver = cfg.data.train_datasets.pushshapes_sim_chain_gripper.resolver
    assert list(chain_resolver.folder_paths) == [
        "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2",
        "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_gen",
    ]

    assert arm == "dp"
    stage = model.stages[1]
    assert stage.action_horizon == 16
    assert set(stage.policies) == EXPECTED_DOMAINS
    assert all(
        policy.noise_scheduler.prediction_type == "epsilon"
        for policy in stage.policies.values()
    )


def verify_smoke_result(
    result_path: Path,
    expected_result_sha: str,
    arm: str,
    expected_head: str,
    norm_artifact: Path,
    expected_norm_sha: str,
    smoke_root: Path,
    expected_smoke_launcher_sha: str,
) -> dict[str, Any]:
    require_sha(result_path, expected_result_sha)
    require_sha(norm_artifact, expected_norm_sha)
    result_path = result_path.resolve()
    output_dir = result_path.parent
    assert result_path.name == "SMOKE_RESULT.json"
    assert output_dir.name.startswith("job_")
    job_id = output_dir.name.removeprefix("job_")
    assert job_id.isdigit(), job_id
    expected_output = smoke_root / "smokes" / "dp_a40" / f"job_{job_id}"
    assert output_dir == expected_output.resolve(), (output_dir, expected_output)

    payload = json.loads(result_path.read_text())
    assert payload["status"] == "passed"
    assert payload["repo_head"] == expected_head
    assert Path(payload["output"]).resolve() == output_dir
    assert payload["global_step"] == 2
    assert payload["precision"] == "bf16"
    assert payload["world_size"] == 2
    assert payload["required_embodiments"] == [19, 20]
    assert payload["wandb_exit_code"] == 0
    assert payload["optimizer_state_count"] >= 1
    assert payload["scheduler_last_epoch"] == 2
    optimizer_lrs = [float(value) for value in payload["optimizer_lrs"]]
    assert optimizer_lrs and all(math.isfinite(value) for value in optimizer_lrs)

    config_path = Path(payload["config"])
    assert config_path.resolve() == (output_dir / ".hydra" / "config.yaml").resolve()
    require_sha(config_path, payload["config_sha256"])
    _verify_resolved_config(config_path, arm, output_dir, job_id, norm_artifact)

    checkpoint_path = Path(payload["checkpoint"])
    assert checkpoint_path.resolve() == (output_dir / "checkpoints" / "last.ckpt").resolve()
    require_sha(checkpoint_path, payload["checkpoint_sha256"])
    stream_path = Path(payload["wandb_stream"])
    assert output_dir in stream_path.resolve().parents
    assert stream_path.name.startswith("run-") and stream_path.suffix == ".wandb"
    require_sha(stream_path, payload["wandb_stream_sha256"])

    provenance_dir = smoke_root / "provenance" / "smoke" / "dp_a40" / f"job_{job_id}"
    provenance_norm = provenance_dir / "norm_stats.json"
    require_sha(provenance_norm, expected_norm_sha)
    require_sha(provenance_dir / "launcher.sbatch", expected_smoke_launcher_sha)
    gpu_lines = (provenance_dir / "gpu.txt").read_text().splitlines()
    assert len(gpu_lines) == 2, gpu_lines
    assert all("A40" in line.upper() for line in gpu_lines), gpu_lines
    slurm_job = (provenance_dir / "slurm_job.txt").read_text()
    assert "Partition=hoffman-lab" in slurm_job
    assert "Account=hoffman-lab" in slurm_job
    steps = _verify_dense_per_step_history(payload)
    validation_step, validation_metrics = _verify_real_validation(payload)
    return {
        "arm": "dp",
        "hardware": "2xa40",
        "job_id": job_id,
        "result": str(result_path),
        "result_sha256": expected_result_sha,
        "config_sha256": payload["config_sha256"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "wandb_stream_sha256": payload["wandb_stream_sha256"],
        "world_size": 2,
        "global_step": 2,
        "dense_training_steps": steps,
        "validation_trainer_global_step": validation_step,
        "validation_metrics": validation_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--norm-artifact", type=Path, required=True)
    parser.add_argument("--expected-norm-sha", required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--dp-result", type=Path, required=True)
    parser.add_argument("--expected-dp-result-sha", required=True)
    parser.add_argument("--expected-smoke-launcher-sha", required=True)
    args = parser.parse_args()

    assert len(args.expected_head) == 40
    int(args.expected_head, 16)
    dp = verify_smoke_result(
        args.dp_result,
        args.expected_dp_result_sha,
        "dp",
        args.expected_head,
        args.norm_artifact,
        args.expected_norm_sha,
        args.smoke_root,
        args.expected_smoke_launcher_sha,
    )
    record = {
        "status": "PASS",
        "repo_head": args.expected_head,
        "norm_artifact": str(args.norm_artifact.resolve()),
        "norm_sha256": args.expected_norm_sha,
        "world_size": 2,
        "max_steps": 2,
        "scheduled_real_validation": True,
        "dense_per_step_metrics": True,
        "hardware": "2xa40",
        "arm": dp,
    }
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
