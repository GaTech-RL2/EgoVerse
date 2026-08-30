import json
import math
from pathlib import Path

import pytest

from egomimic.scripts.flow_transfer_run_bundle import (
    _phase_args,
    _validate_norm_artifact,
)

DOMAINS = {
    "u": {
        "embodiment_id": 19,
        "state_agent_obj_shape": [2],
        "actions_shape": [3, 2],
    },
    "chain": {
        "embodiment_id": 20,
        "state_agent_obj_shape": [2],
        "actions_shape": [3, 2],
    },
}
REPO_ROOT = Path(__file__).parents[1]


def _stats(shape):
    size = math.prod(shape)
    flat = [float(index) for index in range(size)]

    def reshape(values, dims):
        if len(dims) == 1:
            return values
        stride = math.prod(dims[1:])
        return [
            reshape(values[index : index + stride], dims[1:])
            for index in range(0, len(values), stride)
        ]

    low = reshape(flat, shape)
    high = reshape([value + 1.0 for value in flat], shape)
    return {
        "mean": low,
        "std": high,
        "min": low,
        "max": high,
        "median": low,
        "quantile_1": low,
        "quantile_99": high,
        "quantile_0_01": low,
        "quantile_99_99": high,
    }


def _norm_payload():
    stats = {}
    metadata = {}
    for domain, contract in DOMAINS.items():
        embodiment = str(contract["embodiment_id"])
        stats[embodiment] = {
            "state_agent_obj": _stats(contract["state_agent_obj_shape"]),
            "actions": _stats(contract["actions_shape"]),
        }
        frames = 20 if domain == "u" else 40
        metadata[embodiment] = {
            "dataset_size": frames,
            "sampled_frames": math.ceil(0.05 * frames),
            "sample_frac": 0.05,
            "seed": 42,
            "max_samples": None,
        }
    return {
        "norm_mode": "quantile",
        "reduce_all_but_last": False,
        "frames": 3,
        "stats": stats,
        "norm_run_metadata": {
            "embodiments": metadata,
            "total_dataset_frames": 60,
            "total_sampled_frames": 3,
        },
    }


def test_norm_bundle_contract_validates_nested_action_stats(tmp_path: Path) -> None:
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps(_norm_payload()))

    result = _validate_norm_artifact(
        path,
        {
            "norm_mode": "quantile",
            "reduce_all_but_last": False,
            "sample_frac": 0.05,
            "domains": DOMAINS,
        },
        {
            "u": {"dataset_frames": 20, "sampled_frames": 1},
            "chain": {"dataset_frames": 40, "sampled_frames": 2},
        },
        42,
    )

    assert result == {
        "dataset_frames": 60,
        "sampled_frames": 3,
        "embodiments": ["19", "20"],
    }


def test_norm_bundle_contract_rejects_wrong_shape_and_nonfinite(tmp_path: Path) -> None:
    payload = _norm_payload()
    payload["stats"]["19"]["actions"]["mean"] = [[0.0, 1.0]]
    path = tmp_path / "wrong_shape.json"
    path.write_text(json.dumps(payload))
    contract = {
        "norm_mode": "quantile",
        "reduce_all_but_last": False,
        "sample_frac": 0.05,
        "domains": DOMAINS,
    }
    counts = {
        "u": {"dataset_frames": 20, "sampled_frames": 1},
        "chain": {"dataset_frames": 40, "sampled_frames": 2},
    }
    with pytest.raises(AssertionError):
        _validate_norm_artifact(path, contract, counts, 42)

    payload = _norm_payload()
    payload["stats"]["19"]["actions"]["mean"][0][0] = float("nan")
    path.write_text(json.dumps(payload))
    with pytest.raises(AssertionError):
        _validate_norm_artifact(path, contract, counts, 42)


def test_phase_args_bind_norm_and_step_logging_identity(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    run = tmp_path / "smoke"
    norm = tmp_path / "norm.json"
    args = _phase_args(
        repo=repo,
        experiment="pusht/example_smoke",
        mode="train",
        run_dir=run,
        norm_cache_dir=None,
        norm_artifact=norm,
        world_size=2,
        wandb={
            "project": "project",
            "entity": "rl2-group",
            "group": "group",
            "id": "run-id",
        },
        offline=True,
    )

    assert "launch_params.gpus_per_node=2" in args
    assert f"norm_stats.precomputed_norm_path={norm}" in args
    assert "norm_stats.save_cache_dir=null" in args
    assert "logger.wandb.entity=rl2-group" in args
    assert "logger.wandb.id=run-id" in args
    assert "++logger.wandb.resume=never" in args

    full_args = _phase_args(
        repo=repo,
        experiment="pusht/example",
        mode="train",
        run_dir=tmp_path / "full",
        norm_cache_dir=None,
        norm_artifact=norm,
        world_size=2,
        wandb={
            "project": "project",
            "entity": "rl2-group",
            "group": "group",
            "id": "full-id",
        },
        offline=False,
    )
    assert "logger.wandb.offline=false" in full_args
    assert "++logger.wandb.resume=never" in full_args
    assert "++logger.wandb.resume=allow" not in full_args


def test_launcher_hides_cuda_for_strict_checkpoint_verification() -> None:
    launcher = (REPO_ROOT / "scripts/train/flow_transfer_run_bundle.sbatch").read_text()

    assert launcher.count('CUDA_VISIBLE_DEVICES= "$PYTHON" "$SMOKE_VERIFIER"') == 2
    assert launcher.count('CUDA_VISIBLE_DEVICES= "$PYTHON" "$FULL_VERIFIER"') == 2
