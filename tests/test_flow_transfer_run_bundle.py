import json
import math
import os
import subprocess
import sys
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


def test_bundle_tool_bootstraps_repo_for_absolute_path_invocation(
    tmp_path: Path,
) -> None:
    tool = REPO_ROOT / "egomimic/scripts/flow_transfer_run_bundle.py"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(tool), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Materialize and verify immutable" in result.stdout


def test_shell_entry_points_prepend_repo_to_pythonpath() -> None:
    for relative_path in (
        "scripts/train/submit_flow_transfer_run_bundle.sh",
        "scripts/train/flow_transfer_run_bundle.sbatch",
    ):
        source = (REPO_ROOT / relative_path).read_text()
        assert "export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}" in source


def test_bundle_submitter_records_only_safe_explicit_node_exclusions() -> None:
    source = (
        REPO_ROOT / "scripts/train/submit_flow_transfer_run_bundle.sh"
    ).read_text()

    assert "[[ -v FLOW_TRANSFER_EXCLUDE_NODES ]]" in source
    assert "-z $FLOW_TRANSFER_EXCLUDE_NODES" in source
    assert "^[][[:alnum:]_.,-]+$" in source
    assert '"$SCONTROL" show hostnames "$FLOW_TRANSFER_EXCLUDE_NODES"' in source
    assert 'COMMON+=("--exclude=$FLOW_TRANSFER_EXCLUDE_NODES")' in source
    assert 'printf \'%q \' "${COMMAND[@]}"' in source


def test_bundle_submitter_exclusion_is_validated_and_recorded(
    tmp_path: Path,
) -> None:
    py_env = tmp_path / "env"
    slurm_bin = tmp_path / "slurm"
    output_root = tmp_path / "output"
    (py_env / "bin").mkdir(parents=True)
    slurm_bin.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")

    fake_python = py_env / "bin" / "python"
    fake_python.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ $* == *' print-field '* ]]; then
  field=${!#}
  case "$field" in
    outputs.root) printf '%s\\n' "$FAKE_OUTPUT_ROOT" ;;
    run_id) echo test-run ;;
    resources.account) echo rl2-lab ;;
    resources.partition) echo normal ;;
    resources.cpus_per_task) echo 2 ;;
    resources.memory) echo 4G ;;
    resources.world_size) echo 2 ;;
    resources.gpu_type) echo L40S ;;
    resources.smoke_qos) echo normal ;;
    resources.smoke_time) echo 00:10:00 ;;
    *) exit 3 ;;
  esac
fi
"""
    )
    fake_sbatch = slurm_bin / "sbatch"
    fake_sbatch.write_text("#!/bin/bash\nprintf '987654\\n'\n")
    fake_scontrol = slurm_bin / "scontrol"
    fake_scontrol.write_text(
        """#!/bin/bash
set -euo pipefail
[[ $1 == show && $2 == hostnames ]]
case "$3" in
  bishop) echo bishop ;;
  'node[01-02]') printf 'node01\\nnode02\\n' ;;
  *) exit 1 ;;
esac
"""
    )
    for path in (fake_python, fake_sbatch, fake_scontrol):
        path.chmod(0o755)

    submitter = REPO_ROOT / "scripts/train/submit_flow_transfer_run_bundle.sh"
    base_env = {
        **os.environ,
        "PY_ENV": str(py_env),
        "SLURM_BIN": str(slurm_bin),
        "FAKE_OUTPUT_ROOT": str(output_root),
    }
    valid = subprocess.run(
        [str(submitter), "norm", str(manifest)],
        env={**base_env, "FLOW_TRANSFER_EXCLUDE_NODES": "bishop"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert valid.returncode == 0, valid.stderr
    receipt = output_root / "submissions" / "norm_job_987654.txt"
    assert "--exclude=bishop" in receipt.read_text()

    for invalid in ("", "bishop;touch_bad", "node[02-01]"):
        result = subprocess.run(
            [str(submitter), "norm", str(manifest)],
            env={**base_env, "FLOW_TRANSFER_EXCLUDE_NODES": invalid},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2
