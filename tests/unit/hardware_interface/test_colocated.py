"""Co-located execution preserves native isolation, credentials and trial records."""

import json
from pathlib import Path

import pytest
import yaml

from experiments.threading_hardware_interface import prepare as package
from experiments.threading_hardware_interface.colocated import (
    NATIVE_IMAGE,
    child_environments,
    mirror,
)


def test_native_runtime_is_the_original_pinned_image():
    root = Path(__file__).resolve().parents[3]
    lock = root / "experiments/libero_hardware_interface/locks/container.json"
    assert NATIVE_IMAGE == json.loads(lock.read_text())["image"]


def test_native_runtime_and_model_environment_are_separate(tmp_path):
    server, native = child_environments(
        {
            "PATH": "/usr/local/bin:/usr/bin",
            "R2_SECRET_ACCESS_KEY": "storage-fixture",
            "R2_ENDPOINT_URL": "https://storage.invalid",
            "OPENAI_API_KEY": "hosted-fixture",
            "HARDWARE_API_KEY_FILE": "/private/key",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "LD_LIBRARY_PATH": "/cuda/host",
            "PYTHONPATH": "/host/packages",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "HARDWARE_WORKFLOW": "fixture",
        },
        tmp_path,
    )
    assert not any(key.startswith(("R2_", "OPENAI_")) for key in server)
    assert "HARDWARE_API_KEY_FILE" not in server
    assert server["LD_LIBRARY_PATH"] == "/cuda/host"
    assert server["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert native["R2_SECRET_ACCESS_KEY"] == "storage-fixture"
    assert native["HARDWARE_API_KEY_FILE"] == "/private/key"
    assert not any(
        key in native
        for key in ("LD_LIBRARY_PATH", "PYTHONPATH", "VLLM_USE_V2_MODEL_RUNNER")
    )
    assert native["MUJOCO_GL"] == native["PYOPENGL_PLATFORM"] == "osmesa"
    assert native["CUDA_VISIBLE_DEVICES"] == ""
    assert native["HARDWARE_BOOTSTRAP_FILE_LOG"] == "1"


def test_persistent_mirror_preserves_updates_and_excludes_scratch(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "preserved"
    source.mkdir()
    (source / "events.jsonl").write_text("first\n")
    (source / "scratch-runtime").mkdir()
    (source / "scratch-runtime/private").write_text("not an artifact")
    seen = {}
    mirror(source, destination, seen)
    assert (destination / "events.jsonl").read_text() == "first\n"
    assert not (destination / "scratch-runtime").exists()
    (source / "events.jsonl").write_text("first\nsecond\n")
    mirror(source, destination, seen)
    assert (destination / "events.jsonl").read_text() == "first\nsecond\n"
    (source / "inference-api-key").write_text("credential fixture")
    with pytest.raises(ValueError, match="private_credential_in_artifacts"):
        mirror(source, destination, seen)
    assert not (destination / "inference-api-key").exists()


@pytest.mark.parametrize("native_only", [True, False])
def test_colocated_workflow_uses_eight_gpus_total_and_preserves_both_models(
    tmp_path, monkeypatch, native_only
):
    def base(destination):
        destination.mkdir()
        (destination / "workflow.yaml").write_text(
            yaml.safe_dump(
                {
                    "workflow": {
                        "groups": [
                            {
                                "tasks": [
                                    {"environment": {"VLLM_USE_V2_MODEL_RUNNER": "1"}},
                                    {
                                        "environment": {
                                            "HARDWARE_SOURCE_COMMIT": "fixture",
                                            "HARDWARE_MANIFEST": "old.yaml",
                                        },
                                        "credentials": {"r2-fixture": {}},
                                    },
                                ]
                            }
                        ]
                    }
                }
            )
        )
        return {"source_commit": "fixture"}

    monkeypatch.setattr(package, "prepare_group", base)
    receipt = package.prepare_colocated(tmp_path / "package", native_only=native_only)
    spec = yaml.safe_load((tmp_path / "package/workflow.yaml").read_text())["workflow"]
    assert "groups" not in spec and len(spec["tasks"]) == 1
    assert sum(value["gpu"] for value in spec["resources"].values()) == 8
    task = spec["tasks"][0]
    assert task["image"] == package.VLLM_IMAGE
    assert task["exitActions"] == {"COMPLETE": "0", "FAIL": "1-9999"}
    assert task["environment"]["HARDWARE_STAGE"] == (
        "commission" if native_only else "pilot"
    )
    assert "HARDWARE_MANIFEST" not in task["environment"]
    assert (
        "astra-hardware-inference-20261010" in task["credentials"]
    ) is not native_only
    assert receipt["planned_trials"] == (0 if native_only else 40)
    assert receipt["evaluator_gpus"] == 0 and receipt["allocated_gpus"] == 8
    assert receipt["conditions"] == [
        "astra_CAP",
        "astra_direct",
        "deepseek_CAP",
        "deepseek_direct",
    ]
