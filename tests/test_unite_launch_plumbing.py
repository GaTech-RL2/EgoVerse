"""Static fail-closed checks for the canonical UNITE training plumbing."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from egomimic.utils.unite_readiness import (
    HANDOFF_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    SMOKE_BLOCKER,
    ReadinessTransitionError,
    verify_readiness_transition,
)

ROOT = Path(__file__).parents[1]
EXPERIMENT = (
    ROOT
    / "egomimic/hydra_configs/experiment/pusht"
    / "unite_usocket_register_sweep_val01_h16.yaml"
)
LAUNCHER = Path(
    "/coc/flash7/paphiwetsa3/scripts/train/"
    "flow_transfer_unite_skynet_x2_v20.sbatch"
)
SPLIT = Path(
    "/coc/flash7/paphiwetsa3/experiments/"
    "flow_transfer_dp_transformer_val01_l40sx2_20260831/"
    "source_artifacts/splits/seed42_val01_skynet.json"
)
NORMALIZATION = Path(
    "/coc/flash7/paphiwetsa3/experiments/"
    "flow_transfer_dp_transformer_val01_l40sx2_20260831/"
    "source_artifacts/norm/norm_stats.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repo), *arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _write_manifest(repo: Path, document: dict) -> None:
    (repo / MANIFEST_RELATIVE_PATH).write_text(
        yaml.safe_dump(document, sort_keys=False)
    )


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "--all")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _make_identity_repo(tmp_path: Path, *, ready: bool = False):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "readiness-test@example.invalid")
    _git(repo, "config", "user.name", "Readiness Test")
    document = {
        "schema_version": 2,
        "artifact_status": "LAUNCH_READY" if ready else "SMOKE_READY",
        "launch_status": "READY" if ready else "SMOKE_REQUIRED",
        "fixed_contract": {"latent_dim": 16, "num_latent_tokens": [4, 8]},
        "known_blockers": [] if ready else [SMOKE_BLOCKER],
    }
    _write_manifest(repo, document)
    (repo / HANDOFF_RELATIVE_PATH).write_text("pre-smoke\n")
    (repo / "model.py").write_text("LATENT_DIM = 16\n")
    head = _commit(repo, "base")
    return repo, document, head


def _commit_readiness(repo: Path, document: dict) -> str:
    document["artifact_status"] = "LAUNCH_READY"
    document["launch_status"] = "READY"
    document["known_blockers"] = []
    _write_manifest(repo, document)
    (repo / HANDOFF_RELATIVE_PATH).write_text("ready after four row smokes\n")
    return _commit(repo, "record smoke readiness")


def test_support_experiment_is_usocket_only_and_energy_score_k32():
    config = yaml.safe_load(EXPERIMENT.read_text())
    defaults = config["defaults"]
    assert {"override /data": "pusht/unite_usocket_val01_h16_per_emb_proprio"} in defaults
    assert not any("model" in entry for entry in defaults if isinstance(entry, dict))
    assert config["trainer"]["strategy"] == "ddp_find_unused_parameters_true"
    assert config["model"]["unite_flow_updates_per_reconstruction"] == 0
    energy = config["evaluator"]["energy_score"]
    assert energy["enabled"] is True and energy["sample_count"] == 32
    assert set(energy["action_dims"]) == {"pushshapes_sim_u_socket"}
    assert set(energy["distance_blocks"]) == {"pushshapes_sim_u_socket"}
    assert (
        energy["seed_bank_sha256"]
        == "88657b829905d4374823db145ded19b99cec4735f76694734473bcee068bb5b6"
    )
    precision = energy["provenance"]
    assert precision["model_autocast_precision"] == "bf16"
    assert precision["dopri5_state_precision"] == "fp32"
    assert precision["dopri5_derivative_precision"] == "fp32"
    assert precision["dopri5_error_control_precision"] == "fp32"
    assert config["run_provenance"]["unite_update_schedule"]["mode"] == (
        "joint_reconstruction_plus_flow"
    )


def test_launcher_uses_schema2_four_rows_and_direct_row_selection():
    if not LAUNCHER.is_file():
        pytest.skip("canonical Skynet launcher is external to the repository")
    source = LAUNCHER.read_text()
    assert "unite_usocket_register_sweep_manifest.yaml" in source
    assert "schema_version) == 2" in source
    assert "num_latent_tokens) in {4, 8}" in source
    assert "len(rows) == 4" in source
    assert "model=bf/$SWEEP_TASK_ID" in source
    assert "unite_usocket_register_sweep_val01_h16" in source
    assert "model.unite_flow_updates_per_reconstruction=0" in source
    assert "ddp_find_unused_parameters_true" in source
    assert '{"SMOKE_READY", "LAUNCH_READY"}' in source
    assert '{"SMOKE_REQUIRED", "READY"}' in source
    assert 'cfg.artifact_status == "LAUNCH_READY"' in source
    assert 'cfg.launch_status == "READY"' in source
    assert "row.parameter_manifest_path" in source
    assert "artifacts/unite_register_sweep_20260902/parameter_manifests/" in source
    assert "resolved-stage parameter manifest differs from durable manifest" in source
    assert 'cmp -s "$EXPECTED_PARAMETER_MANIFEST" "$PARAMETER_MANIFEST"' in source
    assert "expected_parameter_manifest.json" in source
    assert "content_projection_weights <= adamw_names" in source
    assert "action_projection_weights <= muon_names" in source
    assert "adamw_names.union(muon_names) == trainable_names" in source
    assert "EXPECTED_SMOKE_HEAD" in source
    assert "verify_readiness_transition" in source
    assert 'p["repo_head"] == sys.argv[11]' in source
    assert "smoke_source_commit.txt" in source
    assert "smoke_source_relation.txt" in source
    assert "readiness_changed_paths.txt" in source
    assert "obs33tok" not in source
    assert "CrossTransformer" not in source


def test_readiness_identity_accepts_exact_ready_head(tmp_path):
    repo, _, head = _make_identity_repo(tmp_path, ready=True)
    identity = verify_readiness_transition(repo, head, head)
    assert identity.smoke_source_head == identity.full_source_head == head
    assert identity.relation == "exact_head"
    assert identity.changed_paths == ()


def test_readiness_identity_accepts_only_documentation_transition(tmp_path):
    repo, document, smoke_head = _make_identity_repo(tmp_path)
    full_head = _commit_readiness(repo, document)
    identity = verify_readiness_transition(repo, smoke_head, full_head)
    assert identity.relation == "readiness_only_descendant"
    assert identity.changed_paths == (
        HANDOFF_RELATIVE_PATH,
        MANIFEST_RELATIVE_PATH,
    )


def test_readiness_identity_rejects_model_drift(tmp_path):
    repo, document, smoke_head = _make_identity_repo(tmp_path)
    document["artifact_status"] = "LAUNCH_READY"
    document["launch_status"] = "READY"
    document["known_blockers"] = []
    _write_manifest(repo, document)
    (repo / "model.py").write_text("LATENT_DIM = 32\n")
    full_head = _commit(repo, "unsafe model and readiness change")
    with pytest.raises(ReadinessTransitionError, match="non-readiness paths"):
        verify_readiness_transition(repo, smoke_head, full_head)


def test_readiness_identity_rejects_manifest_contract_drift(tmp_path):
    repo, document, smoke_head = _make_identity_repo(tmp_path)
    document["fixed_contract"]["latent_dim"] = 32
    full_head = _commit_readiness(repo, document)
    with pytest.raises(ReadinessTransitionError, match="manifest changed beyond"):
        verify_readiness_transition(repo, smoke_head, full_head)


def test_readiness_identity_rejects_dirty_worktree(tmp_path):
    repo, _, head = _make_identity_repo(tmp_path, ready=True)
    (repo / "model.py").write_text("LATENT_DIM = 32\n")
    with pytest.raises(ReadinessTransitionError, match="worktree is not clean"):
        verify_readiness_transition(repo, head, head)


def test_canonical_combined_artifacts_have_required_usocket_subsections():
    if not SPLIT.is_file() or not NORMALIZATION.is_file():
        pytest.skip("canonical Skynet data artifacts are not mounted")
    assert _sha256(SPLIT) == (
        "672f0f519bb7bff5b6b956d1b709abf1a1d387dd6b88b20a7c37536799bce0cd"
    )
    assert _sha256(NORMALIZATION) == (
        "3559aca1ac1279cbdd37de8e5b2da9bb350fbc0f1177d4c669aa590011fd0203"
    )
    split = json.loads(SPLIT.read_text())
    assert split["status"] == "PASS"
    assert split["seed"] == 42 and split["valid_ratio"] == 0.01
    assert split["counts"]["usocket_total"] == 2999
    assert split["counts"]["usocket_train"] == 2970
    assert split["counts"]["usocket_valid"] == 29
    assert split["overlap"]["usocket_ids"] == 0
    assert split["overlap"]["usocket_realpaths"] == 0
    normalization = json.loads(NORMALIZATION.read_text())
    assert normalization["norm_mode"] == "minmax"
    assert normalization["reduce_all_but_last"] is True
    assert normalization["norm_run_metadata"]["embodiments"]["19"] == {
        "dataset_size": 516610,
        "sampled_frames": 516610,
        "sample_frac": 1.0,
        "seed": 42,
        "train_episodes": 2970,
    }
    assert normalization["derivation"]["validation_episode_data_used"] is False
