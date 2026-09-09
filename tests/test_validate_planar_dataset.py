import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import zarr

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "scripts" / "ice" / "validate_planar_dataset.py"
MANIFEST_PATH = (
    ROOT
    / "egomimic"
    / "hydra_configs"
    / "data"
    / "pusht"
    / "planar_v2_usocket_dp_3k_split_seed42_v1.json"
)
SPEC = importlib.util.spec_from_file_location("validate_planar_dataset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _manifest(names, source_root):
    train, valid = MODULE.split_names(names)
    source_paths = {name: source_root / name for name in names}
    train_paths = {source_paths[name] for name in train}
    valid_paths = {source_paths[name] for name in valid}
    return {
        "schema_version": 1,
        "status": "PASS",
        "split_algorithm": MODULE.SPLIT_ALGORITHM,
        "split_seed": 42,
        "valid_ratio": 0.01,
        "generator_sha256": "a" * 64,
        "cross_domain_train_valid_resolved_path_overlap_count": 0,
        "domains": {
            MODULE.DOMAIN: {
                "folder_path": str(source_root),
                "total_count": len(names),
                "inventory_names_sha256": MODULE.names_sha256(names),
                "train_count": len(train),
                "train_names_sha256": MODULE.names_sha256(train),
                "train_resolved_paths_sha256": MODULE.paths_sha256(train_paths),
                "train_ids": sorted(train),
                "valid_count": len(valid),
                "valid_names_sha256": MODULE.names_sha256(valid),
                "valid_resolved_paths_sha256": MODULE.paths_sha256(valid_paths),
                "valid_ids": sorted(valid),
                "id_overlap_count": 0,
                "resolved_path_overlap_count": 0,
                "union_count": len(names),
                "union_matches_inventory": True,
            }
        },
    }


def test_committed_manifest_has_exact_hash_and_contract():
    expected = "3683e3461596eef8df2432fa865779b3c77b2a2057dabd0fea125595729cf313"
    assert hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest() == expected

    result = MODULE.validate_manifest_structure(json.loads(MANIFEST_PATH.read_text()))

    assert result["total_count"] == 2999
    assert result["train_count"] == 2970
    assert result["valid_count"] == 29
    assert result["id_overlap_count"] == 0
    assert result["union_matches_inventory"] is True


def test_portable_inventory_reproduces_split_and_has_no_path_overlap(tmp_path):
    names = {f"episode_{index:03d}" for index in range(100)}
    source_root = Path("/original/dataset")
    manifest = _manifest(names, source_root)
    dataset_root = tmp_path / "portable-dataset"
    dataset_root.mkdir()
    entries = []
    for name in sorted(names):
        path = dataset_root / f"{name}.zarr"
        path.mkdir()
        entries.append((path, name))

    result = MODULE.validate_physical_inventory(manifest, entries, dataset_root)

    assert result["total_count"] == 100
    assert result["train_count"] == 99
    assert result["valid_count"] == 1
    assert result["resolved_path_overlap_count"] == 0
    assert result["union_count"] == 100
    assert result["union_matches_inventory"] is True


def test_inventory_mismatch_fails_closed(tmp_path):
    names = {f"episode_{index:03d}" for index in range(10)}
    manifest = _manifest(names, Path("/original/dataset"))
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    entries = []
    for name in sorted(names - {"episode_009"}):
        path = dataset_root / name
        path.mkdir()
        entries.append((path, name))

    with pytest.raises(RuntimeError, match="physical episode inventory mismatch"):
        MODULE.validate_physical_inventory(manifest, entries, dataset_root)


def test_manifest_split_tampering_fails_closed():
    names = {f"episode_{index:03d}" for index in range(100)}
    manifest = _manifest(names, Path("/original/dataset"))
    tampered = copy.deepcopy(manifest)
    domain = tampered["domains"][MODULE.DOMAIN]
    domain["train_ids"][0], domain["valid_ids"][0] = (
        domain["valid_ids"][0],
        domain["train_ids"][0],
    )
    domain["train_ids"].sort()
    domain["valid_ids"].sort()
    domain["train_names_sha256"] = MODULE.names_sha256(domain["train_ids"])
    domain["valid_names_sha256"] = MODULE.names_sha256(domain["valid_ids"])
    domain["inventory_names_sha256"] = MODULE.names_sha256(
        domain["train_ids"] + domain["valid_ids"]
    )

    with pytest.raises(RuntimeError, match="do not reproduce"):
        MODULE.validate_manifest_structure(tampered)


def test_two_ids_resolving_to_one_physical_directory_fail(tmp_path):
    names = {"episode_a", "episode_b"}
    manifest = _manifest(names, Path("/original/dataset"))
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    first = dataset_root / "episode_a"
    first.mkdir()
    second = dataset_root / "episode_b"
    second.symlink_to(first, target_is_directory=True)

    with pytest.raises(RuntimeError, match="same physical path"):
        MODULE.validate_physical_inventory(
            manifest,
            [(first, "episode_a"), (second, "episode_b")],
            dataset_root,
        )


def test_explicit_deleted_directory_is_excluded_but_accounted(tmp_path):
    names = {"episode_a", "episode_b"}
    manifest = _manifest(names, Path("/original/dataset"))
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    entries = []
    for name in sorted(names):
        path = dataset_root / name
        path.mkdir()
        entries.append((path, name))
    deleted = dataset_root / "deleted_episode"
    deleted.mkdir()

    result = MODULE.validate_physical_inventory(
        manifest,
        entries,
        dataset_root,
        excluded_deleted_paths=[deleted],
    )

    assert result["excluded_deleted_directory_count"] == 1


def test_cli_reuses_training_resolver_and_allows_only_explicitly_deleted_store(
    tmp_path,
):
    names = {f"episode_{index:03d}" for index in range(10)}
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    for name in names:
        zarr.open_group(str(dataset_root / f"{name}.zarr"), mode="w")
    deleted = zarr.open_group(str(dataset_root / "deleted.zarr"), mode="w")
    deleted.attrs["is_deleted"] = True

    manifest = _manifest(names, Path("/original/dataset"))
    generator = ROOT / "scripts" / "data" / "materialize_local_episode_split.py"
    manifest["generator_sha256"] = hashlib.sha256(generator.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    expected_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    output = tmp_path / "validation.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--repo",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--manifest",
            str(manifest_path),
            "--expected-manifest-sha256",
            expected_sha,
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    result = json.loads(output.read_text())
    assert result["status"] == "DATASET_VALIDATED"
    assert result["physical_inventory"]["total_count"] == 10
    assert result["physical_inventory"]["excluded_deleted_directory_count"] == 1
    assert json.loads(completed.stdout.splitlines()[-1]) == result
