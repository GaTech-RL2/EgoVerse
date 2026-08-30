"""Materialize and verify immutable, manifest-driven Flow Transfer run bundles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

import egomimic.utils.hydra_resolvers  # noqa: F401 -- project config resolvers

SCHEMA_VERSION = 1

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    assert path.is_file() and path.stat().st_size > 0, path
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _require_clean_source(repo: Path, expected_head: str | None = None) -> str:
    repo = repo.resolve()
    assert repo.is_dir(), repo
    head = _git(repo, "rev-parse", "HEAD").strip()
    assert len(head) == 40 and all(c in "0123456789abcdef" for c in head), head
    if expected_head is not None:
        assert head == expected_head, (head, expected_head)
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    assert not status, status
    return head


def _environment_manifest() -> str:
    rows = {f"python=={platform.python_version()}"}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            rows.add(f"{name.lower().replace('_', '-')}=={distribution.version}")
    return "\n".join(sorted(rows)) + "\n"


def _episode_names_sha256(names: set[str] | list[str]) -> str:
    return _sha256_bytes("".join(f"{name}\n" for name in sorted(names)).encode())


def _paths_sha256(paths: set[Path]) -> str:
    return _sha256_bytes(
        "".join(f"{path}\n" for path in sorted(map(str, paths))).encode()
    )


def _scan_domain(root: Path) -> tuple[dict[str, dict[str, Any]], bytes, bytes]:
    root = root.resolve()
    assert root.is_dir(), root
    episodes: dict[str, dict[str, Any]] = {}
    for episode in sorted(root.glob("*.zarr"), key=lambda item: item.name):
        assert episode.is_dir(), episode
        episode_id = episode.name.removesuffix(".zarr")
        assert episode_id not in episodes, episode_id
        metadata_path = episode / "zarr.json"
        raw = metadata_path.read_bytes()
        frames = json.loads(raw)["attributes"]["total_frames"]
        assert isinstance(frames, int) and not isinstance(frames, bool) and frames > 0
        episodes[episode_id] = {
            "path": str(episode.resolve()),
            "frames": frames,
            "zarr_json_sha256": _sha256_bytes(raw),
        }
    assert episodes, root
    inventory = "".join(f"{name}\n" for name in sorted(episodes)).encode()
    metadata = "".join(
        f"{name}\t{episodes[name]['path']}\t{episodes[name]['frames']}\t"
        f"{episodes[name]['zarr_json_sha256']}\n"
        for name in sorted(episodes)
    ).encode()
    return episodes, inventory, metadata


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _compose(repo: Path, args: list[str]) -> str:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo)
    environment["HYDRA_FULL_ERROR"] = "1"
    command = [
        sys.executable,
        "-m",
        "egomimic.trainHydra",
        *args,
        "--cfg",
        "job",
        "--resolve",
    ]
    result = subprocess.run(
        command,
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip(), result.stderr
    return result.stdout


def _phase_args(
    *,
    repo: Path,
    experiment: str,
    mode: str,
    run_dir: Path,
    norm_cache_dir: Path | None,
    norm_artifact: Path | None,
    world_size: int,
    wandb: dict[str, str],
    offline: bool,
) -> list[str]:
    args = [
        "--config-name=train_zarr_cartesian",
        f"+experiment={experiment}",
        f"mode={mode}",
        f"hydra.run.dir={run_dir}",
        f"++paths.root_dir={run_dir}",
        f"paths.output_dir={run_dir}",
        f"paths.work_dir={repo}",
        f"launch_params.gpus_per_node={world_size}",
        "launch_params.nodes=1",
    ]
    if norm_cache_dir is not None:
        args.extend(
            [
                f"norm_stats.save_cache_dir={norm_cache_dir}",
                "norm_stats.precomputed_norm_path=null",
            ]
        )
    else:
        assert norm_artifact is not None
        args.extend(
            [
                "norm_stats.save_cache_dir=null",
                f"norm_stats.precomputed_norm_path={norm_artifact}",
                f"logger.wandb.offline={'true' if offline else 'false'}",
                f"logger.wandb.project={wandb['project']}",
                f"logger.wandb.entity={wandb['entity']}",
                f"logger.wandb.group={wandb['group']}",
                f"logger.wandb.id={wandb['id']}",
                f"++logger.wandb.name={wandb['id']}",
                "++logger.wandb.resume=never",
            ]
        )
    return args


def _plain_config(path: Path) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


def _assert_config_contract(
    configs: dict[str, Path], split_manifest: dict[str, Any], world_size: int
) -> dict[str, Any]:
    norm_cfg = OmegaConf.load(configs["norm"])
    smoke_cfg = OmegaConf.load(configs["smoke"])
    full_cfg = OmegaConf.load(configs["full"])
    for key in ("model", "data", "evaluator", "run_provenance"):
        full_value = OmegaConf.to_container(full_cfg[key], resolve=True)
        assert OmegaConf.to_container(smoke_cfg[key], resolve=True) == full_value, key
        assert OmegaConf.to_container(norm_cfg[key], resolve=True) == full_value, key
    assert full_cfg.seed == smoke_cfg.seed == split_manifest["split_seed"]
    assert float(full_cfg.run_provenance.valid_ratio) == float(
        split_manifest["valid_ratio"]
    )
    assert float(split_manifest["valid_ratio"]) == 0.01
    assert int(full_cfg.launch_params.gpus_per_node) == world_size
    assert int(full_cfg.launch_params.nodes) == 1
    assert int(full_cfg.trainer.devices) == world_size
    assert str(full_cfg.trainer.precision) == str(
        full_cfg.run_provenance.training_contract.precision
    )
    assert int(smoke_cfg.trainer.max_steps) == 2
    assert int(smoke_cfg.trainer.limit_train_batches) == 2
    assert int(smoke_cfg.trainer.val_check_interval) == 1
    assert int(smoke_cfg.trainer.limit_val_batches) == 1
    assert int(smoke_cfg.trainer.num_sanity_val_steps) == 0
    assert int(smoke_cfg.trainer.log_every_n_steps) == 1
    assert int(full_cfg.trainer.max_steps) == int(
        full_cfg.run_provenance.training_contract.max_steps
    )
    assert float(full_cfg.trainer.limit_val_batches) > 0
    assert int(full_cfg.trainer.val_check_interval) == int(
        full_cfg.run_provenance.training_contract.validation_interval_steps
    )
    assert full_cfg.model.train_metrics_on_step is True
    assert float(full_cfg.model.optimizer.lr) == float(
        full_cfg.run_provenance.training_contract.peak_lr
    )
    assert int(full_cfg.model.scheduler.warmup_steps) == int(
        full_cfg.run_provenance.training_contract.warmup_steps
    )
    assert float(full_cfg.model.scheduler.eta_min) == float(
        full_cfg.run_provenance.training_contract.cosine_floor_lr
    )
    domains = sorted(map(str, full_cfg.model.robomimic_model.domains))
    assert domains == sorted(split_manifest["domains"])
    assert sorted(full_cfg.data.train_datasets) == domains
    assert sorted(full_cfg.data.valid_datasets) == domains
    for domain in domains:
        train = full_cfg.data.train_datasets[domain]
        valid = full_cfg.data.valid_datasets[domain]
        evidence = split_manifest["domains"][domain]
        assert train.mode == "train" and valid.mode == "valid"
        for dataset in (train, valid):
            assert float(dataset.valid_ratio) == 0.01
            assert int(dataset.split_seed) == int(full_cfg.seed)
            assert int(dataset.expected_train_episode_count) == evidence["train_count"]
            assert int(dataset.expected_valid_episode_count) == evidence["valid_count"]
            assert (
                str(dataset.expected_train_episode_names_sha256)
                == evidence["train_names_sha256"]
            )
            assert (
                str(dataset.expected_valid_episode_names_sha256)
                == evidence["valid_names_sha256"]
            )
            assert str(dataset.resolver.folder_path) == evidence["folder_path"]
            assert (
                int(dataset.resolver.expected_episode_count) == evidence["total_count"]
            )
            assert (
                str(dataset.resolver.expected_episode_names_sha256)
                == evidence["inventory_names_sha256"]
            )
    return {
        "seed": int(full_cfg.seed),
        "domains": domains,
        "required_wandb_metrics": list(
            map(str, full_cfg.run_provenance.required_wandb_metrics)
        ),
        "norm": OmegaConf.to_container(
            full_cfg.run_provenance.norm_contract, resolve=True
        ),
        "training": OmegaConf.to_container(
            full_cfg.run_provenance.training_contract, resolve=True
        ),
        "action_horizon": int(full_cfg.model.robomimic_model.action_horizon),
        "action_dims": OmegaConf.to_container(
            full_cfg.model.robomimic_model.stages[2].action_dims, resolve=True
        ),
    }


def _validate_split_and_snapshot(
    *,
    config: Any,
    manifest: dict[str, Any],
    destination: Path,
    sample_frac: float,
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    assert manifest["status"] == "PASS"
    assert manifest["cross_domain_train_valid_resolved_path_overlap_count"] == 0
    domain_records: dict[str, Any] = {}
    norm_counts: dict[str, dict[str, int]] = {}
    all_train_paths: set[Path] = set()
    all_valid_paths: set[Path] = set()
    for domain in sorted(manifest["domains"]):
        evidence = manifest["domains"][domain]
        root = Path(evidence["folder_path"]).resolve()
        episodes, inventory, metadata = _scan_domain(root)
        names = set(episodes)
        train_ids = set(evidence["train_ids"])
        valid_ids = set(evidence["valid_ids"])
        assert train_ids.isdisjoint(valid_ids)
        assert train_ids | valid_ids == names
        assert len(names) == evidence["total_count"]
        assert _episode_names_sha256(names) == evidence["inventory_names_sha256"]
        assert _episode_names_sha256(train_ids) == evidence["train_names_sha256"]
        assert _episode_names_sha256(valid_ids) == evidence["valid_names_sha256"]
        train_paths = {Path(episodes[name]["path"]) for name in train_ids}
        valid_paths = {Path(episodes[name]["path"]) for name in valid_ids}
        assert _paths_sha256(train_paths) == evidence["train_resolved_paths_sha256"]
        assert _paths_sha256(valid_paths) == evidence["valid_resolved_paths_sha256"]
        assert train_paths.isdisjoint(valid_paths)
        all_train_paths.update(train_paths)
        all_valid_paths.update(valid_paths)
        inventory_path = destination / "inventories" / f"{domain}.txt"
        metadata_path = destination / "inventories" / f"{domain}.tsv"
        _write_bytes(inventory_path, inventory)
        _write_bytes(metadata_path, metadata)
        train_frames = sum(int(episodes[name]["frames"]) for name in train_ids)
        sampled_frames = max(1, math.ceil(sample_frac * train_frames))
        norm_counts[domain] = {
            "dataset_frames": train_frames,
            "sampled_frames": sampled_frames,
        }
        domain_records[domain] = {
            "root": str(root),
            "episode_count": len(names),
            "inventory": _record(inventory_path),
            "episode_metadata": _record(metadata_path),
            "train_frames": train_frames,
            "sampled_frames": sampled_frames,
        }
        resolver = config.data.train_datasets[domain].resolver
        assert str(resolver.folder_path) == str(root)
        assert int(resolver.expected_episode_count) == len(names)
        assert str(resolver.expected_episode_names_sha256) == _episode_names_sha256(
            names
        )
    assert all_train_paths.isdisjoint(all_valid_paths)
    return domain_records, norm_counts


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["status"] in {"STAGED", "READY"}
    return payload


def _assert_record(record: dict[str, Any]) -> Path:
    path = Path(record["path"])
    assert path.is_file() and path.stat().st_size == int(record["bytes"]), path
    assert _sha256(path) == record["sha256"], path
    return path


def _validate_live_domain(record: dict[str, Any]) -> None:
    episodes, inventory, metadata = _scan_domain(Path(record["root"]))
    assert len(episodes) == int(record["episode_count"])
    assert _sha256_bytes(inventory) == record["inventory"]["sha256"]
    assert _sha256_bytes(metadata) == record["episode_metadata"]["sha256"]


def _validate_norm_artifact(
    path: Path, contract: dict[str, Any], counts: dict[str, dict[str, int]], seed: int
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    assert payload["norm_mode"] == contract["norm_mode"]
    assert bool(payload["reduce_all_but_last"]) is bool(contract["reduce_all_but_last"])
    expected_by_embodiment = {
        str(item["embodiment_id"]): (domain, item)
        for domain, item in contract["domains"].items()
    }
    assert set(payload["stats"]) == set(expected_by_embodiment)
    metadata = payload["norm_run_metadata"]
    assert set(metadata["embodiments"]) == set(expected_by_embodiment)
    total_dataset = 0
    total_sampled = 0

    def nested_shape(value: Any) -> list[int]:
        if not isinstance(value, list):
            return []
        assert value, "normalization statistic arrays must not be empty"
        child_shapes = [nested_shape(child) for child in value]
        assert all(shape == child_shapes[0] for shape in child_shapes), (
            "ragged normalization statistic",
            child_shapes,
        )
        return [len(value), *child_shapes[0]]

    for embodiment, (domain, domain_contract) in expected_by_embodiment.items():
        expected_counts = counts[domain]
        item = metadata["embodiments"][embodiment]
        assert int(item["dataset_size"]) == expected_counts["dataset_frames"]
        assert int(item["sampled_frames"]) == expected_counts["sampled_frames"]
        assert float(item["sample_frac"]) == float(contract["sample_frac"])
        assert int(item["seed"]) == seed
        assert item["max_samples"] is None
        total_dataset += int(item["dataset_size"])
        total_sampled += int(item["sampled_frames"])
        stats = payload["stats"][embodiment]
        assert set(stats) == {"state_agent_obj", "actions"}
        for key, expected_shape_key in (
            ("state_agent_obj", "state_agent_obj_shape"),
            ("actions", "actions_shape"),
        ):
            expected_shape = list(domain_contract[expected_shape_key])
            values_by_stat = stats[key]
            assert len(values_by_stat) == 9
            assert {"quantile_1", "quantile_99"}.issubset(values_by_stat)
            for values in values_by_stat.values():
                assert nested_shape(values) == expected_shape
                flattened: list[float] = []
                stack = [values]
                while stack:
                    current = stack.pop()
                    if isinstance(current, list):
                        stack.extend(current)
                    else:
                        flattened.append(float(current))
                assert flattened and all(math.isfinite(value) for value in flattened)
            q1 = values_by_stat["quantile_1"]
            q99 = values_by_stat["quantile_99"]
            q1_stack, q99_stack = [q1], [q99]
            while q1_stack:
                left, right = q1_stack.pop(), q99_stack.pop()
                if isinstance(left, list):
                    assert isinstance(right, list) and len(left) == len(right)
                    q1_stack.extend(left)
                    q99_stack.extend(right)
                else:
                    assert float(left) <= float(right)
    assert int(metadata["total_dataset_frames"]) == total_dataset
    assert int(metadata["total_sampled_frames"]) == total_sampled
    assert int(payload["frames"]) == total_sampled
    return {
        "dataset_frames": total_dataset,
        "sampled_frames": total_sampled,
        "embodiments": sorted(expected_by_embodiment),
    }


def init_bundle(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    output_root = args.output_root.resolve()
    assert not output_root.exists(), output_root
    head = _require_clean_source(repo, args.expected_head)
    python_env = args.python_env.resolve()
    assert Path(sys.executable).resolve() == (python_env / "bin/python").resolve()
    expected_tools = {
        "launcher": repo / "scripts/train/flow_transfer_run_bundle.sbatch",
        "submitter": repo / "scripts/train/submit_flow_transfer_run_bundle.sh",
        "smoke_verifier": repo / "egomimic/scripts/verify_training_smoke.py",
        "full_verifier": repo / "egomimic/scripts/verify_training_full.py",
    }
    supplied_tools = {
        "launcher": args.launcher,
        "submitter": args.submitter,
        "smoke_verifier": args.smoke_verifier,
        "full_verifier": args.full_verifier,
    }
    for name, path in supplied_tools.items():
        assert path.resolve() == expected_tools[name].resolve(), (name, path)
        assert path.is_file()

    staging = output_root / "run_bundle_staging"
    config_dir = staging / "configs"
    config_dir.mkdir(parents=True)
    final_bundle_dir = output_root / "run_bundle"
    final_norm = final_bundle_dir / "artifacts" / "norm_stats.json"
    outputs = {
        "root": str(output_root),
        "staging": str(staging),
        "bundle_dir": str(final_bundle_dir),
        "bundle_manifest": str(final_bundle_dir / "bundle_manifest.json"),
        "norm_run": str(output_root / "norm" / "run"),
        "norm_cache": str(output_root / "norm" / "artifact"),
        "norm_artifact": str(final_norm),
        "smoke_run": str(output_root / "smoke" / "run"),
        "full_run": str(output_root / "full" / "run"),
    }
    wandb = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "group": args.wandb_group,
        "full_id": args.wandb_full_id,
        "smoke_id": args.wandb_smoke_id,
    }
    norm_args = _phase_args(
        repo=repo,
        experiment=args.full_experiment,
        mode="norm_stats",
        run_dir=Path(outputs["norm_run"]),
        norm_cache_dir=Path(outputs["norm_cache"]),
        norm_artifact=None,
        world_size=args.world_size,
        wandb={"project": "", "entity": "", "group": "", "id": ""},
        offline=True,
    )
    smoke_args = _phase_args(
        repo=repo,
        experiment=args.smoke_experiment,
        mode="train",
        run_dir=Path(outputs["smoke_run"]),
        norm_cache_dir=None,
        norm_artifact=final_norm,
        world_size=args.world_size,
        wandb={**wandb, "id": args.wandb_smoke_id},
        offline=True,
    )
    full_args = _phase_args(
        repo=repo,
        experiment=args.full_experiment,
        mode="train",
        run_dir=Path(outputs["full_run"]),
        norm_cache_dir=None,
        norm_artifact=final_norm,
        world_size=args.world_size,
        wandb={**wandb, "id": args.wandb_full_id},
        offline=False,
    )
    full_args.append("callbacks.model_checkpoint.train_time_interval.hours=1")
    hydra_args = {"norm": norm_args, "smoke": smoke_args, "full": full_args}
    config_paths: dict[str, Path] = {}
    for phase, phase_args in hydra_args.items():
        path = config_dir / f"{phase}.yaml"
        path.write_text(_compose(repo, phase_args))
        config_paths[phase] = path

    full_cfg = OmegaConf.load(config_paths["full"])
    split_path = repo / str(full_cfg.run_provenance.split_manifest_path)
    assert _sha256(split_path) == str(full_cfg.run_provenance.split_manifest_sha256)
    split_manifest = json.loads(split_path.read_text())
    contract = _assert_config_contract(config_paths, split_manifest, args.world_size)
    split_copy = staging / "split_manifest.json"
    shutil.copy2(split_path, split_copy)
    sample_frac = float(contract["norm"]["sample_frac"])
    domains, norm_counts = _validate_split_and_snapshot(
        config=full_cfg,
        manifest=split_manifest,
        destination=staging,
        sample_frac=sample_frac,
    )
    environment_path = staging / "python_environment.txt"
    environment_path.write_text(_environment_manifest())
    commit_path = staging / "git_commit.txt"
    commit_path.write_text(_git(repo, "show", "-s", "--format=fuller", head))
    spec = {
        "schema_version": SCHEMA_VERSION,
        "status": "STAGED",
        "run_id": args.run_id,
        "source": {
            "head": head,
            "clean": True,
            "repo": str(repo),
            "commit_record": _record(commit_path),
        },
        "python_environment": {
            "path": str(python_env),
            "manifest": _record(environment_path),
        },
        "tools": {
            "bundle": _record(Path(__file__)),
            "launcher": _record(args.launcher),
            "submitter": _record(args.submitter),
            "smoke_verifier": _record(args.smoke_verifier),
            "full_verifier": _record(args.full_verifier),
        },
        "experiments": {
            "config_name": "train_zarr_cartesian",
            "full": args.full_experiment,
            "smoke": args.smoke_experiment,
        },
        "resources": {
            "account": args.account,
            "partition": args.partition,
            "smoke_qos": args.smoke_qos,
            "full_qos": args.full_qos,
            "gpu_type": args.gpu_type,
            "world_size": args.world_size,
            "nodes": 1,
            "cpus_per_task": args.cpus_per_task,
            "memory": args.memory,
            "smoke_time": args.smoke_time,
            "full_time": args.full_time,
            "full_signal": args.full_signal,
        },
        "outputs": outputs,
        "wandb": wandb,
        "hydra_args": hydra_args,
        "configs": {phase: _record(path) for phase, path in config_paths.items()},
        "split": {
            "manifest": _record(split_copy),
            "domains": domains,
            "cross_split_overlap_count": 0,
        },
        "normalization": {
            "contract": contract["norm"],
            "counts": norm_counts,
            "artifact": None,
        },
        "contract": contract,
    }
    spec_path = staging / "staging_spec.json"
    _atomic_json(spec_path, spec)
    for path in staging.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    print(spec_path)


def verify_bundle(path: Path, phase: str) -> dict[str, Any]:
    manifest = _load_manifest(path)
    repo = Path(manifest["source"]["repo"])
    _require_clean_source(repo, manifest["source"]["head"])
    assert manifest["source"]["clean"] is True
    assert (
        Path(sys.executable).resolve()
        == (Path(manifest["python_environment"]["path"]) / "bin/python").resolve()
    )
    for record in manifest["tools"].values():
        _assert_record(record)
    _assert_record(manifest["source"]["commit_record"])
    environment_record = manifest["python_environment"]["manifest"]
    _assert_record(environment_record)
    assert (
        _sha256_bytes(_environment_manifest().encode()) == environment_record["sha256"]
    )
    for record in manifest["configs"].values():
        _assert_record(record)
    _assert_record(manifest["split"]["manifest"])
    for record in manifest["split"]["domains"].values():
        _assert_record(record["inventory"])
        _assert_record(record["episode_metadata"])
        _validate_live_domain(record)
    if phase != "norm":
        assert manifest["status"] == "READY"
        artifact = _assert_record(manifest["normalization"]["artifact"])
        _validate_norm_artifact(
            artifact,
            manifest["normalization"]["contract"],
            manifest["normalization"]["counts"],
            int(manifest["contract"]["seed"]),
        )
    print(json.dumps({"status": "PASS", "phase": phase}, sort_keys=True))
    return manifest


def finalize_norm(args: argparse.Namespace) -> None:
    spec_path = args.spec.resolve()
    spec = verify_bundle(spec_path, "norm")
    assert spec["status"] == "STAGED"
    generated = Path(spec["outputs"]["norm_cache"]) / "norm_stats" / "norm_stats.json"
    assert generated.is_file(), generated
    norm_summary = _validate_norm_artifact(
        generated,
        spec["normalization"]["contract"],
        spec["normalization"]["counts"],
        int(spec["contract"]["seed"]),
    )
    bundle_dir = Path(spec["outputs"]["bundle_dir"])
    assert not bundle_dir.exists(), bundle_dir
    artifacts_dir = bundle_dir / "artifacts"
    evidence_dir = bundle_dir / "evidence"
    artifacts_dir.mkdir(parents=True)
    evidence_dir.mkdir()
    final_norm = Path(spec["outputs"]["norm_artifact"])
    shutil.copy2(generated, final_norm)
    for name, record in spec["configs"].items():
        destination = evidence_dir / f"resolved_{name}.yaml"
        shutil.copy2(record["path"], destination)
        spec["configs"][name] = _record(destination)
    split_destination = evidence_dir / "split_manifest.json"
    shutil.copy2(spec["split"]["manifest"]["path"], split_destination)
    spec["split"]["manifest"] = _record(split_destination)
    for domain, record in spec["split"]["domains"].items():
        for key, suffix in (("inventory", "txt"), ("episode_metadata", "tsv")):
            destination = evidence_dir / f"{domain}.{suffix}"
            shutil.copy2(record[key]["path"], destination)
            record[key] = _record(destination)
    for section, name in (
        (spec["source"], "commit_record"),
        (spec["python_environment"], "manifest"),
    ):
        destination = evidence_dir / Path(section[name]["path"]).name
        shutil.copy2(section[name]["path"], destination)
        section[name] = _record(destination)
    spec["normalization"]["artifact"] = _record(final_norm)
    spec["normalization"]["summary"] = norm_summary
    spec["staging_spec_sha256"] = _sha256(spec_path)
    spec["status"] = "READY"
    bundle_path = Path(spec["outputs"]["bundle_manifest"])
    _atomic_json(bundle_path, spec)
    for path in bundle_dir.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    verify_bundle(bundle_path, "smoke")
    print(bundle_path)


def _semantic_config_equal(actual: Path, expected: Path) -> bool:
    return _plain_config(actual) == _plain_config(expected)


def _validate_smoke_result_live(
    manifest: dict[str, Any], result_path: Path
) -> dict[str, Any]:
    result_path = result_path.resolve()
    result = json.loads(result_path.read_text())
    assert result["status"] == "passed"
    assert result["repo_head"] == manifest["source"]["head"]
    assert (
        Path(result["output"]).resolve()
        == Path(manifest["outputs"]["smoke_run"]).resolve()
    )
    assert int(result["global_step"]) == 2
    assert int(result["world_size"]) == int(manifest["resources"]["world_size"])
    assert int(result["wandb_exit_code"]) == 0
    assert result["model_wrapper_load"]["status"] == "passed"
    assert result["model_wrapper_load"]["strict"] is True
    assert result["model_wrapper_load"]["map_location"] == "cpu"
    assert (
        result["required_wandb_metrics"]
        == manifest["contract"]["required_wandb_metrics"]
    )
    assert result["dense_training_steps"] == [0, 1]
    assert int(result["validation_trainer_global_step"]) >= 1
    smoke_run = Path(manifest["outputs"]["smoke_run"]).resolve()
    config_path = Path(result["config"])
    assert config_path.resolve() == (smoke_run / ".hydra" / "config.yaml").resolve()
    assert _sha256(config_path) == result["config_sha256"]
    expected_config = Path(manifest["configs"]["smoke"]["path"])
    assert _semantic_config_equal(config_path, expected_config)
    for key, sha_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("wandb_stream", "wandb_stream_sha256"),
    ):
        item = Path(result[key])
        assert smoke_run in item.resolve().parents
        assert _sha256(item) == result[sha_key]
    return result


def verify_smoke_result(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    manifest = verify_bundle(manifest_path, "smoke")
    result_path = args.result.resolve()
    result = _validate_smoke_result_live(manifest, result_path)
    gate = {
        "schema_version": 1,
        "status": "PASS",
        "source_head": manifest["source"]["head"],
        "bundle_manifest": _record(manifest_path),
        "smoke_result": _record(result_path),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "wandb_stream_sha256": result["wandb_stream_sha256"],
        "required_wandb_metrics": result["required_wandb_metrics"],
    }
    gate_path = Path(manifest["outputs"]["bundle_dir"]) / "smoke_gate.json"
    assert not gate_path.exists(), gate_path
    _atomic_json(gate_path, gate)
    gate_path.chmod(0o444)
    print(gate_path)


def verify_smoke_gate(manifest_path: Path) -> dict[str, Any]:
    manifest = verify_bundle(manifest_path, "full")
    gate_path = Path(manifest["outputs"]["bundle_dir"]) / "smoke_gate.json"
    gate = json.loads(gate_path.read_text())
    assert gate["status"] == "PASS"
    assert gate["source_head"] == manifest["source"]["head"]
    assert gate["bundle_manifest"]["sha256"] == _sha256(manifest_path)
    result_record = gate["smoke_result"]
    result_path = _assert_record(result_record)
    result = _validate_smoke_result_live(manifest, result_path)
    assert result["checkpoint_sha256"] == gate["checkpoint_sha256"]
    assert result["wandb_stream_sha256"] == gate["wandb_stream_sha256"]
    print(json.dumps({"status": "PASS", "phase": "full-smoke-gate"}))
    return manifest


def emit_args(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest.resolve())
    for item in manifest["hydra_args"][args.phase]:
        sys.stdout.buffer.write(str(item).encode() + b"\0")


def print_field(args: argparse.Namespace) -> None:
    value: Any = _load_manifest(args.manifest.resolve())
    for part in args.field.split("."):
        value = value[part]
    if isinstance(value, (dict, list)):
        print(json.dumps(value, sort_keys=True))
    elif isinstance(value, bool):
        print("true" if value else "false")
    else:
        print(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--repo", required=True, type=Path)
    init.add_argument("--expected-head", required=True)
    init.add_argument("--python-env", required=True, type=Path)
    init.add_argument("--output-root", required=True, type=Path)
    init.add_argument("--run-id", required=True)
    init.add_argument("--full-experiment", required=True)
    init.add_argument("--smoke-experiment", required=True)
    init.add_argument("--wandb-project", required=True)
    init.add_argument("--wandb-entity", required=True)
    init.add_argument("--wandb-group", required=True)
    init.add_argument("--wandb-full-id", required=True)
    init.add_argument("--wandb-smoke-id", required=True)
    init.add_argument("--account", required=True)
    init.add_argument("--partition", required=True)
    init.add_argument("--smoke-qos", default="short")
    init.add_argument("--full-qos", default="long")
    init.add_argument("--gpu-type", required=True)
    init.add_argument("--world-size", type=int, required=True)
    init.add_argument("--cpus-per-task", type=int, default=8)
    init.add_argument("--memory", default="128G")
    init.add_argument("--smoke-time", default="02:00:00")
    init.add_argument("--full-time", default="2-00:00:00")
    init.add_argument("--full-signal", default="USR1@300")
    init.add_argument("--launcher", required=True, type=Path)
    init.add_argument("--submitter", required=True, type=Path)
    init.add_argument("--smoke-verifier", required=True, type=Path)
    init.add_argument("--full-verifier", required=True, type=Path)
    init.set_defaults(func=init_bundle)

    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--phase", choices=("norm", "smoke", "full"), required=True)
    verify.set_defaults(func=lambda args: verify_bundle(args.manifest, args.phase))

    finalize = commands.add_parser("finalize-norm")
    finalize.add_argument("--spec", required=True, type=Path)
    finalize.set_defaults(func=finalize_norm)

    smoke = commands.add_parser("verify-smoke-result")
    smoke.add_argument("--manifest", required=True, type=Path)
    smoke.add_argument("--result", required=True, type=Path)
    smoke.set_defaults(func=verify_smoke_result)

    gate = commands.add_parser("verify-smoke-gate")
    gate.add_argument("--manifest", required=True, type=Path)
    gate.set_defaults(func=lambda args: verify_smoke_gate(args.manifest))

    emit = commands.add_parser("emit-args")
    emit.add_argument("--manifest", required=True, type=Path)
    emit.add_argument("--phase", choices=("norm", "smoke", "full"), required=True)
    emit.set_defaults(func=emit_args)

    field = commands.add_parser("print-field")
    field.add_argument("--manifest", required=True, type=Path)
    field.add_argument("--field", required=True)
    field.set_defaults(func=print_field)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
