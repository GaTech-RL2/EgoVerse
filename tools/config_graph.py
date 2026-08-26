#!/usr/bin/env python3
"""Build and lint dependency graphs from instantiated PipelineAlgo stages.

The graph source of truth is each instantiated stage's ``contract(mode)``.
YAML is used only for constructor parameters and for deriving the dataset keys
that seed the pipeline.  The output is compatible with the canonical
``pipeline-config-graph`` spread-dependency renderer.

Examples::

    python tools/config_graph.py graph.json path/to/model.yaml
    python tools/config_graph.py --mode rollout --lint graph.json model.yaml
    python tools/config_graph.py --mode both graph.json experiment.yaml

An experiment fragment under ``hydra_configs/experiment`` is supported when
its defaults select ``/model`` and ``/data``.  A fully resolved Hydra config or
a model YAML containing ``robomimic_model.stages`` is also supported.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

_MODES = ("train", "rollout")
_FIXED_RUNTIME_SEEDS = {
    "batch_size",
    "cu_seqlens",
    "max_seq_len",
    "seq_lens",
}


def _plain(value: Any) -> Any:
    """Convert OmegaConf containers without resolving or truncating values."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=False)
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def hyperparams(stage_config: Any) -> dict[str, Any]:
    """Return the complete stage constructor tree for the renderer sidebar."""
    plain = _plain(stage_config)
    if not isinstance(plain, Mapping):
        return {}
    return {str(key): value for key, value in plain.items() if str(key) != "_target_"}


def stream_of(key: str) -> str:
    """Retain the historical stream color metadata used by the renderer."""
    rules = (
        ("route", (r"^route",)),
        ("A", (r"^A(_|$)", r"^a_top$", r"^emb_object$", r"^apex/")),
        ("S", (r"^S(_|$)", r"^s$", r"^emb_pusher$")),
    )
    for name, patterns in rules:
        if any(re.search(pattern, key) for pattern in patterns):
            return name
    return "shared"


def depth_of(key: str) -> int:
    match = re.search(r"_L(\d+)", key)
    return int(match.group(1)) if match else 0


def _matches(pattern: str, key: str) -> bool:
    """Whether a declared key pattern covers a concrete key or pattern."""
    if pattern == key:
        return True
    if pattern.endswith("*") and key.startswith(pattern[:-1]):
        return True
    if key.endswith("*") and pattern.startswith(key[:-1]):
        return True
    return False


def _provided(read: str, available: Iterable[str]) -> bool:
    return any(_matches(candidate, read) for candidate in available)


def _expand_reads(reads: Iterable[str], seed_and_written: Iterable[str]) -> list[str]:
    """Resolve wildcard reads to selected dataset keys when they are known."""
    pool = list(seed_and_written)
    expanded: list[str] = []
    for raw in reads:
        key = str(raw)
        if key.endswith("*"):
            matches = sorted(
                candidate
                for candidate in pool
                if not candidate.endswith("*") and _matches(key, candidate)
            )
            expanded.extend(matches or [key])
        else:
            expanded.append(key)
    return list(dict.fromkeys(expanded))


def _find_config_root(path: Path) -> Path | None:
    for parent in (path.parent, *path.parents):
        if (parent / "model").is_dir() and (parent / "data").is_dir():
            return parent
    return None


def _defaults_selection(raw_config: Any, group: str) -> str | None:
    plain = _plain(raw_config)
    if not isinstance(plain, Mapping):
        return None
    for entry in plain.get("defaults", []) or []:
        if not isinstance(entry, Mapping):
            continue
        for raw_key, value in entry.items():
            key = str(raw_key).replace("override ", "").lstrip("/")
            if key == group and isinstance(value, str):
                return value
    return None


def _locate_model_config(config: Any) -> Any:
    from omegaconf import OmegaConf

    candidates = (
        OmegaConf.select(config, "model.robomimic_model"),
        OmegaConf.select(config, "robomimic_model"),
        config if OmegaConf.select(config, "stages") is not None else None,
    )
    for candidate in candidates:
        if candidate is not None and OmegaConf.select(candidate, "stages") is not None:
            return candidate
    raise ValueError(
        "selected YAML has no PipelineAlgo stage list; pass a model YAML, a "
        "resolved Hydra config, or an experiment fragment selecting /model"
    )


def _load_selected_config(path: Path) -> tuple[Any, dict[str, str]]:
    """Load model/full config, resolving common experiment-default fragments."""
    from omegaconf import OmegaConf

    raw = OmegaConf.load(path)
    try:
        _locate_model_config(raw)
        return raw, {"selected": str(path)}
    except ValueError:
        pass

    root = _find_config_root(path)
    model_name = _defaults_selection(raw, "model")
    if root is None or model_name is None:
        _locate_model_config(raw)  # raise the canonical error
        raise AssertionError("unreachable")

    model_path = root / "model" / f"{model_name}.yaml"
    if not model_path.is_file():
        raise FileNotFoundError(f"selected model config does not exist: {model_path}")
    combined: dict[str, Any] = {
        "model": _plain(OmegaConf.load(model_path)),
    }
    sources = {"selected": str(path), "model": str(model_path)}

    data_name = _defaults_selection(raw, "data")
    if data_name is not None:
        data_path = root / "data" / f"{data_name}.yaml"
        if not data_path.is_file():
            raise FileNotFoundError(f"selected data config does not exist: {data_path}")
        combined["data"] = _plain(OmegaConf.load(data_path))
        sources["data"] = str(data_path)

    # Keep experiment-level metadata available without allowing it to replace
    # the explicitly selected model/data components above.
    raw_plain = _plain(raw)
    if isinstance(raw_plain, Mapping):
        for key, value in raw_plain.items():
            if key not in {"defaults", "model", "data"}:
                combined[str(key)] = value
    return OmegaConf.create(combined), sources


def _instantiate_stages(model_config: Any) -> tuple[list[Any], list[Any]]:
    from hydra.utils import instantiate

    stage_configs = list(model_config.stages)
    stages = [instantiate(stage_config) for stage_config in stage_configs]
    for index, stage in enumerate(stages):
        if not callable(getattr(stage, "contract", None)):
            raise TypeError(
                f"stage {index} {type(stage).__name__} has no contract(mode) method"
            )
    return stage_configs, stages


def _configured_model_observations(
    stage_configs: Iterable[Any], domains: Sequence[str]
) -> dict[str, set[str]]:
    """Derive each domain's exact observation aliases from its encoder tree.

    ``SharedAdapterObsEncoder`` nests the common encoder several levels below
    ``FusedObsEncoder`` and places domain-specific adapters under an
    ``encoders.<domain>`` mapping.  Walking only the first ``obs_specs`` block
    loses both branches.  This traversal narrows its domain scope only when an
    ``encoders`` mapping is actually keyed by configured model domains.
    """
    domain_names = list(domains) or ["<unspecified>"]
    observations = {domain: set() for domain in domain_names}

    def collect(value: Any, scope: set[str]) -> None:
        if isinstance(value, list):
            for item in value:
                collect(item, scope)
            return
        if not isinstance(value, Mapping):
            return

        for field in ("obs_specs", "img_encoders"):
            field_value = value.get(field, {})
            if isinstance(field_value, Mapping):
                for domain in scope:
                    observations[domain].update(f"obs/{key}" for key in field_value)

        domain_encoders = value.get("encoders")
        narrowed = False
        if isinstance(domain_encoders, Mapping):
            configured = set(domain_names).intersection(
                str(key) for key in domain_encoders
            )
            if configured:
                narrowed = True
                for domain in configured:
                    collect(domain_encoders[domain], {domain})

        for key, child in value.items():
            if key in {"obs_specs", "img_encoders"}:
                continue
            if key == "encoders" and narrowed:
                continue
            collect(child, scope)

    for stage_config in stage_configs:
        plain = _plain(stage_config)
        if not isinstance(plain, Mapping):
            continue
        target = str(plain.get("_target_", ""))
        if target.endswith("FusedObsEncoder"):
            collect(plain.get("encoder", {}), set(domain_names))
    return observations


def _transform_key_inventory(keys: set[str], transforms: Iterable[Any]) -> set[str]:
    """Apply transform constructors' explicit key effects to an inventory.

    This is deliberately key-level static analysis: it never opens an episode
    or invents tensor shapes.  The transform objects used by the loaders expose
    their output and deletion names as constructor attributes, which is enough
    to follow Fold's concat/drop pipelines and the in-place PushShapes action
    transforms.
    """
    inventory = set(keys)
    output_attributes = (
        "new_key_name",
        "new_key",
        "out_key",
        "output_action_key",
        "transformed_key_name",
    )
    for transform in transforms:
        additions: set[str] = set()
        removals: set[str] = set()
        for attribute in output_attributes:
            value = getattr(transform, attribute, None)
            if isinstance(value, str):
                additions.add(value)

        output_key_list = getattr(transform, "output_key_list", None)
        if output_key_list is not None:
            additions.update(str(item[0]) for item in output_key_list)
        parts = getattr(transform, "parts", None)
        if parts is not None:
            additions.update(str(item[0]) for item in parts)

        name = type(transform).__name__
        if name == "ConcatKeys" and bool(getattr(transform, "delete_old_keys", False)):
            removals.update(str(key) for key in transform.key_list)
        elif name == "SplitConcat" and bool(
            getattr(transform, "delete_old_key", False)
        ):
            removals.add(str(transform.in_key))
        elif name == "DeleteKeys":
            removals.update(str(key) for key in transform.keys_to_delete)
        elif name == "DropKeys":
            removals.update(str(key) for key in transform.keys)

        inventory.difference_update(removals)
        inventory.update(additions)
    return inventory


def _dataset_contracts(
    config: Any,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Call selected key-map/transform factories without constructing datasets."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    data = OmegaConf.select(config, "data")
    if data is None and OmegaConf.select(config, "train_datasets") is not None:
        data = config
    if data is None:
        return {}, []
    datasets = OmegaConf.select(data, "train_datasets")
    if datasets is None:
        datasets = OmegaConf.select(data, "valid_datasets")
    if datasets is None:
        return {}, []

    by_domain: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for dataset_name, dataset_config in datasets.items():
        keymap_config = OmegaConf.select(dataset_config, "resolver.key_map")
        if keymap_config is None:
            keymap_config = OmegaConf.select(dataset_config, "key_map")
        if keymap_config is None:
            warnings.append(f"dataset {dataset_name}: no key_map config")
            continue
        # Key derivation is best effort; stage instantiation itself remains
        # strict and is handled separately.
        try:
            keymap = instantiate(keymap_config)
        except Exception as exc:
            warnings.append(
                f"dataset {dataset_name}: key_map could not be called: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if not isinstance(keymap, Mapping):
            warnings.append(f"dataset {dataset_name}: key_map did not return a mapping")
            continue
        observations: set[str] = set()
        for alias, info in keymap.items():
            if not isinstance(info, Mapping):
                continue
            if info.get("key_type") in {
                "camera_keys",
                "proprio_keys",
                "lang_keys",
            }:
                observations.add(str(alias))

        transform_config = OmegaConf.select(dataset_config, "resolver.transform_list")
        if transform_config is None:
            transform_config = OmegaConf.select(dataset_config, "transform_list")
        transforms: list[Any] = []
        if transform_config is not None:
            try:
                instantiated = instantiate(transform_config)
                transforms = list(instantiated or [])
            except Exception as exc:
                warnings.append(
                    f"dataset {dataset_name}: transform_list could not be called: "
                    f"{type(exc).__name__}: {exc}"
                )

        raw_keys = {str(key) for key in keymap}
        final_keys = _transform_key_inventory(raw_keys, transforms)
        domain = OmegaConf.select(dataset_config, "resolver.embodiment_override")
        domain = str(domain or dataset_name)
        contract = {
            "datasets": [str(dataset_name)],
            "raw_observations": observations,
            "final_keys": final_keys,
            "transform_count": len(transforms),
        }
        if domain not in by_domain:
            by_domain[domain] = contract
        else:
            # Every leaf feeding one model domain must satisfy its observation
            # contract, so availability is an intersection, not a union.
            existing = by_domain[domain]
            existing["datasets"].append(str(dataset_name))
            existing["raw_observations"].intersection_update(observations)
            existing["final_keys"].intersection_update(final_keys)
            existing["transform_count"] += len(transforms)
    return by_domain, warnings


def _raw_contracts(
    stages: Iterable[Any], mode: str
) -> list[tuple[list[str], list[str]]]:
    contracts = []
    for stage in stages:
        reads, writes = stage.contract(mode)
        contracts.append(
            ([str(key) for key in reads or ()], [str(key) for key in writes or ()])
        )
    return contracts


def _seed_inventory(
    *,
    config: Any,
    model_config: Any,
    stage_configs: list[Any],
    stages: list[Any],
    mode: str,
) -> tuple[
    list[str],
    dict[str, list[str]],
    str,
    list[str],
    list[str],
    dict[str, dict[str, Any]],
]:
    """Derive exact per-domain ambient keys from data/model configuration."""

    contracts = _raw_contracts(stages, mode)
    all_reads = [key for reads, _ in contracts for key in reads]
    domains = [str(item) for item in _plain(model_config.get("domains", []))]
    dataset_contracts, warnings = _dataset_contracts(config)
    if not domains:
        domains = sorted(dataset_contracts) or ["<unspecified>"]
    model_obs = _configured_model_observations(stage_configs, domains)
    has_model_obs = any(model_obs.values())
    seed_problems: list[str] = []
    details: dict[str, dict[str, Any]] = {}

    if dataset_contracts:
        has_transforms = any(
            contract["transform_count"] for contract in dataset_contracts.values()
        )
        source = (
            "dataset-keymap+transform-contract" if has_transforms else "dataset-keymap"
        )
        observations_by_domain: dict[str, set[str]] = {}
        for domain in domains:
            expected = set(model_obs.get(domain, set()))
            dataset_contract = dataset_contracts.get(domain)
            if dataset_contract is None:
                warnings.append(
                    f"model domain {domain}: no matching selected dataset contract"
                )
                seed_problems.append(
                    f"{domain}: selected config has no dataset contract for this "
                    "model domain"
                )
                observations_by_domain[domain] = set()
                details[domain] = {
                    "model_observations": sorted(expected),
                    "dataset_names": [],
                    "dataset_raw_observations": [],
                    "dataset_final_keys": [],
                }
                continue

            final_keys = set(dataset_contract["final_keys"])
            raw_observations = set(dataset_contract["raw_observations"])
            if expected:
                available = {
                    key for key in expected if key.removeprefix("obs/") in final_keys
                }
                missing = sorted(expected - available)
                if missing:
                    seed_problems.append(
                        f"{domain}: model observations absent after selected "
                        f"dataset key_map and transforms: {missing}"
                    )
                observations_by_domain[domain] = available
            else:
                # Legacy model configs may not expose nested observation specs.
                # Retain only loader-declared observations that survive every
                # selected transform for this domain.
                observations_by_domain[domain] = {
                    f"obs/{key}" for key in raw_observations if key in final_keys
                }
            details[domain] = {
                "model_observations": sorted(expected),
                "dataset_names": list(dataset_contract["datasets"]),
                "dataset_raw_observations": sorted(
                    f"obs/{key}" for key in raw_observations
                ),
                "dataset_final_keys": sorted(final_keys),
            }
    elif has_model_obs:
        source = "model-observation-config"
        observations_by_domain = {
            domain: set(model_obs.get(domain, set())) for domain in domains
        }
        for domain in domains:
            details[domain] = {
                "model_observations": sorted(model_obs.get(domain, set())),
                "dataset_names": [],
                "dataset_raw_observations": [],
                "dataset_final_keys": [],
            }
    else:
        source = "contract-fallback"
        exact_obs = {
            key for key in all_reads if key.startswith("obs/") and not key.endswith("*")
        }
        if not exact_obs and any(key == "obs/*" for key in all_reads):
            exact_obs = {"obs/*"}
        observations_by_domain = {domain: set(exact_obs) for domain in domains}
        for domain in domains:
            details[domain] = {
                "model_observations": [],
                "dataset_names": [],
                "dataset_raw_observations": [],
                "dataset_final_keys": [],
            }

    needs_embodiment = "embodiment" in all_reads or bool(domains)
    needs_actions = mode == "train" and "actions" in all_reads
    fixed = sorted(key for key in _FIXED_RUNTIME_SEEDS if key in all_reads)
    by_domain: dict[str, list[str]] = {}
    for domain, observations in observations_by_domain.items():
        keys = set(observations)
        if needs_embodiment:
            keys.add("embodiment")
        if needs_actions:
            keys.add("actions")
        keys.update(fixed)
        if mode == "rollout":
            keys.add("rollout_t")
        by_domain[domain] = sorted(keys)
    union = sorted({key for keys in by_domain.values() for key in keys})
    return union, by_domain, source, warnings, seed_problems, details


def _stage_name(stage_config: Any, stage: Any) -> tuple[str, str]:
    target = str(_plain(stage_config).get("_target_", ""))
    return (target.rsplit(".", 1)[-1] or type(stage).__name__, target)


def _build_graph_from_loaded(
    *,
    selected_path: Path,
    config: Any,
    sources: dict[str, str],
    model_config: Any,
    stage_configs: list[Any],
    stages: list[Any],
    mode: str,
) -> dict[str, Any]:
    if mode not in _MODES:
        raise ValueError(f"mode must be train|rollout, got {mode!r}")
    (
        seeds,
        seeds_by_domain,
        seed_source,
        seed_warnings,
        seed_problems,
        seed_details,
    ) = _seed_inventory(
        config=config,
        model_config=model_config,
        stage_configs=stage_configs,
        stages=stages,
        mode=mode,
    )

    nodes: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_by_domain = {
        domain: list(domain_seeds) for domain, domain_seeds in seeds_by_domain.items()
    }
    for source_index, (stage_config, stage) in enumerate(zip(stage_configs, stages)):
        name, target = _stage_name(stage_config, stage)
        if mode == "rollout" and bool(getattr(stage, "train_only", False)):
            skipped.append(
                {
                    "source_i": source_index,
                    "t": name,
                    "target": target,
                    "reason": "train-only",
                }
            )
            continue
        declared_reads, declared_writes = stage.contract(mode)
        reads_by_domain = {
            domain: _expand_reads(
                (str(key) for key in declared_reads or ()), domain_seen
            )
            for domain, domain_seen in seen_by_domain.items()
        }
        reads = list(
            dict.fromkeys(
                key for domain_reads in reads_by_domain.values() for key in domain_reads
            )
        )
        # Wildcard writes describe a family and must not disappear merely
        # because this static graph cannot know the concrete runtime suffixes.
        writes = list(dict.fromkeys(str(key) for key in declared_writes or ()))
        streams = {stream_of(key) for key in (writes or reads)} or {"shared"}
        stream = next(iter(streams)) if len(streams) == 1 else "both"
        node_index = len(nodes)
        nodes.append(
            {
                "i": node_index,
                "source_i": source_index,
                "t": name,
                "target": target,
                # Compatibility metadata only; the canonical renderer has no
                # section bands and the builder does not synthesize any.
                "sec": "PIPELINE",
                "stream": stream,
                "depth": max((depth_of(key) for key in writes or reads), default=0),
                "mode": mode,
                "in": reads,
                "in_by_domain": reads_by_domain,
                "out": writes,
                "declared_in": [str(key) for key in declared_reads or ()],
                "declared_out": [str(key) for key in declared_writes or ()],
                "p": hyperparams(stage_config),
            }
        )
        for domain_seen in seen_by_domain.values():
            domain_seen.extend(key for key in writes if key not in domain_seen)

    edges: list[dict[str, Any]] = []
    for node in nodes:
        for read in node["in"]:
            earlier = [
                writer["i"]
                for writer in nodes[: node["i"]]
                if any(_matches(write, read) for write in writer["out"])
            ]
            if earlier:
                producer = max(earlier)
            elif _provided(read, seeds):
                continue
            else:
                later = [
                    writer["i"]
                    for writer in nodes[node["i"] + 1 :]
                    if any(_matches(write, read) for write in writer["out"])
                ]
                if not later:
                    continue
                producer = min(later)
            edges.append(
                {
                    "a": producer,
                    "b": node["i"],
                    "k": read,
                    "s": stream_of(read),
                }
            )

    graph: dict[str, Any] = {
        "nodes": nodes,
        "edges": edges,
        "domains": _plain(model_config.get("domains", [])),
        "source": str(selected_path),
        "source_name": selected_path.name,
        "component_sources": sources,
        "mode": mode,
        "seed_keys": seeds,
        "seed_keys_by_domain": seeds_by_domain,
        "seed_key_source": seed_source,
        "seed_key_details": seed_details,
        "seed_warnings": seed_warnings,
        "seed_problems": seed_problems,
        "skipped_stages": skipped,
    }
    graph["lint"] = lint(graph)
    return graph


def build_graph(path: str | os.PathLike[str], mode: str = "train") -> dict[str, Any]:
    """Instantiate a selected config and return one mode-specific graph."""
    selected_path = Path(path).resolve()
    config, sources = _load_selected_config(selected_path)
    model_config = _locate_model_config(config)
    stage_configs, stages = _instantiate_stages(model_config)
    return _build_graph_from_loaded(
        selected_path=selected_path,
        config=config,
        sources=sources,
        model_config=model_config,
        stage_configs=stage_configs,
        stages=stages,
        mode=mode,
    )


def _cycle(edges: Iterable[Mapping[str, Any]], node_count: int) -> list[int] | None:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        adjacency[int(edge["a"])].append(int(edge["b"]))
    state = [0] * node_count
    stack: list[int] = []

    def visit(node: int) -> list[int] | None:
        state[node] = 1
        stack.append(node)
        for neighbor in adjacency[node]:
            if state[neighbor] == 0:
                found = visit(neighbor)
                if found:
                    return found
            elif state[neighbor] == 1:
                start = stack.index(neighbor)
                return stack[start:] + [neighbor]
        stack.pop()
        state[node] = 2
        return None

    for node in range(node_count):
        if state[node] == 0:
            found = visit(node)
            if found:
                return found
    return None


def lint(graph: Mapping[str, Any]) -> list[str]:
    """Report unresolved reads, duplicate concrete writers, and cycles."""
    problems = [str(problem) for problem in graph.get("seed_problems", [])]
    nodes = list(graph.get("nodes", []))
    seeds_by_domain = graph.get("seed_keys_by_domain") or {
        "<unspecified>": list(graph.get("seed_keys", []))
    }
    multiple_domains = len(seeds_by_domain) > 1
    for domain, seed_keys in seeds_by_domain.items():
        available = set(str(key) for key in seed_keys)
        for node in nodes:
            reads_by_domain = node.get("in_by_domain", {})
            reads = reads_by_domain.get(domain, node.get("in", []))
            for read in reads:
                if not _provided(str(read), available):
                    prefix = f"{domain}: " if multiple_domains else ""
                    problems.append(
                        f"{prefix}stage {node['i']} {node['t']}: reads "
                        f"{read!r} before any seed or writer provides it"
                    )
            available.update(str(key) for key in node.get("out", []))

    writer_by_key: dict[str, int] = {}
    for node in nodes:
        for raw_key in node.get("out", []):
            key = str(raw_key)
            # A wildcard is a family declaration, not a claim to every concrete
            # suffix. Duplicate concrete keys remain unambiguous lint failures.
            if key.endswith("*"):
                continue
            if key in writer_by_key:
                first = writer_by_key[key]
                problems.append(
                    f"duplicate writer for {key!r}: stages {first} and {node['i']}"
                )
            else:
                writer_by_key[key] = int(node["i"])

    cycle = _cycle(graph.get("edges", []), len(nodes))
    if cycle:
        labels = " -> ".join(f"{index}:{nodes[index]['t']}" for index in cycle)
        problems.append(f"dependency cycle: {labels}")
    return list(dict.fromkeys(problems))


def _graph_key(path: Path, mode: str) -> str:
    return f"{path.stem} [{mode}]"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_json", type=Path)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--mode", choices=(*_MODES, "both"), default="both")
    parser.add_argument(
        "--lint",
        action="store_true",
        help="print successful lint summaries (lint failures are always fatal)",
    )
    args = parser.parse_args(argv)

    modes = _MODES if args.mode == "both" else (args.mode,)
    graphs: dict[str, Any] = {}
    failures: list[str] = []
    for config_path in args.configs:
        selected_path = config_path.resolve()
        try:
            config, sources = _load_selected_config(selected_path)
            model_config = _locate_model_config(config)
            stage_configs, stages = _instantiate_stages(model_config)
            for mode in modes:
                key = _graph_key(selected_path, mode)
                if key in graphs:
                    raise ValueError(
                        f"duplicate graph name {key!r}; config basenames must be unique"
                    )
                graph = _build_graph_from_loaded(
                    selected_path=selected_path,
                    config=config,
                    sources=sources,
                    model_config=model_config,
                    stage_configs=stage_configs,
                    stages=stages,
                    mode=mode,
                )
                graphs[key] = graph
                print(
                    f"  {key:48s} {len(graph['nodes']):2d} nodes "
                    f"{len(graph['edges']):3d} edges"
                )
                if graph["lint"]:
                    for problem in graph["lint"]:
                        print(f"      ! {problem}", file=sys.stderr)
                    failures.extend(f"{key}: {problem}" for problem in graph["lint"])
                elif args.lint:
                    print("      lint: clean")
        except Exception as exc:
            message = f"{selected_path}: FAILED: {type(exc).__name__}: {exc}"
            print(message, file=sys.stderr)
            failures.append(message)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(graphs, separators=(",", ":"), ensure_ascii=False)
    args.out_json.write_text(payload)
    print(f"wrote {args.out_json} ({args.out_json.stat().st_size} bytes)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
