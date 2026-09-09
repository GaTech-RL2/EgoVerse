#!/usr/bin/env python3
"""Build and lint renderer-compatible graphs from clean Pipeline configs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

_MODES = ("train", "inference")
_ZARR_DATASET_TARGET = (
    "egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver"
)
# ZarrDataset adds these fields after reading and transforming its key map.
_ZARR_RUNTIME_KEYS = frozenset({"embodiment", "episode_hash", "intrinsics"})


def _plain(value: Any, *, resolve: bool = False) -> Any:
    """Return ordinary JSON-shaped containers without shortening nested config."""
    from omegaconf import OmegaConf

    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=resolve)
    if isinstance(value, Mapping):
        return {str(key): _plain(item, resolve=resolve) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item, resolve=resolve) for item in value]
    return value


def hyperparams(stage_config: Any) -> dict[str, Any]:
    """Preserve the complete constructor tree for the renderer sidebar."""
    value = _plain(stage_config, resolve=True)
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items() if key != "_target_"}


def _matches(left: str, right: str) -> bool:
    """Return whether two exact-or-trailing-wildcard key declarations overlap."""
    if left == right:
        return True
    if left.endswith("*") and right.startswith(left[:-1]):
        return True
    return right.endswith("*") and left.startswith(right[:-1])


def _provided(read: str, available: Iterable[str]) -> bool:
    return any(_matches(read, str(candidate)) for candidate in available)


def _config_root(path: Path) -> Path | None:
    for candidate in path.parents:
        if (
            (candidate / "train_zarr_cartesian.yaml").is_file()
            and (candidate / "model").is_dir()
            and (candidate / "data").is_dir()
        ):
            return candidate
    return None


def _compose_selected(path: Path, root: Path, overrides: Sequence[str]):
    """Compose a selected Hydra model or experiment fragment when necessary."""
    from hydra import compose, initialize_config_dir

    relative = path.relative_to(root).with_suffix("")
    group, *parts = relative.parts
    name = "/".join(parts)
    if group == "experiment":
        selected_overrides = [f"+experiment={name}", "++paths.root_dir=."]
    elif group == "model":
        selected_overrides = [f"model={name}", "++paths.root_dir=."]
    else:
        raise ValueError(
            "selected YAML must contain model.pipeline.stages or be a Hydra "
            "model/experiment config"
        )
    selected_overrides.extend(overrides)
    with initialize_config_dir(version_base=None, config_dir=str(root.resolve())):
        return compose(
            config_name="train_zarr_cartesian", overrides=selected_overrides
        )


def _load_selected(
    path: Path, overrides: Sequence[str] = ()
) -> tuple[Any, dict[str, Any]]:
    from omegaconf import OmegaConf

    raw = OmegaConf.load(path)
    root = _config_root(path)
    provenance: dict[str, Any] = {
        "selected": str(path),
        "overrides": list(overrides),
    }
    if overrides:
        if root is None:
            raise ValueError("Hydra overrides require a config under a config root")
        provenance["config_root"] = str(root)
        return _compose_selected(path, root, overrides), provenance
    if OmegaConf.select(raw, "model.pipeline.stages") is not None:
        return raw, provenance
    if OmegaConf.select(raw, "pipeline.stages") is not None:
        return OmegaConf.create({"model": raw}), provenance
    if root is None:
        return raw, provenance
    provenance["config_root"] = str(root)
    return _compose_selected(path, root, ()), provenance


def _pipeline_config(config: Any):
    from omegaconf import OmegaConf

    pipeline = OmegaConf.select(config, "model.pipeline")
    if pipeline is None or OmegaConf.select(pipeline, "stages") is None:
        raise ValueError("selected config has no model.pipeline.stages")
    return pipeline


def _stages(pipeline_config: Any) -> tuple[list[Any], list[Any]]:
    from hydra.utils import instantiate

    from egomimic.pipeline.core import Stage

    configs = list(pipeline_config.stages)
    stages = [instantiate(config) for config in configs]
    for index, stage in enumerate(stages):
        if not isinstance(stage, Stage):
            raise TypeError(
                f"configured stage {index} is {type(stage).__name__}, expected Stage"
            )
    return configs, stages


def _nodes(
    stage_configs: Sequence[Any], stages: Sequence[Any], mode: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for source_index, (config, stage) in enumerate(zip(stage_configs, stages)):
        raw = _plain(config)
        target = str(raw.get("_target_", "")) if isinstance(raw, Mapping) else ""
        name = target.rsplit(".", 1)[-1] or type(stage).__name__
        restricted = None
        if mode == "inference" and bool(getattr(stage, "train_only", False)):
            restricted = "train-only"
        elif mode == "train" and bool(getattr(stage, "inference_only", False)):
            restricted = "inference-only"
        if restricted is not None:
            skipped.append(
                {
                    "source_i": source_index,
                    "t": name,
                    "target": target,
                    "reason": restricted,
                }
            )
            continue
        reads, writes = stage.contract(mode)
        reads = list(dict.fromkeys(str(key) for key in reads))
        writes = list(dict.fromkeys(str(key) for key in writes))
        nodes.append(
            {
                "i": len(nodes),
                "source_i": source_index,
                "t": name,
                "target": target,
                "sec": "PIPELINE",
                "stream": "shared",
                "depth": 0,
                "mode": mode,
                "in": reads,
                "out": writes,
                "declared_in": reads,
                "declared_out": writes,
                "p": hyperparams(config),
            }
        )
    return nodes, skipped


def _external_reads(nodes: Sequence[Mapping[str, Any]]) -> list[str]:
    writes = [str(key) for node in nodes for key in node.get("out", ())]
    return list(
        dict.fromkeys(
            str(read)
            for node in nodes
            for read in node.get("in", ())
            if not any(_matches(str(read), write) for write in writes)
        )
    )


def _select(value: Any, *paths: str):
    from omegaconf import OmegaConf

    for path in paths:
        selected = OmegaConf.select(value, path)
        if selected is not None:
            return selected
    return None


def _declared_batch_keys(dataset_config: Any) -> set[str]:
    keys: set[str] = set()
    for path in (
        "provided_keys",
        "batch_keys",
        "resolver.provided_keys",
        "resolver.batch_keys",
    ):
        value = _plain(_select(dataset_config, path))
        if isinstance(value, list):
            keys.update(str(key) for key in value)
    target = str(_select(dataset_config, "_target_") or "")
    if target == _ZARR_DATASET_TARGET:
        keys.update(_ZARR_RUNTIME_KEYS)
    return keys


def _seed_keys(
    config: Any, external_reads: Sequence[str]
) -> tuple[dict[str, list[str]], str, list[str], dict[str, Any]]:
    """Resolve each opaque loader source's batch-key inventory from its key map."""
    from hydra.utils import instantiate

    data = _select(config, "data")
    if data is None and _select(config, "train_datasets", "valid_datasets") is not None:
        data = config
    datasets = (
        None
        if data is None
        else _select(data, "train_datasets", "valid_datasets")
    )
    if datasets is None:
        return {"<unselected>": []}, "no-selected-data", [], {}

    seeds_by_source: dict[str, list[str]] = {}
    details: dict[str, Any] = {}
    warnings: list[str] = []
    for raw_source, dataset_config in datasets.items():
        source = str(raw_source)
        keymap_config = _select(dataset_config, "resolver.key_map", "key_map")
        inventory = _declared_batch_keys(dataset_config)
        if keymap_config is None:
            warnings.append(f"source {source!r}: no selected key_map")
        else:
            try:
                keymap = instantiate(keymap_config)
                if not isinstance(keymap, Mapping):
                    raise TypeError("key_map did not return a mapping")
                inventory.update(str(key) for key in keymap)
            except Exception as exc:
                warnings.append(
                    f"source {source!r}: key_map failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        seeds = sorted(
            key
            for key in inventory
            if any(_matches(key, read) for read in external_reads)
        )
        seeds_by_source[source] = seeds
        details[source] = {
            "available_batch_keys": sorted(inventory),
            "external_seed_keys": seeds,
        }
    return seeds_by_source, "selected-data-keymap", warnings, details


def _edges(
    nodes: Sequence[Mapping[str, Any]], seeds_by_source: Mapping[str, Sequence[str]]
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for node in nodes:
        index = int(node["i"])
        for read in node.get("in", ()):
            earlier = [
                int(writer["i"])
                for writer in nodes[:index]
                if any(_matches(str(read), str(key)) for key in writer.get("out", ()))
            ]
            if earlier:
                producer = max(earlier)
            elif all(
                _provided(str(read), seeds) for seeds in seeds_by_source.values()
            ):
                continue
            else:
                later = [
                    int(writer["i"])
                    for writer in nodes[index + 1 :]
                    if any(
                        _matches(str(read), str(key))
                        for key in writer.get("out", ())
                    )
                ]
                if not later:
                    continue
                producer = min(later)
            edges.append(
                {"a": producer, "b": index, "k": str(read), "s": "shared"}
            )
    return edges


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
    """Report missing source inputs, duplicate concrete writers, and cycles."""
    problems: list[str] = []
    nodes = list(graph.get("nodes", ()))
    sources = graph.get("seed_keys_by_source") or {"<unselected>": []}
    for source, seeds in sources.items():
        available = {str(key) for key in seeds}
        for node in nodes:
            missing = [
                str(read)
                for read in node.get("in", ())
                if not _provided(str(read), available)
            ]
            for read in missing:
                problems.append(
                    f"source {source!r}, stage {node['i']} {node['t']}: "
                    f"reads {read!r} before any seed or writer provides it"
                )
            if not missing:
                available.update(str(key) for key in node.get("out", ()))

    writers: dict[str, int] = {}
    for node in nodes:
        for raw_key in node.get("out", ()):
            key = str(raw_key)
            if key.endswith("*"):
                continue
            if key in writers:
                problems.append(
                    f"duplicate writer for {key!r}: stages "
                    f"{writers[key]} and {node['i']}"
                )
            else:
                writers[key] = int(node["i"])

    cycle = _cycle(graph.get("edges", ()), len(nodes))
    if cycle:
        labels = " -> ".join(f"{index}:{nodes[index]['t']}" for index in cycle)
        problems.append(f"dependency cycle: {labels}")
    return list(dict.fromkeys(problems))


def _build_graph(
    *,
    selected: Path,
    config: Any,
    component_sources: Mapping[str, Any],
    stage_configs: Sequence[Any],
    stages: Sequence[Any],
    mode: str,
) -> dict[str, Any]:
    if mode not in _MODES:
        raise ValueError(f"mode must be train|inference, got {mode!r}")
    nodes, skipped = _nodes(stage_configs, stages, mode)
    external_reads = _external_reads(nodes)
    seeds_by_source, seed_source, warnings, details = _seed_keys(
        config, external_reads
    )
    graph: dict[str, Any] = {
        "nodes": nodes,
        "edges": _edges(nodes, seeds_by_source),
        "source": str(selected),
        "source_name": selected.name,
        "component_sources": component_sources,
        "mode": mode,
        "external_reads": external_reads,
        "seed_keys": sorted({key for keys in seeds_by_source.values() for key in keys}),
        "seed_keys_by_source": seeds_by_source,
        "seed_key_source": seed_source,
        "seed_key_details": details,
        "seed_warnings": warnings,
        "skipped_stages": skipped,
    }
    graph["lint"] = lint(graph)
    return graph


def build_graph(
    path: str | Path,
    mode: str = "train",
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Instantiate one selected config and return its mode-specific graph."""
    selected = Path(path).resolve()
    config, component_sources = _load_selected(selected, overrides)
    stage_configs, stages = _stages(_pipeline_config(config))
    return _build_graph(
        selected=selected,
        config=config,
        component_sources=component_sources,
        stage_configs=stage_configs,
        stages=stages,
        mode=mode,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_json", type=Path)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--mode", choices=(*_MODES, "both"), default="both")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="repeatable Hydra override applied after the selected config",
    )
    parser.add_argument("--lint", action="store_true")
    args = parser.parse_args(argv)

    modes = _MODES if args.mode == "both" else (args.mode,)
    graphs: dict[str, Any] = {}
    failed = False
    for config in args.configs:
        try:
            selected = config.resolve()
            loaded, component_sources = _load_selected(selected, args.override)
            stage_configs, stages = _stages(_pipeline_config(loaded))
            for mode in modes:
                key = f"{config.stem} [{mode}]"
                if key in graphs:
                    raise ValueError(f"duplicate graph name {key!r}")
                graph = _build_graph(
                    selected=selected,
                    config=loaded,
                    component_sources=component_sources,
                    stage_configs=stage_configs,
                    stages=stages,
                    mode=mode,
                )
                graphs[key] = graph
                if graph["lint"]:
                    failed = True
                    for problem in graph["lint"]:
                        print(f"{key}: {problem}", file=sys.stderr)
                elif args.lint:
                    print(f"{key}: lint clean")
        except Exception as exc:
            failed = True
            print(
                f"{config}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(graphs, separators=(",", ":"), ensure_ascii=False)
    )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
