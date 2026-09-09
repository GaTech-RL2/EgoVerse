from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from egomimic.pipeline.stages_sampler import GaussianLatentNoise

_TOOL = Path(__file__).resolve().parents[1] / "tools/config_graph.py"
_SPEC = importlib.util.spec_from_file_location("config_graph_under_test", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
config_graph = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(config_graph)


def _model_yaml(note: str = "") -> str:
    return f"""
_target_: egomimic.pl_utils.pl_model.ModelWrapper
pipeline:
  _target_: egomimic.pipeline.algo.PipelineAlgo
  stages:
    - _target_: egomimic.pipeline.stages_sampler.KeyedFeatureProjection
      selector_key: route
      input_key: features
      output_key: condition
      output_dim: 3
      selector_aliases: {{1: image}}
      projections:
        image:
          input_dim: 2
          hidden_dim: 4
          semantic: {json.dumps(note)}
    - _target_: egomimic.pipeline.stages_sampler.GaussianLatentNoise
      num_tokens: 4
      latent_dim: 3
    - _target_: egomimic.pipeline.stages_io.ActionTargetBuilder
""".lstrip()


def _data_yaml(*, second_source_has_route: bool = True) -> str:
    second_batch_keys = "[route]" if second_source_has_route else "[]"
    return f"""
train_datasets:
  opaque/source-A:
    batch_keys: [route]
    key_map: &keymap
      features: {{zarr_key: observations.features, key_type: proprio_keys}}
      actions: {{zarr_key: actions, key_type: action_keys}}
  "42":
    batch_keys: {second_batch_keys}
    key_map:
      <<: *keymap
""".lstrip()


def _write_full_config(
    path: Path, *, note: str = "", second_source_has_route: bool = True
) -> None:
    model = "\n".join(f"  {line}" for line in _model_yaml(note).splitlines())
    data = "\n".join(
        f"  {line}"
        for line in _data_yaml(
            second_source_has_route=second_source_has_route
        ).splitlines()
    )
    path.write_text(f"model:\n{model}\ndata:\n{data}\n")


def test_clean_pipeline_contracts_and_nested_params_are_preserved(
    tmp_path: Path, monkeypatch
) -> None:
    selected = tmp_path / "selected.yaml"
    note = "nested-stage-parameter-" * 30
    _write_full_config(selected, note=note)
    calls = []
    original = GaussianLatentNoise.contract

    def record_contract(self, mode="train"):
        calls.append(mode)
        return original(self, mode)

    monkeypatch.setattr(GaussianLatentNoise, "contract", record_contract)
    train = config_graph.build_graph(selected, mode="train")
    inference = config_graph.build_graph(selected, mode="inference")

    assert calls == ["train", "inference"]
    assert train["lint"] == inference["lint"] == []
    assert train["seed_keys_by_source"] == {
        "opaque/source-A": ["actions", "features", "route"],
        "42": ["actions", "features", "route"],
    }
    assert inference["seed_keys"] == ["features", "route"]
    assert [node["t"] for node in inference["nodes"]] == [
        "KeyedFeatureProjection",
        "GaussianLatentNoise",
    ]
    assert inference["skipped_stages"] == [
        {
            "source_i": 2,
            "t": "ActionTargetBuilder",
            "target": "egomimic.pipeline.stages_io.ActionTargetBuilder",
            "reason": "train-only",
        }
    ]
    assert train["edges"] == [
        {"a": 0, "b": 1, "k": "condition", "s": "shared"}
    ]
    params = train["nodes"][0]["p"]
    assert params["projections"]["image"]["semantic"] == note
    assert len(params["projections"]["image"]["semantic"]) > 400


def test_source_names_are_opaque_and_missing_keymap_inputs_are_linted(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "missing.yaml"
    _write_full_config(selected, second_source_has_route=False)

    graph = config_graph.build_graph(selected)

    assert set(graph["seed_keys_by_source"]) == {"opaque/source-A", "42"}
    assert any(
        "source '42'" in problem and "reads 'route'" in problem
        for problem in graph["lint"]
    )
    assert not any(
        "source 'opaque/source-A'" in problem and "reads 'route'" in problem
        for problem in graph["lint"]
    )


def test_hydra_experiment_accepts_exact_model_override(tmp_path: Path) -> None:
    root = tmp_path / "hydra_configs"
    (root / "model").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "experiment").mkdir()
    (root / "train_zarr_cartesian.yaml").write_text(
        "defaults:\n  - model: null\n  - data: null\n  - _self_\n"
    )
    (root / "model/selected.yaml").write_text(_model_yaml())
    (root / "data/selected.yaml").write_text(_data_yaml())
    experiment = root / "experiment/selected.yaml"
    experiment.write_text(
        "# @package _global_\n"
        "defaults:\n"
        "  - override /data: selected\n"
        "  - _self_\n"
    )

    overrides = ["model=selected", "++run_tag=exact-row"]
    graph = config_graph.build_graph(experiment, overrides=overrides)

    assert graph["lint"] == []
    assert graph["seed_key_source"] == "selected-data-keymap"
    assert graph["component_sources"]["config_root"] == str(root)
    assert graph["component_sources"]["overrides"] == overrides


def test_legacy_model_key_is_not_considered(tmp_path: Path) -> None:
    selected = tmp_path / "legacy.yaml"
    selected.write_text(
        "robomimic_model:\n"
        "  stages:\n"
        "    - _target_: egomimic.pipeline.stages_io.ActionTargetBuilder\n"
    )

    with pytest.raises(ValueError, match=r"model\.pipeline\.stages"):
        config_graph.build_graph(selected)


def test_lint_reports_duplicate_concrete_writers_and_cycles() -> None:
    graph = {
        "nodes": [
            {"i": 0, "t": "First", "in": ["b"], "out": ["a"]},
            {"i": 1, "t": "Second", "in": ["a"], "out": ["b", "a"]},
        ],
        "edges": [
            {"a": 1, "b": 0, "k": "b", "s": "shared"},
            {"a": 0, "b": 1, "k": "a", "s": "shared"},
        ],
        "seed_keys_by_source": {"anything": []},
    }

    problems = config_graph.lint(graph)

    assert any("reads 'b' before" in problem for problem in problems)
    assert any("duplicate writer for 'a'" in problem for problem in problems)
    assert any("dependency cycle" in problem for problem in problems)


def test_cli_emits_renderer_compatible_both_mode_json(
    tmp_path: Path, monkeypatch
) -> None:
    selected = tmp_path / "selected.yaml"
    output = tmp_path / "graph.json"
    _write_full_config(selected)

    instantiations = 0
    original = config_graph._stages

    def count_instantiations(pipeline_config):
        nonlocal instantiations
        instantiations += 1
        return original(pipeline_config)

    monkeypatch.setattr(config_graph, "_stages", count_instantiations)
    result = config_graph.main(
        [str(output), str(selected), "--mode", "both", "--lint"]
    )
    payload = json.loads(output.read_text())

    assert result == 0
    assert instantiations == 1
    assert set(payload) == {"selected [train]", "selected [inference]"}
    for graph in payload.values():
        assert isinstance(graph["nodes"], list)
        assert isinstance(graph["edges"], list)
        assert all({"i", "t", "in", "out", "p"} <= set(node) for node in graph["nodes"])
        assert all({"a", "b", "k", "s"} <= set(edge) for edge in graph["edges"])
