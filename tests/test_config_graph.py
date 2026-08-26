from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "config_graph.py"
_SPEC = importlib.util.spec_from_file_location("config_graph_under_test", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
config_graph = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(config_graph)


def _write_model(path: Path, *, long_note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
robomimic_model:
  _target_: egomimic.pipeline.algo.PipelineAlgo
  domains: [pushshapes_sim_chain_gripper]
  stages:
  - _target_: egomimic.pipeline.stages_sampler.FusedObsEncoder
    n_obs_steps: 1
    encoder:
      _target_: egomimic.pipeline.stages_sampler.DPStyleObsEncoder
      obs_specs:
        state_agent_obj:
          input_dim: 3
          input_slice: [0, 3]
          note: {json.dumps(long_note)}
      img_encoders:
        front_img_1:
          _target_: torch.nn.Identity
  - _target_: egomimic.pipeline.stages_sampler.GaussianLatentNoise
    action_horizon: 2
    latent_dim: 4
""".lstrip()
    )


def test_mode_contracts_and_nested_hydra_are_preserved(tmp_path: Path) -> None:
    model = tmp_path / "model.yaml"
    note = "nested-component-value-" * 40
    _write_model(model, long_note=note)

    train = config_graph.build_graph(model, mode="train")
    rollout = config_graph.build_graph(model, mode="rollout")

    assert train["lint"] == []
    assert rollout["lint"] == []
    assert train["nodes"][0]["declared_in"] == [
        "obs/*",
        "embodiment",
        "actions",
    ]
    assert "actions" in train["nodes"][0]["in"]
    assert "target" in train["nodes"][0]["out"]
    assert "actions" not in rollout["nodes"][0]["in"]
    assert "target" not in rollout["nodes"][0]["out"]
    assert rollout["seed_keys"] == [
        "embodiment",
        "obs/front_img_1",
        "obs/state_agent_obj",
        "rollout_t",
    ]

    encoder = train["nodes"][0]["p"]["encoder"]
    assert isinstance(encoder, dict)
    assert encoder["_target_"].endswith("DPStyleObsEncoder")
    assert encoder["obs_specs"]["state_agent_obj"]["note"] == note
    assert len(encoder["obs_specs"]["state_agent_obj"]["note"]) > 400


def test_experiment_defaults_select_dataset_seed_keys(tmp_path: Path) -> None:
    root = tmp_path / "hydra_configs"
    model = root / "model" / "bf" / "selected.yaml"
    data = root / "data" / "pusht" / "selected.yaml"
    experiment = root / "experiment" / "pusht" / "selected.yaml"
    _write_model(model)
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(
        """
train_datasets:
  pushshapes_sim_chain_gripper:
    resolver:
      embodiment_override: pushshapes_sim_chain_gripper
      key_map:
        _target_: egomimic.rldb.embodiment.pushshapes.get_keymap_hpt
        action_horizon: 2
""".lstrip()
    )
    experiment.parent.mkdir(parents=True, exist_ok=True)
    experiment.write_text(
        """
defaults:
  - override /model: bf/selected
  - override /data: pusht/selected
name: graph-test
""".lstrip()
    )

    graph = config_graph.build_graph(experiment, mode="train")

    assert graph["lint"] == []
    assert graph["seed_key_source"] == "dataset-keymap"
    assert graph["seed_keys"] == [
        "actions",
        "embodiment",
        "obs/front_img_1",
        "obs/state_agent_obj",
    ]
    assert graph["component_sources"]["model"] == str(model)
    assert graph["component_sources"]["data"] == str(data)


def test_lint_reports_unresolved_duplicate_writer_and_cycle() -> None:
    graph = {
        "nodes": [
            {"i": 0, "t": "First", "in": ["b"], "out": ["a"]},
            {"i": 1, "t": "Second", "in": ["a"], "out": ["b", "a"]},
        ],
        "edges": [
            {"a": 1, "b": 0, "k": "b", "s": "shared"},
            {"a": 0, "b": 1, "k": "a", "s": "shared"},
        ],
        "seed_keys": [],
    }

    problems = config_graph.lint(graph)

    assert any("reads 'b' before" in problem for problem in problems)
    assert any("duplicate writer for 'a'" in problem for problem in problems)
    assert any("dependency cycle" in problem for problem in problems)


def test_cli_emits_separate_entries_for_both_modes(tmp_path: Path) -> None:
    model = tmp_path / "selected.yaml"
    output = tmp_path / "graph.json"
    _write_model(model)

    result = config_graph.main([str(output), str(model), "--mode", "both", "--lint"])
    payload = json.loads(output.read_text())

    assert result == 0
    assert set(payload) == {"selected [train]", "selected [rollout]"}
    assert payload["selected [train]"]["mode"] == "train"
    assert payload["selected [rollout]"]["mode"] == "rollout"


def test_cli_fails_nonzero_when_lint_fails(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    output = tmp_path / "bad-graph.json"
    bad.write_text(
        """
robomimic_model:
  _target_: egomimic.pipeline.algo.PipelineAlgo
  domains: [pushshapes_sim_chain_gripper]
  stages:
  - _target_: egomimic.pipeline.stages_sampler.GaussianLatentNoise
    action_horizon: 2
    latent_dim: 4
""".lstrip()
    )

    result = config_graph.main([str(output), str(bad), "--mode", "train"])
    payload = json.loads(output.read_text())

    assert result == 1
    assert any(
        "reads 'condition' before" in item for item in payload["bad [train]"]["lint"]
    )


def test_rollout_omits_train_only_stages(tmp_path: Path) -> None:
    model = tmp_path / "train-only.yaml"
    model.write_text(
        """
robomimic_model:
  _target_: egomimic.pipeline.algo.PipelineAlgo
  domains: [pushshapes_sim_chain_gripper]
  stages:
  - _target_: egomimic.pipeline.stages_sampler.NativeActionMSELoss
""".lstrip()
    )

    graph = config_graph.build_graph(model, mode="rollout")

    assert graph["nodes"] == []
    assert graph["lint"] == []
    assert graph["skipped_stages"][0]["reason"] == "train-only"
