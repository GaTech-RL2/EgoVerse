"""Tests for independent local-velocity ARC graph nodes."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from egomimic.pipeline.core import Pipeline, Stage
from egomimic.pipeline.stages_arc import ArcDetokenizeStage, ArcTokenizeStage
from egomimic.rldb.zarr.planar_arc import PLANAR_ACTION_DIM, TokenizeUSocketArcVelocity

_REPO = Path(__file__).resolve().parents[1]
_EXPERIMENT = (
    _REPO / "egomimic/hydra_configs/experiment/pusht/planar_v2_usocket_arc_graph_tok.yaml"
)
_SPEC = importlib.util.spec_from_file_location(
    "config_graph_arc", _REPO / "tools/config_graph.py"
)
config_graph = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(config_graph)

_M = 8
_DT = 1.0 / 30.0


def _actions(steps=20):
    result = torch.zeros(2, steps, 3, dtype=torch.float64)
    result[:, :, 0] = torch.linspace(0.0, 19.0, steps)
    result[:, :, 2] = torch.linspace(0.0, 0.95, steps)
    return result


def _tokenizer(**kwargs):
    params = dict(
        min_distance_unit=50.0,
        rotation_distance_unit=2.0,
        resampled_vector_length=_M,
        dt=_DT,
    )
    params.update(kwargs)
    return ArcTokenizeStage(**params)


def _detokenizer(**kwargs):
    params = dict(resampled_vector_length=_M, action_horizon=16, dt=_DT)
    params.update(kwargs)
    return ArcDetokenizeStage(**params)


def test_stage_mode_contracts():
    assert _tokenizer().contract("train") == (("actions",), ("target",))
    _, excluded = Pipeline([_tokenizer()]).plan(["actions"], mode="inference")
    assert excluded[0][1] == ["<train-only>"]
    _, excluded = Pipeline([_detokenizer()]).plan(["pred_action"], mode="train")
    assert excluded[0][1] == ["<inference-only>"]


def test_inference_only_stage_does_not_block_train_but_missing_read_does():
    assert Pipeline([_detokenizer()]).execute({"unrelated": 1}, mode="train") == {
        "unrelated": 1
    }

    class NeedsMissing(Stage):
        reads = ("absent",)
        writes = ("out",)

        def forward(self, batch):
            return batch

    with pytest.raises(RuntimeError, match="blocked stages"):
        Pipeline([NeedsMissing()]).execute({"present": 1}, mode="train")


def test_tokenize_emits_two_m_row_streams_and_consumes_actions():
    actions = _actions()
    out = _tokenizer().forward({"actions": actions})
    assert out["target"].shape == (2, 2 * _M, PLANAR_ACTION_DIM)
    assert "actions" not in out
    assert out["target"].dtype == actions.dtype
    torch.testing.assert_close(
        out["target"][:, :_M, 2:4], torch.zeros(2, _M, 2, dtype=actions.dtype)
    )
    torch.testing.assert_close(
        out["target"][:, _M:, :2], torch.zeros(2, _M, 2, dtype=actions.dtype)
    )


def test_graph_and_loader_tokenizers_match_exactly():
    actions = _actions()
    graph = _tokenizer().forward({"actions": actions})["target"][0].numpy()
    loader = TokenizeUSocketArcVelocity(
        min_distance_unit=50.0,
        rotation_distance_unit=2.0,
        resampled_vector_length=_M,
        dt=_DT,
    ).tokenize(actions[0].numpy())
    np.testing.assert_allclose(graph, loader)


def test_detokenizer_recovers_constant_translation_and_angular_rates():
    token = _tokenizer().forward({"actions": _actions()})["target"]
    native = _detokenizer(action_horizon=16).forward({"pred_action": token})[
        "pred_action_native"
    ]
    expected = _actions()[:, :16]
    torch.testing.assert_close(native[..., :2], expected[..., :2], atol=2e-6, rtol=0)
    # Compare wrapped angular error.
    error = torch.atan2(
        torch.sin(native[..., 2] - expected[..., 2]),
        torch.cos(native[..., 2] - expected[..., 2]),
    )
    torch.testing.assert_close(error, torch.zeros_like(error), atol=5e-5, rtol=0)


def test_detokenizer_uses_local_rates_for_variable_speed():
    # Geometry is uniform in arc, but interval rates differ. A mean-speed
    # decoder would return x=2 at t=1; local speeds recover x=1.
    token = torch.zeros(1, 2 * _M, 5, dtype=torch.float64)
    token[0, :_M, 0] = torch.linspace(0.0, 7.0, _M)
    token[0, :_M, 4] = torch.tensor([1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    token[0, _M:, 2] = 1.0
    native = _detokenizer(action_horizon=3, dt=1.0).forward({"pred_action": token})[
        "pred_action_native"
    ]
    torch.testing.assert_close(
        native[0, :, 0], torch.tensor([0.0, 1.0, 4.0], dtype=token.dtype)
    )


def test_translation_and_rotation_replay_on_independent_clocks():
    token = torch.zeros(1, 2 * _M, 5, dtype=torch.float64)
    token[0, :_M, 0] = torch.linspace(0.0, 7.0, _M)
    token[0, :_M, 4] = 1.0
    theta = torch.linspace(0.0, 0.7, _M, dtype=torch.float64)
    token[0, _M:, 2] = torch.cos(theta)
    token[0, _M:, 3] = torch.sin(theta)
    token[0, _M:, 4] = 0.2
    native = _detokenizer(action_horizon=4, dt=1.0).forward({"pred_action": token})[
        "pred_action_native"
    ]
    torch.testing.assert_close(
        native[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=token.dtype)
    )
    torch.testing.assert_close(
        native[0, :, 2], torch.tensor([0.0, 0.2, 0.4, 0.6], dtype=token.dtype),
        atol=2e-4,
        rtol=0,
    )


def test_negative_angular_velocity_replays_decreasing_angle():
    action = torch.zeros(1, 8, 3, dtype=torch.float64)
    action[0, :, 2] = torch.linspace(0.4, -0.3, 8)
    token = _tokenizer().forward({"actions": action})["target"]
    assert torch.all(token[0, _M:, 4] < 0)
    native = _detokenizer(action_horizon=8).forward({"pred_action": token})[
        "pred_action_native"
    ]
    torch.testing.assert_close(native[0, :, 2], action[0, :, 2], atol=2e-6, rtol=0)


def test_pi_seam_and_zero_motion_are_stable():
    action = torch.zeros(1, 8, 3, dtype=torch.float64)
    action[0, :, 2] = torch.linspace(math.pi - 0.2, math.pi + 0.2, 8)
    token = _tokenizer().forward({"actions": action})["target"]
    native = _detokenizer(action_horizon=8).forward({"pred_action": token})[
        "pred_action_native"
    ]
    assert torch.all(native[0, :, 2].abs() > math.pi - 0.3)

    still = torch.tensor([[[4.0, 7.0, 0.5]]], dtype=torch.float64).repeat(1, 8, 1)
    token = _tokenizer().forward({"actions": still})["target"]
    native = _detokenizer(action_horizon=8).forward({"pred_action": token})[
        "pred_action_native"
    ]
    torch.testing.assert_close(native, still)


def test_detokenizer_shape_validation_and_logs():
    with pytest.raises(ValueError, match="ArcDetokenizeStage expects"):
        _detokenizer().forward({"pred_action": torch.zeros(2, _M, 5)})
    out = _detokenizer().forward(
        {"pred_action": _tokenizer().forward({"actions": _actions()})["target"]}
    )
    assert {
        "log/ArcLinearSpeed",
        "log/ArcAngularVelocityAbs",
        "log/ArcTranslationDistance",
        "log/ArcRotationDistance",
    } <= out.keys()


@pytest.mark.parametrize(
    "kwargs",
    [{"resampled_vector_length": 1}, {"action_horizon": 0}, {"native_action_dim": 7}],
)
def test_detokenizer_rejects_impossible_settings(kwargs):
    with pytest.raises(ValueError):
        _detokenizer(**kwargs)


@pytest.mark.parametrize("mode", ["train", "inference"])
def test_arc_graph_experiment_lints_clean(mode):
    assert config_graph.build_graph(_EXPERIMENT, mode=mode)["lint"] == []


def test_graph_has_one_mode_specific_arc_node_and_one_target_writer():
    train = config_graph.build_graph(_EXPERIMENT, mode="train")
    inference = config_graph.build_graph(_EXPERIMENT, mode="inference")
    train_names = [node["t"] for node in train["nodes"]]
    inference_names = [node["t"] for node in inference["nodes"]]
    assert "ArcTokenizeStage" in train_names and "ArcDetokenizeStage" not in train_names
    assert "ArcDetokenizeStage" in inference_names and "ArcTokenizeStage" not in inference_names
    assert [n["t"] for n in train["nodes"] if "target" in n["out"]] == [
        "ArcTokenizeStage"
    ]


def test_config_uses_two_m_rows_and_shared_dt():
    from omegaconf import OmegaConf

    config, _ = config_graph._load_selected(_EXPERIMENT)
    assert OmegaConf.select(config, "planar.arc_token_rows") == 2 * OmegaConf.select(
        config, "planar.arc_waypoints"
    )
    train = config_graph.build_graph(_EXPERIMENT, mode="train")
    inference = config_graph.build_graph(_EXPERIMENT, mode="inference")
    tokenize = next(n for n in train["nodes"] if n["t"] == "ArcTokenizeStage")
    decode = next(n for n in inference["nodes"] if n["t"] == "ArcDetokenizeStage")
    assert tokenize["p"]["dt"] == decode["p"]["dt"]


def test_loader_hands_raw_window_to_graph_without_arc_transform():
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config, _ = config_graph._load_selected(_EXPERIMENT)
    raw = OmegaConf.select(config, "planar.raw_action_horizon")
    for dataset in OmegaConf.select(config, "data.train_datasets").values():
        assert OmegaConf.select(dataset, "resolver.key_map.action_horizon") == raw
        transforms = instantiate(OmegaConf.select(dataset, "resolver.transform_list"))
        assert not any(isinstance(t, TokenizeUSocketArcVelocity) for t in transforms)
    assert OmegaConf.select(config, "planar.arc_dt") == pytest.approx(1 / 30)
