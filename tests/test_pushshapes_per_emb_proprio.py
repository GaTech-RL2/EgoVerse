from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Stage
from egomimic.pipeline.pushshapes import (
    ChainModelStateObservationAdapter,
    USocketModelStateObservationAdapter,
)
from egomimic.pipeline.stages_sampler import EmbodimentProprioProjection
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.pushshapes import (
    get_keymap_hpt_per_emb_proprio,
    get_usocket_rotvec_action_state_transform_list,
)
from egomimic.rldb.zarr.action_chunk_transforms import PlanarAgentStateToRotVec4
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

CONFIG_DIR = Path(__file__).parents[1] / "egomimic" / "hydra_configs"
EXPERIMENT = (
    "pusht/pipeline_sampler_usocket_chain_newdata_"
    "cotrain12_per_emb_proprio_h16"
)
U_DOMAIN = "pushshapes_sim_u_socket"
CHAIN_DOMAIN = "pushshapes_sim_chain_gripper"


class _StateCondition(Stage):
    reads = ["obs/state_agent_model", "embodiment"]
    writes = ["condition"]
    rollout_obs_steps = 1

    def forward(self, batch):
        batch["condition"] = batch["obs/state_agent_model"]
        return batch


class _FixedPerDomainPrediction(Stage):
    reads = ["condition", "embodiment"]
    writes = ["pred_action"]

    def forward(self, batch):
        width = 4 if batch["embodiment"] == U_DOMAIN else 6
        batch["pred_action"] = torch.zeros(
            (batch["condition"].shape[0], 16, width),
            device=batch["condition"].device,
        )
        return batch


class _CaptureContextAdapter:
    def __init__(self):
        self.context = None

    def decode(self, actions, context=None):
        self.context = dict(context or {})
        return actions


def _per_emb_norm_stats() -> MultiDataset:
    norm_stats = MultiDataset(
        state={},
        norm_mode="minmax",
        reduce_all_but_last=True,
    )
    u_id = get_embodiment_id(U_DOMAIN)
    chain_id = get_embodiment_id(CHAIN_DOMAIN)
    norm_stats.embodiments = {u_id, chain_id}
    norm_stats.key_types = {
        emb_id: {
            "actions": "action_keys",
            "state_agent_obj": "metadata_keys",
            "state_agent_model": "proprio_keys",
        }
        for emb_id in (u_id, chain_id)
    }
    norm_stats.zarr_keys = {
        emb_id: {
            "actions": "actions",
            "state_agent_obj": "state_agent_obj",
            "state_agent_model": "state_agent_model",
        }
        for emb_id in (u_id, chain_id)
    }
    norm_stats.shapes = {
        u_id: {"actions": (16, 4), "state_agent_model": (4,)},
        chain_id: {"actions": (16, 6), "state_agent_model": (6,)},
    }
    norm_stats.norm_stats = {
        u_id: {
            "actions": {
                "min": np.full(4, -2.0, dtype=np.float32),
                "max": np.full(4, 2.0, dtype=np.float32),
            },
            "state_agent_model": {
                "min": np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32),
                "max": np.array([20.0, 40.0, 1.0, 1.0], dtype=np.float32),
            },
        },
        chain_id: {
            "actions": {
                "min": np.full(6, -2.0, dtype=np.float32),
                "max": np.full(6, 2.0, dtype=np.float32),
            },
            "state_agent_model": {
                "min": np.full(6, -100.0, dtype=np.float32),
                "max": np.full(6, 100.0, dtype=np.float32),
            },
        },
    }
    return norm_stats


def _compose():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={EXPERIMENT}"],
        )


def test_keymap_preserves_native_state_and_adds_model_proprio_alias() -> None:
    keymap = get_keymap_hpt_per_emb_proprio(action_horizon=16)
    assert keymap["state_agent_obj"] == {
        "key_type": "metadata_keys",
        "zarr_key": "observations.state",
    }
    assert keymap["state_agent_model"] == {
        "key_type": "proprio_keys",
        "zarr_key": "observations.state",
    }
    assert keymap["actions"]["horizon"] == 16


@pytest.mark.parametrize("as_torch", [False, True])
def test_usocket_transform_is_exact_state4_and_preserves_native_context(
    as_torch: bool,
) -> None:
    raw_state = np.array(
        [114.5, 97.25, -1.2, 389.75, 301.8, 1.873], dtype=np.float32
    )
    actions = np.tile(np.array([10.0, 20.0, -0.4], dtype=np.float32), (16, 1))
    if as_torch:
        raw_value = torch.from_numpy(raw_state.copy())
        model_value = raw_value.clone()
        action_value = torch.from_numpy(actions.copy())
    else:
        raw_value = raw_state.copy()
        model_value = raw_state.copy()
        action_value = actions.copy()
    sample = {
        "state_agent_obj": raw_value,
        "state_agent_model": model_value,
        "actions": action_value,
    }
    for transform in get_usocket_rotvec_action_state_transform_list():
        sample = transform.transform(sample)

    assert tuple(sample["state_agent_model"].shape) == (4,)
    assert tuple(sample["actions"].shape) == (16, 4)
    expected_pair = [np.cos(-1.2), np.sin(-1.2)]
    if as_torch:
        torch.testing.assert_close(sample["state_agent_obj"], raw_value)
        torch.testing.assert_close(
            sample["state_agent_model"][2:4],
            torch.tensor(expected_pair, dtype=sample["state_agent_model"].dtype),
            atol=1e-6,
            rtol=0,
        )
    else:
        np.testing.assert_array_equal(sample["state_agent_obj"], raw_state)
        np.testing.assert_allclose(
            sample["state_agent_model"][2:4], expected_pair, atol=1e-6
        )


def test_rollout_model_state_adapters_match_training_contract() -> None:
    raw = torch.tensor([[10.0, 20.0, -1.2, 30.0, 40.0, 0.5]])

    u = USocketModelStateObservationAdapter().encode({"state_agent_obj": raw})
    assert u["state_agent_obj"] is raw
    assert u["state_agent_model"].shape == (1, 4)
    torch.testing.assert_close(
        u["state_agent_model"][0, 2:4],
        torch.tensor(
            [np.cos(-1.2), np.sin(-1.2)],
            dtype=u["state_agent_model"].dtype,
        ),
        atol=1e-6,
        rtol=0,
    )

    chain = ChainModelStateObservationAdapter().encode(
        {"state_agent_obj": raw}
    )
    assert chain["state_agent_obj"] is raw
    assert chain["state_agent_model"] is raw
    assert chain["state_agent_model"].shape == (1, 6)


def test_usocket_state_transform_rejects_non_native_source_width() -> None:
    transform = PlanarAgentStateToRotVec4(keys=["state_agent_model"])
    with pytest.raises(ValueError, match="exact source width 6"):
        transform.transform({"state_agent_model": torch.zeros(1, 7)})


def test_rollout_adapts_before_norm_and_preserves_raw_chain_ik_context() -> None:
    norm_stats = _per_emb_norm_stats()
    u_capture = _CaptureContextAdapter()
    chain_capture = _CaptureContextAdapter()
    algo = PipelineAlgo(
        stages=[_StateCondition(), _FixedPerDomainPrediction()],
        norm_stats=norm_stats,
        domains=[U_DOMAIN, CHAIN_DOMAIN],
        ac_keys={U_DOMAIN: "actions", CHAIN_DOMAIN: "actions"},
        rollout_adapters={U_DOMAIN: u_capture, CHAIN_DOMAIN: chain_capture},
        rollout_observation_adapters={
            U_DOMAIN: USocketModelStateObservationAdapter(),
            CHAIN_DOMAIN: ChainModelStateObservationAdapter(),
        },
        action_horizon=16,
        device=torch.device("cpu"),
    )
    u_raw = torch.tensor([[10.0, 20.0, np.pi / 2, 30.0, 40.0, -0.2]])
    chain_raw = torch.tensor([[11.0, 21.0, -0.7, 31.0, 41.0, 0.4]])

    processed = algo.process_batch_for_rollout(
        {
            U_DOMAIN: {"state_agent_obj": u_raw},
            CHAIN_DOMAIN: {"state_agent_obj": chain_raw},
        }
    )
    u_id = get_embodiment_id(U_DOMAIN)
    chain_id = get_embodiment_id(CHAIN_DOMAIN)
    torch.testing.assert_close(processed[u_id]["state_agent_obj"], u_raw)
    torch.testing.assert_close(processed[chain_id]["state_agent_obj"], chain_raw)

    u_model_raw = torch.tensor([[10.0, 20.0, 0.0, 1.0]])
    expected_u = norm_stats.normalize(
        {"state_agent_model": u_model_raw}, u_id
    )["state_agent_model"]
    expected_chain = norm_stats.normalize(
        {"state_agent_model": chain_raw}, chain_id
    )["state_agent_model"]
    torch.testing.assert_close(
        processed[u_id]["state_agent_model"], expected_u, atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        processed[chain_id]["state_agent_model"], expected_chain
    )

    algo.forward_rollout(processed)
    assert u_capture.context is not None
    assert chain_capture.context is not None
    torch.testing.assert_close(u_capture.context["state_agent_obj"], u_raw)
    torch.testing.assert_close(
        chain_capture.context["state_agent_obj"], chain_raw
    )


def _projection() -> EmbodimentProprioProjection:
    return EmbodimentProprioProjection(
        output_dim=64,
        projections={
            U_DOMAIN: {
                "source_dim": 4,
                "hidden_dim": 64,
                "semantic": "x_y_cos_theta_sin_theta",
            },
            CHAIN_DOMAIN: {
                "source_dim": 6,
                "hidden_dim": 64,
                "semantic": "agent_x_y_theta_object_x_y_theta",
            },
        },
    )


@pytest.mark.parametrize(
    ("domain", "width"),
    [(U_DOMAIN, 4), (CHAIN_DOMAIN, 6)],
)
def test_projection_routes_variable_widths_to_shared_64d(
    domain: str, width: int
) -> None:
    projection = _projection()
    value = torch.randn(3, width, requires_grad=True)
    batch = projection(
        {"embodiment": domain, "obs/state_agent_model": value}
    )
    output = batch["obs/proprio_condition"]
    assert output.shape == (3, 64)
    output.square().mean().backward()
    assert value.grad is not None
    own_grad = any(
        parameter.grad is not None
        for parameter in projection.projections[domain].parameters()
    )
    assert own_grad


def test_projection_rejects_missing_usocket_rotvec_transform() -> None:
    with pytest.raises(ValueError, match="expected 4"):
        _projection()(
            {
                "embodiment": U_DOMAIN,
                "obs/state_agent_model": torch.zeros(2, 6),
            }
        )


def test_cotrain_config_has_explicit_projection_node_and_no_hidden384() -> None:
    cfg = _compose()
    model = cfg.model.robomimic_model
    projection, fused, noise, sampler, decoder, _loss = model.stages

    assert projection._target_.endswith("EmbodimentProprioProjection")
    assert projection.output_dim == 64
    assert projection.projections[U_DOMAIN].source_dim == 4
    assert projection.projections[CHAIN_DOMAIN].source_dim == 6
    assert fused.required_obs_keys == ["front_img_1", "proprio_condition"]
    assert fused.encoder.obs_specs.proprio_condition.input_dim == 64
    assert noise.num_tokens == 16
    assert sampler.condition_input_dim == 128
    assert sampler.denoising_module.hidden_dim == 256
    assert sampler.denoising_module.nblocks == 12
    assert noise.latent_dim == sampler.latent_dim == 8
    assert sampler._target_.endswith("LatentFlowSampler")
    for decoder_field in (
        "action_horizon",
        "action_dims",
        "decoder_type",
        "decoder_hidden_dim",
        "latent_horizon",
    ):
        assert decoder_field not in sampler
    assert decoder._target_.endswith("PerEmbodimentActionDecoder")
    assert decoder.decoders[U_DOMAIN]._target_.endswith(
        "TokenwiseMLPActionDecoder"
    )
    assert decoder.decoders[CHAIN_DOMAIN]._target_.endswith(
        "TokenwiseMLPActionDecoder"
    )
    assert decoder.decoders[U_DOMAIN].action_dim == 4
    assert decoder.decoders[CHAIN_DOMAIN].action_dim == 6
    assert decoder.decoders[U_DOMAIN].hidden_dim == 32
    assert decoder.decoders[CHAIN_DOMAIN].hidden_dim == 32
    assert decoder.decoders[U_DOMAIN].num_layers == 3
    assert decoder.decoders[CHAIN_DOMAIN].num_layers == 3
    assert "extra_hidden_layers" not in decoder.decoders[U_DOMAIN]
    assert "extra_hidden_layers" not in decoder.decoders[CHAIN_DOMAIN]

    assert set(model.rollout_observation_adapters) == {U_DOMAIN, CHAIN_DOMAIN}
    for split_name in ("train_datasets", "valid_datasets"):
        split = cfg.data[split_name]
        u = split[U_DOMAIN].resolver
        chain = split[CHAIN_DOMAIN].resolver
        assert u.key_map._target_.endswith("get_keymap_hpt_per_emb_proprio")
        assert chain.key_map._target_.endswith("get_keymap_hpt_per_emb_proprio")
        assert u.transform_list._target_.endswith(
            "get_usocket_rotvec_action_state_transform_list"
        )
        assert chain.transform_list._target_.endswith(
            "get_chain_gripper_point_transform_list"
        )

    stages = instantiate(model.stages)
    train_runnable, train_excluded = _plan(stages, mode="train")
    rollout_runnable, rollout_excluded = _plan(stages, mode="rollout")
    assert train_runnable == [
        "EmbodimentProprioProjection",
        "FusedObsEncoder",
        "GaussianLatentNoise",
        "LatentFlowSampler",
        "PerEmbodimentActionDecoder",
        "NativeActionMSELoss",
    ]
    assert train_excluded == []
    assert rollout_runnable == train_runnable[:-1]
    assert rollout_excluded == ["NativeActionMSELoss"]


def _plan(stages, mode: str):
    from egomimic.pipeline.core import Pipeline

    seed = [
        "obs/state_agent_model",
        "obs/front_img_1",
        "embodiment",
    ]
    if mode == "train":
        seed.append("actions")
    runnable, excluded = Pipeline(list(stages)).plan(seed, mode=mode)
    return (
        [type(stage).__name__ for stage in runnable],
        [type(stage).__name__ for stage, _missing in excluded],
    )
